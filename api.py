# api.py — SobatNavi AI Agent v9.0
# ============================================================
# PERUBAHAN UTAMA dari v8:
#   1. RADIUS & ANCHOR-FIRST: DBSCAN diganti dengan metode clustering
#      berbasis anchor per hari yang geographically tight
#   2. AI MEAL SCHEDULING: Restoran kini dikirim dalam pool per hari
#      dan AI (Heidi) yang menjadwalkan di waktu makan yang tepat
#   3. HOTEL GUARANTEE: Tidak berubah dari v8
#   4. TOPSIS RANKING: Tetap digunakan untuk scoring internal
#   5. ANTI-DUPLICATION: Set-based dedup mencegah POI/restoran
#      muncul berulang lintas hari
# ============================================================

import uvicorn
import asyncio
import os
import json
import logging
import math
import re
import uuid
from fastapi import FastAPI, HTTPException, Depends, Security, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict, Optional, Literal
from datetime import datetime, timedelta

from app.schemas.response_schema import BaseHotel, FinalAIResponse, ItinerarySummary, RouteSegment, DailyItinerary, PlaceItem, BudgetBreakdown
from app.services.supabase_service import supabase_service
from app.services.tomtom_service import tomtom_service
from app.services.live_intel_service import live_intel_service
from app.engine.odalan_checker import extract_global_avoid_zones
from app.engine.recommender import generate_clustered_pool_delivery, rank_pois_by_topsis
from app.core.config import settings
from app.core.rate_limiter import (
    limiter,
    RATE_CHAT,
    RATE_ITINERARY_WRITE,
    RATE_ITINERARY_READ,
    RATE_SEARCH,
    RATE_HEALTH,
)

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ============================================================
# SECURITY HEADERS MIDDLEWARE
# ============================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects security-related HTTP response headers on every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response


# ============================================================
# FASTAPI APP + MIDDLEWARE STACK
# ============================================================

app = FastAPI(
    title="SobatNavi AI Agent API",
    version="8.0",
    description="AI Travel Assistant API untuk Bali — Heidi",
    # Do not expose internal server errors in the OpenAPI /docs error examples
    docs_url="/docs" if os.getenv("ENVIRONMENT", "production") != "production" else None,
    redoc_url=None,
)

# Attach slowapi limiter to the app state and register the 429 handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# TrustedHost — reject requests with unexpected Host headers (reverse-proxy safety)
_allowed_hosts = settings.allowed_hosts_list
if _allowed_hosts and _allowed_hosts != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts)

# CORS — restrict cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    max_age=600,
)

# Security headers on every response
app.add_middleware(SecurityHeadersMiddleware)

if not settings.openai_api_key:
    logger.warning(
        "OPENAI_API_KEY belum diset di .env! "
        "Endpoint /api/chat akan gagal saat dipanggil."
    )

_openai_client = None

def _get_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY belum dikonfigurasi di .env")
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client

security = HTTPBearer(auto_error=False)


# ============================================================
# AUTH MIDDLEWARE
# ============================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    if not credentials:
        raise HTTPException(status_code=401, detail="Token autentikasi diperlukan.")
    token = credentials.credentials
    try:
        user_response = await asyncio.to_thread(
            supabase_service.client.auth.get_user, token
        )
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Token tidak valid atau sudah kedaluwarsa.")
        return user_response.user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Autentikasi gagal: {str(e)}")


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    if not credentials or not credentials.credentials:
        return None
    try:
        user_response = await asyncio.to_thread(
            supabase_service.client.auth.get_user,
            credentials.credentials
        )
        if user_response and user_response.user:
            return user_response.user
        return None
    except Exception:
        return None


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    # message capped at 4000 chars to prevent prompt injection and token abuse
    message: str = Field(..., description="Pesan user ke Heidi", max_length=4000)
    mode: Literal["general", "deep_research"] = Field("general")
    budget_preference: Literal["budget", "moderate", "luxury"] = Field(
        "moderate",
        description="Preferensi anggaran perjalanan: 'budget' (hemat), 'moderate' (menengah), 'luxury' (mewah)."
    )
    session_id: Optional[str] = Field(None, max_length=128)
    # history capped at 50 items to prevent memory/token exhaustion
    history: Optional[List[Dict]] = Field(default_factory=list, max_length=50)
    current_itinerary: Optional[Dict] = Field(None)
    itinerary_id: Optional[str] = Field(None, max_length=128)

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Pesan tidak boleh kosong.")
        return stripped

    @field_validator("history", mode="before")
    @classmethod
    def history_max_items(cls, v):
        if v and len(v) > 50:
            # Silently truncate to the last 50 messages (keep recent context)
            return v[-50:]
        return v


class UpdateItineraryRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    itinerary_data: Dict = Field(...)
    total_budget_idr: Optional[int] = Field(None, ge=0)
    is_public: Optional[bool] = Field(None)


class ToggleVisibilityRequest(BaseModel):
    is_public: bool = Field(...)


# ============================================================
# POI BUDGET CALCULATOR
# ============================================================
# Logika utama untuk menentukan berapa banyak POI yang masuk akal
# per hari berdasarkan pace (kecepatan), durasi, dan durasi hari itu sendiri.

PACE_CONFIG = {
    # pace_keyword → (min_poi, ideal_poi, max_poi)
    "santai":  (2, 3, 4),   # Makan + 2-3 atraksi, banyak istirahat
    "normal":  (3, 4, 5),   # Default: 3-4 atraksi + makan siang + makan malam
    "padat":   (4, 5, 7),   # Maksimalkan kunjungan, jarang berhenti lama
    "custom":  (None, None, None),  # Diatur manual oleh user
}

def calculate_poi_budget(
    num_days: int,
    pace: str = "normal",
    user_requested_pois: dict = None,
    is_half_day: bool = False,
) -> dict:
    """Menghitung 'anggaran' POI per hari secara programatik."""
    was_capped = False
    max_allowed_pois = 7

    # 1. Determine base_default using pace logic
    if is_half_day:
        base_default = 2
    else:
        pace_key = pace.lower() if pace.lower() in PACE_CONFIG else "normal"
        min_poi, ideal_poi, max_poi = PACE_CONFIG[pace_key]
        if num_days >= 5:
            ideal_poi = max(min_poi, ideal_poi - 1)
        base_default = ideal_poi

    # 2. Extract custom_days and check for default_count override
    custom_days = {}
    if isinstance(user_requested_pois, dict):
        custom_days = user_requested_pois.get("custom_days") or {}
        default_count = user_requested_pois.get("default_count")
        if default_count is not None:
            try:
                base_default = int(default_count)
            except (ValueError, TypeError):
                pass

    # 3. Generate full_targets mapping every day (from 1 to num_days) to an explicit integer
    full_targets = {}
    for d in range(1, num_days + 1):
        target = base_default
        day_str = str(d)
        if day_str in custom_days:
            try:
                target = int(custom_days[day_str])
            except (ValueError, TypeError):
                pass
        
        # Clamp between 1 and 7
        if target > max_allowed_pois:
            target = max_allowed_pois
            was_capped = True
        elif target < 1:
            target = 1
        full_targets[day_str] = target

    daily_targets_str = json.dumps(full_targets)
    avg_attractions = int(sum(full_targets.values()) / len(full_targets)) if full_targets else base_default

    return {
        "attractions_per_day": avg_attractions,
        "meals_per_day": 1 if is_half_day else (2 if avg_attractions >= 4 else 1),
        "pace_label": "setengah hari" if is_half_day else (f"custom ({avg_attractions} tempat)" if isinstance(user_requested_pois, dict) else pace),
        "daily_targets": daily_targets_str,
        "was_capped": was_capped,
        "full_targets": full_targets
    }



def extract_number_of_days(durasi_str: str) -> Optional[int]:
    if not durasi_str:
        return None
    match = re.search(r"(\d+)", str(durasi_str))
    if match:
        val = int(match.group(1))
        return max(1, min(val, 14))
    return None


async def check_deep_research_variables(message: str, history: list[dict]) -> dict:
    """
    Menganalisis apakah 6 variabel penting sudah terisi dari chat history dan pesan saat ini.
    """
    try:
        client = _get_client()
        
        # Serialize history into user/assistant conversation string
        history_str = ""
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                history_str += f"{role.upper()}: {content}\n"
        
        system_prompt = (
            "You are an information extraction assistant. Analyze the user's travel planning request and conversation history.\n"
            "Your task is to check if the following 6 variables are provided/filled by the user:\n"
            "1. lokasi (e.g., 'Ubud', 'Kuta', 'Seminyak')\n"
            "2. tanggal (e.g., 'Besok', '12 Agustus', '2026-08-12')\n"
            "3. durasi (e.g., '3 hari', '3 days', '1 minggu')\n"
            "4. budget (e.g., 'Hemat', 'Menengah', 'Mewah', 'Budget', 'Luxury')\n"
            "5. teman_perjalanan (e.g., 'Sendiri', 'Keluarga', 'Teman', 'Pasangan')\n"
            "6. pace (e.g., 'Santai', 'Padat', 'Normal')\n\n"
            "Return a JSON object with this exact structure:\n"
            "{\n"
            "  \"variables\": {\n"
            "    \"lokasi\": string or null,\n"
            "    \"tanggal\": string or null,\n"
            "    \"durasi\": string or null,\n"
            "    \"budget\": string or null,\n"
            "    \"teman_perjalanan\": string or null,\n"
            "    \"pace\": string or null\n"
            "  },\n"
            "  \"all_filled\": boolean,\n"
            "  \"missing_variables\": [\"lokasi\", \"tanggal\", ...]\n"
            "}\n\n"
            "Rules:\n"
            "- If a variable is mentioned in either the current message or the history, extract it.\n"
            "- Do not guess or assume. If it is not clearly specified, set it to null.\n"
            "- 'all_filled' should be true ONLY if all 6 variables are non-null."
        )
        
        user_content = f"History:\n{history_str}\n\nCurrent Message: {message}"
        
        response = await client.chat.completions.create(
            model=settings.openai_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=250
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Gagal mengekstrak variabel deep research: {e}")
        return {
            "variables": {
                "lokasi": None,
                "tanggal": None,
                "durasi": None,
                "budget": None,
                "teman_perjalanan": None,
                "pace": None
            },
            "all_filled": False,
            "missing_variables": ["lokasi", "tanggal", "durasi", "budget", "teman_perjalanan", "pace"]
        }


async def extract_trip_parameters_from_message(message: str, db_districts: list[str]) -> dict:
    try:
        logger.info(">>> MENJALANKAN EXTRACT PARAMS <<<")
        client = _get_client()
        prompt = (
            "Analyze the travel planning message and return a JSON object containing exactly these fields:\n"
            "- intent: strictly one of 'add_place', 'delete_place', 'swap_place', 'create', or 'chat'.\n"
            "   * Choose 'add_place' if the user explicitly wants to add, insert, or include places/activities.\n"
            "   * Choose 'delete_place' if the user explicitly wants to delete, remove, or exclude places/activities.\n"
            "   * Choose 'swap_place' if the user explicitly wants to swap, replace, or exchange one place/day with another.\n"
            "   * Choose 'create' if the user wants to plan a new itinerary.\n"
            "   * Choose 'chat' for general greetings or questions.\n"
            "- pace: string, one of 'santai', 'padat', 'normal'.\n"
            "- is_half_day: boolean, true if half-day trip.\n"
            "- user_requested_pois: object containing 'custom_days' (map of day string to integer) AND 'default_count' (integer). E.g., if user says 'day 1 needs 1 place, other days 3 places', return {\"custom_days\": {\"1\": 1}, \"default_count\": 3}. If '4 places per day', return {\"custom_days\": {}, \"default_count\": 4}. Return null if completely unspecified.\n"
            "- detected_location: string or null.\n"
            "- normalized_districts: list of strings matching this allowed list:\n"
            f"{db_districts}\n\n"
            "Return ONLY a raw JSON object."
        )
        response = await client.chat.completions.create(
            model=settings.openai_model_id,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=150
        )
        data = json.loads(response.choices[0].message.content)
        
        intent = str(data.get("intent", "chat")).lower()
        pace = str(data.get("pace", "normal")).lower()
        is_half_day = bool(data.get("is_half_day", False))
        
        user_requested_pois = data.get("user_requested_pois")
        # user_requested_pois can now be an int, a dict, or None
                
        detected_location = data.get("detected_location")
        if detected_location: detected_location = str(detected_location).strip()
            
        normalized_districts = data.get("normalized_districts", [])
        if not isinstance(normalized_districts, list): normalized_districts = []
        valid_districts = [d for d in normalized_districts if d in db_districts]

    except Exception as e:
        logger.warning(f"Gagal LLM extract parameters: {e}")
        intent = "chat"
        pace = "normal"
        is_half_day = False
        user_requested_pois = None
        detected_location = None
        valid_districts = []

    # ============================================================
    # 🛡️ JARING PENGAMAN (OVERRIDE MUTLAK)
    # ============================================================
    # Jika LLM gagal paham, tapi user jelas-jelas bilang "hapus/tambah/tukar",
    # kita paksa intent sesuai kata kunci.
    msg_lower = message.lower()
    if any(k in msg_lower for k in ["tukar", "swap", "swapping", "ganti", "replace", "switch"]):
        logger.info(f"Hybrid Override: Kata kunci swap terdeteksi, memaksa intent='swap_place'")
        intent = "swap_place"
    elif any(k in msg_lower for k in ["hapus", "delete", "remove", "keluarkan", "buang"]):
        logger.info(f"Hybrid Override: Kata kunci hapus terdeteksi, memaksa intent='delete_place'")
        intent = "delete_place"
    elif any(k in msg_lower for k in ["tambah", "tambhakan", "tambahkan", "add", "insert", "inject", "masukkan"]):
        logger.info(f"Hybrid Override: Kata kunci tambah terdeteksi, memaksa intent='add_place'")
        intent = "add_place"
        
    if intent not in ["add_place", "delete_place", "swap_place", "create", "chat"]:
        intent = "chat"
        
    # Jaringan pengaman lokasi: Jika LLM gagal mendeteksi lokasi tapi ada nama distrik Bali di pesan
    if not detected_location:
        bali_districts = [
            "ubud", "kuta", "seminyak", "canggu", "sanur", "nusa dua",
            "jimbaran", "uluwatu", "tabanan", "singaraja", "lovina",
            "amed", "padangbai", "candidasa", "denpasar", "legian",
            "nusa penida", "nusa lembongan", "buleleng", "gianyar",
            "klungkung", "karangasem", "bangli",
        ]
        for d in bali_districts:
            if d in msg_lower:
                logger.info(f"Hybrid Override: Distrik terdeteksi dari pesan: '{d}'")
                detected_location = d.title()
                break

    if detected_location and not valid_districts:
        det_low = detected_location.lower()
        matches = [db_d for db_d in db_districts if det_low in db_d.lower() or db_d.lower() in det_low]
        if matches:
            valid_districts = matches
    # ============================================================

    return {
        "intent": intent,
        "pace": pace if pace in ["santai", "normal", "padat"] else "normal",
        "is_half_day": is_half_day,
        "user_requested_pois": user_requested_pois,
        "detected_location": detected_location,
        "normalized_districts": valid_districts
    }


# ============================================================
# ITINERARY EDIT UTILITIES
# ============================================================

def add_minutes_to_time(time_str: str, minutes: int) -> str:
    """Tambahkan durasi menit ke string waktu HH:MM."""
    try:
        if not time_str:
            return "18:00"
        t = datetime.strptime(time_str, "%H:%M")
        t_new = t + timedelta(minutes=minutes)
        return t_new.strftime("%H:%M")
    except Exception:
        return "18:00"


def calculate_day_centroid(day_places: list, base_hotel: dict = None) -> tuple[float, float]:
    """Hitung rata-rata koordinat (sentroid) dari daftar tempat pada suatu hari."""
    coords = []
    for p in day_places:
        lat = p.get("latitude")
        lng = p.get("longitude")
        if lat is not None and lng is not None:
            coords.append((float(lat), float(lng)))
            
    if coords:
        lat_avg = sum(c[0] for c in coords) / len(coords)
        lng_avg = sum(c[1] for c in coords) / len(coords)
        return lat_avg, lng_avg
        
    if base_hotel:
        h_lat = base_hotel.get("latitude")
        h_lng = base_hotel.get("longitude")
        if h_lat is not None and h_lng is not None:
            return float(h_lat), float(h_lng)
            
    return -8.4095, 115.1889  # Default: Bali Tengah


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Hitung jarak geografis antara dua titik dalam kilometer menggunakan Haversine."""
    R = 6371.0  # Radius bumi dalam km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def recalculate_itinerary_budget(itinerary: dict) -> dict:
    """Hitung ulang total anggaran dan rincian budget berdasarkan daftar tempat saat ini."""
    days = itinerary.get("itinerary_days", [])
    num_days = len(days)
    
    # 1. Akomodasi: harga kamar per malam * jumlah hari (asumsi hari = malam)
    hotel = itinerary.get("base_hotel")
    hotel_price = 0
    if hotel:
        hotel_price = hotel.get("price_per_night_idr") or 0
    accommodation_idr = hotel_price * max(1, num_days)
    
    # 2. Tiket Masuk & Makanan dari tempat-tempat di itinerary
    entrance_fee_idr = 0
    food_idr = 0
    for day in days:
        places_list = day.get("places") or []
        for place in places_list:
            # handle place as dictionary or Pydantic model
            if hasattr(place, "estimated_cost_idr"):
                cost = getattr(place, "estimated_cost_idr") or 0
                cat = getattr(place, "category") or "attraction"
            else:
                cost = place.get("estimated_cost_idr") or 0
                cat = place.get("category") or "attraction"
                
            if cat == "restaurant":
                food_idr += cost
            else:
                entrance_fee_idr += cost
                
    # 3. Transportasi: default Rp 150.000 per hari
    transport_idr = 150000 * num_days
    
    # 4. Lain-lain: 10% dari total komponen di atas
    subtotal = accommodation_idr + food_idr + transport_idr + entrance_fee_idr
    miscellaneous_idr = int(subtotal * 0.1)
    
    total_budget_idr = subtotal + miscellaneous_idr
    
    itinerary["total_budget_idr"] = total_budget_idr
    itinerary["budget_breakdown"] = {
        "accommodation_idr": accommodation_idr,
        "food_idr": food_idr,
        "transport_idr": transport_idr,
        "entrance_fee_idr": entrance_fee_idr,
        "miscellaneous_idr": miscellaneous_idr
    }
    return itinerary


async def extract_edit_details(message: str, itinerary: dict) -> dict:
    """Menggunakan LLM untuk mengekstrak tempat-tempat yang ingin ditambahkan."""
    client = _get_client()
    days_summary = []
    for day in itinerary.get("itinerary_days", []):
        place_names = []
        for p in day.get("places", []):
            if hasattr(p, "name"):
                place_names.append(getattr(p, "name"))
            elif isinstance(p, dict):
                place_names.append(p.get("name"))
        days_summary.append(f"Day {day.get('day')}: {', '.join(place_names)}")
    itinerary_context = "\n".join(days_summary)

    prompt = (
        "Analyze the user's travel planning request to add places to their itinerary.\n"
        f"Here is the current itinerary:\n{itinerary_context}\n\n"
        "Extract the places the user wants to add. For each place, determine:\n"
        "1. 'query': The place name, type, or description to search for (e.g. 'Tegenungan Waterfall', 'air terjun', 'mall', 'apa aja').\n"
        "2. 'is_specific': Boolean. True if it's a specific named place (like 'Tanah Lot', 'Waterbom Bali', 'Pura Ulun Danu Beratan'). False if it's a generic description, type, category, or vibe (like 'air terjun', 'mall', 'pantai sunset', 'tempat apa aja').\n"
        "3. 'day': Integer (1-based index). The day to add it to. If the user does not specify a day or it is ambiguous/not mentioned, set it to null.\n"
        "4. 'count': Integer. The number of places of this type to add. Defaults to 1 if not specified.\n\n"
        "Format the response as a JSON object with this schema:\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"query\": \"string\",\n"
        "      \"is_specific\": boolean,\n"
        "      \"day\": integer or null,\n"
        "      \"count\": integer\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Return ONLY the raw JSON object."
    )

    response = await client.chat.completions.create(
        model=settings.openai_model_id,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Failed to parse extract_edit_details JSON: {e}")
        return {"items": []}


async def pinpoint_places_to_delete(message: str, itinerary: dict) -> dict:
    """Menggunakan LLM untuk mengidentifikasi indeks tempat yang ingin dihapus."""
    client = _get_client()
    places_list = []
    for day in itinerary.get("itinerary_days", []):
        day_num = day.get("day")
        places_in_day = day.get("places") or []
        for idx, p in enumerate(places_in_day):
            if hasattr(p, "name"):
                name = getattr(p, "name")
                category = getattr(p, "category", "")
                description = getattr(p, "description", "")
                place_id = getattr(p, "place_id", "")
            else:
                name = p.get("name")
                category = p.get("category", "")
                description = p.get("description", "")
                place_id = p.get("place_id", "")

            places_list.append({
                "day": day_num,
                "index": idx,
                "place_id": place_id,
                "name": name,
                "category": category,
                "description": description
            })

    prompt = (
        "You are an expert travel assistant. The user wants to delete a place or multiple places from their itinerary.\n"
        f"User message: '{message}'\n\n"
        f"Here is the list of places currently in the itinerary:\n{json.dumps(places_list, indent=2)}\n\n"
        "Identify which place(s) the user wants to delete based on their message. "
        "Return the day and index of each place to delete.\n"
        "Format the response as a JSON object with this schema:\n"
        "{\n"
        "  \"delete_indices\": [\n"
        "    {\n"
        "      \"day\": integer,\n"
        "      \"index\": integer\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "If no place matches the user's request, return an empty list for 'delete_indices'.\n"
        "Return ONLY the raw JSON object."
    )

    response = await client.chat.completions.create(
        model=settings.openai_model_id,
        messages=[
            {"role": "system", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Failed to parse pinpoint_places_to_delete JSON: {e}")
        return {"delete_indices": []}


async def generate_edit_confirmation_message(
    message: str,
    original_itinerary: dict,
    modified_itinerary: dict,
    change_summary: str
) -> dict:
    """Menggunakan LLM untuk menghasilkan narasi konfirmasi perubahan itinerary yang ramah dan alami."""
    client = _get_client()
    prompt = (
        "You are Heidi, the AI travel assistant for Bali.\n"
        "The user asked to edit their itinerary. The edit was successfully executed.\n"
        f"User edit request: '{message}'\n"
        f"Summary of changes made: {change_summary}\n\n"
        "Please write a warm, friendly response message (in Indonesian) to the user confirming the change. "
        "Briefly explain what was added or deleted and why, and confirm their itinerary has been updated. "
        "Always use rich markdown formatting (bold, italic, headings, lists, emojis).\n\n"
        "Return a JSON object with two fields:\n"
        "- 'message_to_user': The markdown message confirming the changes.\n"
        "- 'suggested_replies': A list of 3 relevant follow-up questions/suggestions (e.g. asking to adjust times, add more places, check route, etc.).\n"
        "Return ONLY the raw JSON object."
    )
    
    response = await client.chat.completions.create(
        model=settings.openai_model_id,
        messages=[
            {"role": "system", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Failed to parse generate_edit_confirmation_message JSON: {e}")
        return {
            "message_to_user": f"Jadwal kamu telah diperbarui: {change_summary}",
            "suggested_replies": ["Tampilkan rute terbaru", "Sesuaikan waktu kunjungan", "Saran tempat lainnya"]
        }


async def execute_itinerary_edit(message: str, itinerary: dict, intent: str) -> FinalAIResponse:
    import copy
    
    # 1. Clone itinerary to ensure immutability
    edited_itinerary = copy.deepcopy(itinerary)
    itinerary_days = edited_itinerary.get("itinerary_days", [])
    if not itinerary_days:
        return FinalAIResponse(
            response_type="chat",
            message_to_user="Maaf, itinerary Anda kosong atau tidak valid.",
            suggested_replies=["Buatkan itinerary 3 hari"],
            itinerary_days=None,
            base_hotel=None,
            trip_title=None
        )
        
    change_summary_parts = []
    
    # ==========================================
    # FLOW: ADD PLACE
    # ==========================================
    if intent == "add_place":
        extracted = await extract_edit_details(message, itinerary)
        items_to_add = extracted.get("items", [])
        if not items_to_add:
            return FinalAIResponse(
                response_type="chat",
                message_to_user="Aku tidak mengerti tempat apa yang ingin kamu tambahkan. Bisa tolong sebutkan secara jelas? 😊",
                suggested_replies=["Tambahkan Pantai Pandawa", "Tambahkan air terjun di Hari 1"],
                itinerary_days=None,
                base_hotel=None,
                trip_title=None
            )
            
        # 1. Edge Case: Check if any item does not specify a day
        for item in items_to_add:
            if item.get("day") is None:
                q = item.get("query", "tempat")
                return FinalAIResponse(
                    response_type="chat",
                    message_to_user=f"Di hari ke berapa kamu ingin menambahkan '{q}'? Silakan sebutkan nomor hari (misalnya: Hari 1) agar aku bisa menempatkannya dengan tepat. 😊",
                    suggested_replies=[f"Tambahkan di Hari 1", f"Tambahkan di Hari 2"],
                    itinerary_days=None,
                    base_hotel=None,
                    trip_title=None
                )
                
        # Existing place IDs/names for duplication filtering
        existing_place_ids = set()
        for day in itinerary_days:
            for place in day.get("places", []):
                pid = place.get("place_id")
                if pid:
                    existing_place_ids.add(pid)
                name = place.get("name")
                if name:
                    existing_place_ids.add(name.lower())
                    
        # Process additions
        for item in items_to_add:
            query = item.get("query")
            is_specific = item.get("is_specific", False)
            day_num = item.get("day")
            count = max(1, min(item.get("count", 1), 5)) # Clamp count
            
            # Find the corresponding day object
            day_obj = next((d for d in itinerary_days if d.get("day") == day_num), None)
            if not day_obj:
                return FinalAIResponse(
                    response_type="chat",
                    message_to_user=f"Maaf, Hari {day_num} tidak ada dalam itinerary kamu saat ini.",
                    suggested_replies=["Tampilkan itinerary", "Tambahkan tempat di Hari 1"],
                    itinerary_days=None,
                    base_hotel=None,
                    trip_title=None
                )
                
            # Get centroid coordinates for the day
            centroid_lat, centroid_lng = calculate_day_centroid(day_obj.get("places", []), edited_itinerary.get("base_hotel"))
            
            # Sub-flow: Specific Place
            if is_specific:
                results = await supabase_service.search_specific_place(query, category="attraction")
                if not results:
                    results = await supabase_service.search_specific_place(query, category="restaurant")
                    
                if not results:
                    return FinalAIResponse(
                        response_type="chat",
                        message_to_user=f"Maaf, tempat '{query}' tidak ditemukan di database kami. Kamu bisa mencoba mencari tempat lain, atau memasukkannya secara manual.",
                        suggested_replies=["Cari alternatif lainnya", "Tambahkan tempat lain"],
                        itinerary_days=None,
                        base_hotel=None,
                        trip_title=None
                    )
                    
                # Pick the first result as the best match
                place_data = results[0]
                place_id = place_data.get("place_id")
                
                # Check for duplication
                if place_id in existing_place_ids or place_data.get("name", "").lower() in existing_place_ids:
                    return FinalAIResponse(
                        response_type="chat",
                        message_to_user=f"Tempat '{place_data.get('name')}' sudah ada dalam itinerary kamu.",
                        suggested_replies=["Tambahkan tempat lain", "Lihat itinerary"],
                        itinerary_days=None,
                        base_hotel=None,
                        trip_title=None
                    )
                    
                is_resto = "price_level" in place_data or place_data.get("category") == "restaurant"
                cat_canonical = "restaurant" if is_resto else "attraction"
                
                # Build PlaceItem
                new_place = {
                    "place_id": place_id,
                    "poi_id": str(place_data.get("id")) if place_data.get("id") else None,
                    "name": place_data.get("name"),
                    "category": cat_canonical,
                    "description": place_data.get("content") or f"Kunjungan menarik ke {place_data.get('name')}.",
                    "latitude": place_data.get("latitude"),
                    "longitude": place_data.get("longitude"),
                    "district": place_data.get("district"),
                    "image_url": place_data.get("image_url"),
                    "rating": place_data.get("rating"),
                    "user_rating_count": place_data.get("user_rating_count"),
                    "estimated_cost_idr": 75000 if is_resto else 25000,
                    "tags": ["wisata", "bali"],
                    "visit_duration_mins": 60 if is_resto else 75,
                    "visit_time": None,
                    "tips": f"Selamat menikmati perjalanan Anda di {place_data.get('name')}."
                }
                
                # Calculate visit time
                places_in_day = day_obj.get("places", [])
                if places_in_day:
                    last_p = places_in_day[-1]
                    # handle dictionary vs Pydantic object
                    last_visit_time = last_p.get("visit_time") if isinstance(last_p, dict) else getattr(last_p, "visit_time")
                    last_duration = last_p.get("visit_duration_mins") if isinstance(last_p, dict) else getattr(last_p, "visit_duration_mins")
                    new_place["visit_time"] = add_minutes_to_time(last_visit_time, (last_duration or 60) + 30)
                else:
                    new_place["visit_time"] = "08:00"
                    
                day_obj["places"].append(new_place)
                existing_place_ids.add(place_id)
                change_summary_parts.append(f"menambahkan '{place_data.get('name')}' di Hari {day_num}")
                
            # Sub-flow: Generic Place (e.g. "air terjun", "mall", or "apa aja")
            else:
                is_generic_random = query.strip().lower() in ["apa aja", "apa saja", "mana aja", "bebas", "random", "tempat apa aja", "terserah", "whatever", "tempat apa saja"]
                
                added_count = 0
                if is_generic_random:
                    nearby_results = await supabase_service.search_pois_nearby(centroid_lat, centroid_lng, radius_m=20000, limit=20)
                    valid_nearby = []
                    for r in nearby_results:
                        pid = r.get("place_id")
                        name = r.get("name", "").lower()
                        if pid not in existing_place_ids and name not in existing_place_ids:
                            valid_nearby.append(r)
                            
                    for i in range(min(count, len(valid_nearby))):
                        place_data = valid_nearby[i]
                        place_id = place_data.get("place_id")
                        new_place = {
                            "place_id": place_id,
                            "poi_id": str(place_data.get("id")) if place_data.get("id") else None,
                            "name": place_data.get("name"),
                            "category": "attraction",
                            "description": place_data.get("content") or f"Menikmati suasana indah di {place_data.get('name')}.",
                            "latitude": place_data.get("latitude"),
                            "longitude": place_data.get("longitude"),
                            "district": place_data.get("district"),
                            "image_url": place_data.get("image_url"),
                            "rating": place_data.get("rating"),
                            "user_rating_count": place_data.get("user_rating_count"),
                            "estimated_cost_idr": 25000,
                            "tags": ["wisata", "bali"],
                            "visit_duration_mins": 75,
                            "visit_time": None,
                            "tips": "Nikmati keindahan tempat wisata terdekat ini."
                        }
                        
                        places_in_day = day_obj.get("places", [])
                        if places_in_day:
                            last_p = places_in_day[-1]
                            last_visit_time = last_p.get("visit_time") if isinstance(last_p, dict) else getattr(last_p, "visit_time")
                            last_duration = last_p.get("visit_duration_mins") if isinstance(last_p, dict) else getattr(last_p, "visit_duration_mins")
                            new_place["visit_time"] = add_minutes_to_time(last_visit_time, (last_duration or 60) + 30)
                        else:
                            new_place["visit_time"] = "08:00"
                            
                        day_obj["places"].append(new_place)
                        existing_place_ids.add(place_id)
                        added_count += 1
                        change_summary_parts.append(f"menambahkan '{place_data.get('name')}' di Hari {day_num}")
                else:
                    semantic_results = await supabase_service.search_pois_semantic(query, limit=50)
                    if not semantic_results:
                        return FinalAIResponse(
                            response_type="chat",
                            message_to_user=f"Maaf, tidak ditemukan tempat wisata yang cocok untuk pencarian '{query}' di database.",
                            suggested_replies=["Coba kata kunci lain", "Tambahkan tempat lain"],
                            itinerary_days=None,
                            base_hotel=None,
                            trip_title=None
                        )
                        
                    sorted_results = []
                    for r in semantic_results:
                        r_lat = r.get("latitude")
                        r_lng = r.get("longitude")
                        if r_lat is not None and r_lng is not None:
                            dist = haversine_distance(centroid_lat, centroid_lng, float(r_lat), float(r_lng))
                            sorted_results.append((dist, r))
                            
                    sorted_results.sort(key=lambda x: x[0])
                    
                    valid_semantic = []
                    for dist, r in sorted_results:
                        pid = r.get("place_id")
                        name = r.get("name", "").lower()
                        if pid not in existing_place_ids and name not in existing_place_ids:
                            valid_semantic.append(r)
                            
                    if not valid_semantic:
                        return FinalAIResponse(
                            response_type="chat",
                            message_to_user=f"Maaf, semua tempat wisata '{query}' terdekat sudah ada di itinerary kamu.",
                            suggested_replies=["Tambahkan kategori lain", "Lihat itinerary"],
                            itinerary_days=None,
                            base_hotel=None,
                            trip_title=None
                        )
                        
                    for i in range(min(count, len(valid_semantic))):
                        place_data = valid_semantic[i]
                        place_id = place_data.get("place_id")
                        new_place = {
                            "place_id": place_id,
                            "poi_id": str(place_data.get("id")) if place_data.get("id") else None,
                            "name": place_data.get("name"),
                            "category": "attraction",
                            "description": place_data.get("content") or f"Menjelajahi keindahan {place_data.get('name')}.",
                            "latitude": place_data.get("latitude"),
                            "longitude": place_data.get("longitude"),
                            "district": place_data.get("district"),
                            "image_url": place_data.get("image_url"),
                            "rating": place_data.get("rating"),
                            "user_rating_count": place_data.get("user_rating_count"),
                            "estimated_cost_idr": 25000,
                            "tags": ["wisata", "bali"],
                            "visit_duration_mins": 75,
                            "visit_time": None,
                            "tips": "Jangan lupa siapkan kamera untuk mengambil momen indah."
                        }
                        
                        places_in_day = day_obj.get("places", [])
                        if places_in_day:
                            last_p = places_in_day[-1]
                            last_visit_time = last_p.get("visit_time") if isinstance(last_p, dict) else getattr(last_p, "visit_time")
                            last_duration = last_p.get("visit_duration_mins") if isinstance(last_p, dict) else getattr(last_p, "visit_duration_mins")
                            new_place["visit_time"] = add_minutes_to_time(last_visit_time, (last_duration or 60) + 30)
                        else:
                            new_place["visit_time"] = "08:00"
                            
                        day_obj["places"].append(new_place)
                        existing_place_ids.add(place_id)
                        added_count += 1
                        change_summary_parts.append(f"menambahkan '{place_data.get('name')}' di Hari {day_num}")
                        
                if added_count == 0:
                    return FinalAIResponse(
                        response_type="chat",
                        message_to_user="Maaf, kami tidak berhasil menemukan tempat terdekat yang cocok untuk ditambahkan.",
                        suggested_replies=["Ubah kata kunci pencarian", "Lihat itinerary"],
                        itinerary_days=None,
                        base_hotel=None,
                        trip_title=None
                    )

    # ==========================================
    # FLOW: DELETE PLACE
    # ==========================================
    elif intent == "delete_place":
        pinpoint_result = await pinpoint_places_to_delete(message, itinerary)
        delete_indices = pinpoint_result.get("delete_indices", [])
        if not delete_indices:
            return FinalAIResponse(
                response_type="chat",
                message_to_user="Maaf, aku tidak menemukan tempat yang ingin kamu hapus di itinerary. Kamu bisa menghapusnya secara manual melalui editor. 😊",
                suggested_replies=["Hapus tempat lain", "Tampilkan itinerary"],
                itinerary_days=None,
                base_hotel=None,
                trip_title=None
            )
            
        deletes_by_day = {}
        for item in delete_indices:
            d = item.get("day")
            idx = item.get("index")
            if d not in deletes_by_day:
                deletes_by_day[d] = []
            deletes_by_day[d].append(idx)
            
        for d, idxs in deletes_by_day.items():
            day_obj = next((day for day in itinerary_days if day.get("day") == d), None)
            if day_obj and "places" in day_obj:
                for idx in sorted(idxs, reverse=True):
                    if 0 <= idx < len(day_obj["places"]):
                        p = day_obj["places"][idx]
                        p_name = p.get("name") if isinstance(p, dict) else getattr(p, "name", "Tempat")
                        day_obj["places"].pop(idx)
                        change_summary_parts.append(f"menghapus '{p_name}' di Hari {d}")
                        
        if not change_summary_parts:
            return FinalAIResponse(
                response_type="chat",
                message_to_user="Maaf, aku tidak menemukan tempat yang ingin kamu hapus di itinerary. Kamu bisa menghapusnya secara manual.",
                suggested_replies=["Tampilkan itinerary"],
                itinerary_days=None,
                base_hotel=None,
                trip_title=None
            )

    # 3. Recalculate routes fields as null so that backend TomTom routing runs
    for day in itinerary_days:
        day["day_total_distance_km"] = None
        day["day_total_travel_time_mins"] = None
        day["day_full_polyline"] = None
        day["route_from_hotel"] = None
        for p in day.get("places", []):
            if isinstance(p, dict):
                p["route_to_next"] = None
            else:
                setattr(p, "route_to_next", None)
                
    # 4. Recalculate budget
    edited_itinerary = recalculate_itinerary_budget(edited_itinerary)
    
    # 5. Generate confirmation narrative
    change_summary = ", ".join(change_summary_parts)
    confirmation = await generate_edit_confirmation_message(message, itinerary, edited_itinerary, change_summary)
    
    # 6. Parse into FinalAIResponse
    response_dict = {
        "response_type": "itinerary",
        "message_to_user": confirmation.get("message_to_user", f"Jadwal kamu telah diperbarui: {change_summary}."),
        "suggested_replies": confirmation.get("suggested_replies", ["Tampilkan rute terbaru", "Sesuaikan waktu kunjungan"]),
        "trip_title": edited_itinerary.get("trip_title"),
        "base_hotel": edited_itinerary.get("base_hotel"),
        "itinerary_days": edited_itinerary.get("itinerary_days"),
        "total_budget_idr": edited_itinerary.get("total_budget_idr"),
        "budget_breakdown": edited_itinerary.get("budget_breakdown"),
        "itinerary_id": edited_itinerary.get("itinerary_id")
    }
    
    return FinalAIResponse.model_validate(response_dict)


# ============================================================
# MEAL SLOT DEFINITIONS (DEPRECATED v9.0 — kept for reference)
# ============================================================
# Meal scheduling is now handled by Heidi AI using the restaurant
# pool data from generate_clustered_pool_delivery.
# The backend no longer injects restaurants programmatically.

# MEAL_SLOTS = {
#     "sarapan": {
#         "visit_time": "08:00",
#         "visit_duration_mins": 45,
#         "estimated_cost_idr": 50000,
#         "tags": ["sarapan", "kuliner", "pagi"],
#         "label": "☕ Sarapan",
#     },
#     "makan_siang": {
#         "visit_time": "12:30",
#         "visit_duration_mins": 60,
#         "estimated_cost_idr": 75000,
#         "tags": ["makan siang", "kuliner", "restoran"],
#         "label": "🍽️ Makan Siang",
#     },
#     "makan_malam": {
#         "visit_time": "19:00",
#         "visit_duration_mins": 75,
#         "estimated_cost_idr": 100000,
#         "tags": ["makan malam", "kuliner", "dinner"],
#         "label": "🌙 Makan Malam",
#     },
# }

# def assign_meal_slots(meals_per_day: int) -> list[str]:
#     if meals_per_day <= 0:
#         return []
#     elif meals_per_day == 1:
#         return ["makan_siang"]
#     elif meals_per_day == 2:
#         return ["makan_siang", "makan_malam"]
#     else:
#         return ["sarapan", "makan_siang", "makan_malam"]


# ============================================================
# AI TOOL FUNCTIONS
# ============================================================

async def get_bali_context(date_start: str, date_end: str, district: str = "Bali") -> dict:
    """Ambil info cuaca & daftar zona hindari Odalan untuk tanggal perjalanan."""
    weather = await live_intel_service.search_tavily(
        f"cuaca dan kondisi jalan di {district}, Bali pada tanggal {date_start}"
    )
    active_odalans = await supabase_service.get_all_active_odalans(date_start, date_end)
    avoid_zones = extract_global_avoid_zones(active_odalans)
    return {
        "weather_info": weather,
        "active_odalans_count": len(active_odalans),
        "avoid_zones": avoid_zones,
        "avoid_zones_info": "Daftar bbox zona hindari Odalan yang aktif pada rentang tanggal tersebut"
    }


async def get_smart_recommendations(
    query: str,
    num_days: int = 1,
    category: str = "poi",
    preference_mode: str = "standard",
    user_detected_location: str = None,
    user_requested_pois: Optional[dict] = None,
) -> list[dict]:
    """
    Cari tempat wisata menggunakan Radius & Anchor-First clustering.

    Untuk category='poi':
      Menggunakan generate_clustered_pool_delivery untuk menghasilkan
      pool POI + restoran per hari yang geographically tight.

    Untuk category='hotel'/'restaurant':
      Menggunakan semantic search + TOPSIS ranking (tanpa spatial clustering).
    """
    if category == "hotel":
        raw = await supabase_service.search_amenities_semantic(query, "hotel", limit=40)
        return rank_pois_by_topsis(raw, category="hotel", preference_mode=preference_mode, top_n=10)
    elif category == "restaurant":
        raw = await supabase_service.search_amenities_semantic(query, "restaurant", limit=40)
        return rank_pois_by_topsis(raw, category="restaurant", preference_mode=preference_mode, top_n=10)
    else:
        # POI: Radius & Anchor-First clustering
        return await generate_clustered_pool_delivery(
            supabase_service=supabase_service,
            query=query,
            num_days=num_days,
            user_detected_location=user_detected_location,
            preference_mode=preference_mode,
            user_requested_pois=user_requested_pois,
        )


async def get_general_recommendations(
    query: str,
    category: str = "poi",
    limit: int = 5,
) -> list[dict]:
    """
    Gunakan tool ini HANYA untuk rekomendasi tempat umum tanpa membuat itinerary.
    """
    if category == "hotel":
        raw = await supabase_service.search_amenities_semantic(query, "hotel", limit=limit * 4)
        return rank_pois_by_topsis(raw, category="hotel", preference_mode="standard", top_n=limit)
    elif category == "restaurant":
        raw = await supabase_service.search_amenities_semantic(query, "restaurant", limit=limit * 4)
        return rank_pois_by_topsis(raw, category="restaurant", preference_mode="standard", top_n=limit)
    else:
        raw = await supabase_service.search_pois_semantic(query=query, limit=limit * 4)
        return rank_pois_by_topsis(raw, category="poi", preference_mode="standard", top_n=limit)


async def search_specific_place(query: str, category: str = "attraction") -> dict:
    """Cari tempat SPESIFIK berdasarkan nama dari database."""
    results = await supabase_service.search_specific_place(query, category)
    return {
        "found": len(results) > 0,
        "count": len(results),
        "results": results,
        "message": (
            f"Ditemukan {len(results)} tempat untuk '{query}'."
            if results
            else f"Tempat '{query}' TIDAK DITEMUKAN di database. Jangan mengarang data. Tawarkan alternatif kepada user."
        )
    }


async def search_specific_place_nearby(
    query: str, lat: float, lng: float, radius_m: float = 15000
) -> dict:
    """
    Cari tempat SPESIFIK berdasarkan nama DAN memastikan lokasinya
    berdekatan dengan koordinat (lat, lng) dalam radius radius_m meter.
    Gunakan tool ini saat MENAMBAH tempat ke itinerary yang sedang diedit,
    agar tempat baru tidak merusak rute harian yang sudah ada.
    """
    results = await supabase_service.search_specific_place_nearby(query, lat, lng, radius_m)
    return {
        "found": len(results) > 0,
        "count": len(results),
        "results": results,
        "message": (
            f"Ditemukan {len(results)} tempat '{query}' dalam radius {radius_m/1000:.1f} km."
            if results
            else (
                f"Tempat '{query}' TIDAK DITEMUKAN dalam radius {radius_m/1000:.1f} km dari koordinat tersebut. "
                "Coba perluas radius, atau gunakan search_specific_place tanpa filter lokasi."
            )
        ),
    }

async def validate_itinerary_safety(
    poi_ids: list,
    date_start: str,
    date_end: str
) -> dict:
    """
    Mengecek apakah POI-POI yang dipilih terblokir oleh upacara Odalan.
    Gunakan place_id (string) dari hasil get_smart_recommendations.
    """
    from app.engine.odalan_checker import evaluate_odalan_status
    active_odalans = await supabase_service.get_all_active_odalans(date_start, date_end)
    blocked_pois = []
    for poi_id in poi_ids:
        check = evaluate_odalan_status(str(poi_id), active_odalans)
        if check.status == "BLOCKED":
            blocked_pois.append({"poi_id": poi_id, "reason": check.message})
    return {
        "status": "CONFLICT" if blocked_pois else "SAFE",
        "blocked_pois": blocked_pois,
        "message": (
            f"⚠️ {len(blocked_pois)} tempat terblokir Odalan!" if blocked_pois
            else "✅ Semua POI aman dari konflik Odalan."
        )
    }

async def get_nearby_places(
    lat: float, lng: float, category: str, radius_km: float = 3.0
) -> list[dict]:
    """Cari attraction, hotel, atau restoran terdekat dari koordinat tertentu."""
    if category == "attraction":
        return await supabase_service.search_pois_nearby(lat, lng, radius_km * 1000)
    else:
        return await supabase_service.search_amenities_nearby(
            category, lat, lng, radius_km * 1000, limit=5
        )


async def get_inspiration_narration(query: str, limit: int = 3) -> list[dict]:
    """Ambil cerita/narasi puitis tentang suatu tempat atau kawasan di Bali."""
    return await supabase_service.search_inspiration_narrations(query, limit)


# ============================================================
# OPENAI TOOLS SCHEMA
# ============================================================

AVAILABLE_FUNCTIONS = {
    "get_bali_context": get_bali_context,
    "get_smart_recommendations": get_smart_recommendations,
    "get_general_recommendations": get_general_recommendations,
    "search_specific_place": search_specific_place,
    "search_specific_place_nearby": search_specific_place_nearby,
    "get_nearby_places": get_nearby_places,
    "validate_itinerary_safety": validate_itinerary_safety,
    "get_inspiration_narration": get_inspiration_narration,
}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_bali_context",
            "description": "Ambil info cuaca & daftar zona hindari Odalan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_start": {"type": "string"},
                    "date_end": {"type": "string"},
                    "district": {"type": "string"}
                },
                "required": ["date_start", "date_end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_smart_recommendations",
            "description": (
                "Cari tempat wisata dengan Radius & Anchor-First clustering + TOPSIS ranking. "
                "Tool ini menghasilkan pool POI dan restoran per hari yang geographically tight. "
                "Untuk category='poi': mengembalikan list per hari berisi {day, anchor, pois[], restaurants[]}. "
                "Restoran SUDAH disertakan di output tool — kamu WAJIB memasukkannya ke dalam places di waktu makan. "
                "Gunakan preference_mode='hidden_gem' untuk tempat tersembunyi, "
                "'luxury' untuk premium, 'budget' untuk hemat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": (
                            "WAJIB LAKUKAN QUERY TRANSFORMATION! Jangan masukkan kalimat percakapan user secara mentah. "
                            "Ubah permintaan user menjadi 3-6 kata kunci deskriptif (semantic vibe) yang padat makna untuk pencarian database vektor. "
                            "Contoh: Jika user minta 'tolong cariin pantai yang bagus buat sunset di kuta', ubah query menjadi 'pantai pasir putih ombak sunset kuta bali'. "
                            "Pastikan nama daerah (misal: kuta, ubud) selalu ikut dimasukkan ke dalam query jika user menyebutkannya."
                        )
                    },
                    "num_days": {"type": "integer", "description": "Jumlah hari perjalanan. WAJIB sesuai permintaan user."},
                    "category": {"type": "string", "enum": ["poi", "hotel", "restaurant"]},
                    "preference_mode": {
                        "type": "string",
                        "enum": ["standard", "hidden_gem", "luxury", "budget"],
                    },
                    "user_detected_location": {
                        "type": "string",
                        "description": "District/area Bali yang disebutkan user (misal: 'Ubud', 'Kuta'). Opsional."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_general_recommendations",
            "description": (
                "Gunakan tool ini HANYA untuk rekomendasi tempat umum tanpa membuat itinerary. "
                "WAJIB LAKUKAN QUERY TRANSFORMATION! Ubah ke frasa deskriptif (maks 15 kata). "
                "EDGE CASE HANDLING: Jika user bertanya terlalu luas (misal: 'tempat bagus'), tambahkan kata kunci default seperti 'wisata populer indah bali'. "
                "Jika hasil dari database kosong, JANGAN mengarang tempat palsu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Kata kunci semantic (vibe) dan daerah."},
                    "category": {"type": "string", "enum": ["poi", "hotel", "restaurant"]},
                    "limit": {"type": "integer", "description": "Jumlah rekomendasi (default 5)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_specific_place",
            "description": (
                "Cari tempat SPESIFIK berdasarkan nama dari database. "
                "Jika field 'found' = false, tempat TIDAK ADA — JANGAN mengarang koordinat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearby_places",
            "description": "Cari tempat atraksi, hotel atau restoran terdekat dari koordinat tertentu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                    "category": {"type": "string"},
                    "radius_km": {"type": "number"}
                },
                "required": ["lat", "lng", "category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_itinerary_safety",
            "description": (
                "Cek apakah tempat-tempat yang dipilih terblokir upacara Odalan pada tanggal perjalanan. "
                "Panggil SETELAH get_smart_recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_ids": {"type": "array", "items": {"type": "string"}},
                    "date_start": {"type": "string"},
                    "date_end": {"type": "string"}
                },
                "required": ["poi_ids", "date_start", "date_end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_specific_place_nearby",
            "description": (
                "Cari tempat SPESIFIK berdasarkan nama DAN memastikan lokasinya berdekatan "
                "dengan koordinat referensi dalam radius tertentu. "
                "WAJIB digunakan saat MENAMBAH tempat baru ke itinerary yang sedang diedit — "
                "ambil lat/lng dari salah satu tempat yang sudah ada di hari tersebut. "
                "Ini mencegah tempat baru yang jauh merusak efisiensi rute harian."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Nama atau kata kunci tempat yang dicari."},
                    "lat": {"type": "number", "description": "Latitude titik referensi (dari tempat yang sudah ada di hari itu)."},
                    "lng": {"type": "number", "description": "Longitude titik referensi (dari tempat yang sudah ada di hari itu)."},
                    "radius_m": {"type": "number", "description": "Radius pencarian dalam meter. Default: 15000 (15 km)."}
                },
                "required": ["query", "lat", "lng"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_inspiration_narration",
            "description": (
                "Ambil cerita/narasi puitis tentang suatu tempat atau kawasan di Bali. "
                "Panggil ini untuk narasi itinerary yang kaya dan emosional."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]


# ============================================================
# HEIDI SYSTEM PROMPT LOAD HELPER
# ============================================================

def load_prompt_from_md(file_name: str, replacements: dict) -> str:
    filepath = os.path.join(os.path.dirname(__file__), "prompts", file_name)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            for key, value in replacements.items():
                content = content.replace(f"[{key}]", str(value))
            return content
    except FileNotFoundError:
        return "Prompt file not found."


# ============================================================
# BACKEND MEAL INJECTION (DEPRECATED v9.0 — kept for reference)
# ============================================================
# Meal injection is now handled by Heidi AI.
# Restaurants are delivered in the per-day pool from
# generate_clustered_pool_delivery, and AI places them at
# logical dining times (Lunch 12:00-13:30, Dinner 18:30-20:00).

# async def inject_meals_to_itinerary(
#     parsed_data: "FinalAIResponse",
#     poi_budget: dict,
#     district_hint: str = "Bali",
#     preference_mode: str = "standard",
# ) -> "FinalAIResponse":
#     """DEPRECATED v9.0 — Meal injection sekarang dilakukan oleh AI."""
#     pass


# ============================================================
# HOTEL GUARANTEE (Backend Fallback)
# ============================================================

async def guarantee_base_hotel(
    parsed_data: "FinalAIResponse",
    district_hint: str = "Bali",
    preference_mode: str = "standard",
) -> "FinalAIResponse":
    """
    Memastikan base_hotel SELALU ada.
    Jika AI tidak mengisi base_hotel, backend mencarinya secara otomatis.

    Strategi:
    1. Cek apakah base_hotel sudah ada dan memiliki koordinat valid
    2. Jika tidak ada / koordinat null → ambil koordinat sentroid dari hari pertama
    3. Cari hotel terdekat dari sentroid menggunakan search_amenities_nearby
    4. Fallback: semantic search hotel jika nearby gagal
    5. Fallback final: isi dengan placeholder jika semua gagal (agar tidak crash)
    """
    hotel = parsed_data.base_hotel

    # Cek apakah hotel valid
    hotel_valid = (
        hotel is not None
        and hotel.name
        and hotel.latitude is not None
        and hotel.longitude is not None
    )

    if hotel_valid:
        # Hotel sudah ada, pastikan description tidak kosong
        if not hotel.description:
            hotel.description = "Akomodasi pilihan yang nyaman untuk perjalanan Anda di Bali."
        return parsed_data

    logger.warning("base_hotel tidak ada atau tidak valid — backend mencari hotel secara otomatis.")

    # Ambil koordinat referensi dari hari pertama (sentroid atraksi hari 1)
    ref_lat, ref_lng = -8.4095, 115.1889  # Default: Bali tengah
    if parsed_data.itinerary_days:
        day1 = parsed_data.itinerary_days[0]
        coords = [
            (p.latitude, p.longitude)
            for p in (day1.places or [])
            if p.latitude is not None and p.longitude is not None
        ]
        if coords:
            ref_lat = sum(c[0] for c in coords) / len(coords)
            ref_lng = sum(c[1] for c in coords) / len(coords)

    hotel_data = None

    # Coba nearby
    try:
        nearby_hotels = await supabase_service.search_amenities_nearby(
            amenity_type="hotel",
            lat=ref_lat,
            lng=ref_lng,
            radius_m=10000,  # 10km radius untuk hotel
            limit=5,
        )
        if nearby_hotels:
            hotel_data = nearby_hotels[0]
    except Exception as e:
        logger.warning(f"Hotel nearby search gagal: {e}")

    # Fallback semantic
    if not hotel_data:
        try:
            query_mode = {
                "luxury": "luxury resort hotel bali",
                "budget": "budget guesthouse hostel bali murah",
                "hidden_gem": "boutique hotel villa tersembunyi bali",
            }.get(preference_mode, f"hotel {district_hint} bali")

            semantic_hotels = await supabase_service.search_amenities_semantic(
                query=query_mode,
                amenity_type="hotel",
                limit=10,
            )
            if semantic_hotels:
                hotel_data = semantic_hotels[0]
        except Exception as e:
            logger.warning(f"Hotel semantic search fallback gagal: {e}")

    if hotel_data:
        metadata = hotel_data.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        parsed_data.base_hotel = BaseHotel(
            place_id=hotel_data.get("place_id"),
            name=hotel_data.get("name", "Hotel Bali"),
            latitude=hotel_data.get("latitude"),
            longitude=hotel_data.get("longitude"),
            district=hotel_data.get("district", district_hint),
            rating=hotel_data.get("rating"),
            image_url=hotel_data.get("image_url"),
            description=(
                hotel_data.get("content")
                or metadata.get("description")
                or "Akomodasi yang nyaman untuk perjalanan Anda di Bali."
            ),
        )
        logger.info(f"Hotel otomatis dipilih backend: {parsed_data.base_hotel.name}")
    else:
        # Final fallback: jika benar-benar tidak ada hotel di database
        # Isi dengan placeholder agar tidak crash
        logger.error("Tidak ada hotel ditemukan di database! Mengisi placeholder.")
        parsed_data.base_hotel = BaseHotel(
            place_id=None,
            name="Hotel (Hubungi Admin)",
            latitude=ref_lat,
            longitude=ref_lng,
            district=district_hint,
            rating=None,
            image_url=None,
            description="Sistem tidak dapat menemukan hotel di database. Silakan pilih hotel manual.",
        )

    return parsed_data


# ============================================================
# COORDINATE ENRICHMENT (Backend Safety Net)
# ============================================================

async def enrich_place_coordinates(parsed_data: "FinalAIResponse") -> "FinalAIResponse":
    """
    Safety net: jika AI lupa mengisi latitude/longitude pada PlaceItem,
    backend akan mencarinya secara otomatis menggunakan place_id.

    Strategi:
    1. Kumpulkan semua place_id dari tempat yang lat/lng-nya null
    2. Batch-query ke poi_attractions, hotel_amenities, culinary_amenities
    3. Injeksi koordinat yang ditemukan kembali ke PlaceItem
    """
    if not parsed_data.itinerary_days:
        return parsed_data

    # Kumpulkan place_id yang koordinatnya null
    missing_ids: set[str] = set()
    for day in parsed_data.itinerary_days:
        for place in (day.places or []):
            if place.place_id and (place.latitude is None or place.longitude is None):
                missing_ids.add(place.place_id)

    # Cek juga base_hotel
    hotel = parsed_data.base_hotel
    hotel_needs_coords = (
        hotel is not None
        and hotel.place_id
        and (hotel.latitude is None or hotel.longitude is None)
    )
    if hotel_needs_coords:
        missing_ids.add(hotel.place_id)

    if not missing_ids:
        return parsed_data

    logger.info(f"Coordinate enrichment: {len(missing_ids)} place_id butuh koordinat: {list(missing_ids)[:5]}")

    # Batch-lookup dari semua tabel relevan
    coord_map: dict[str, dict] = {}
    tables = [
        ("poi_attractions", "place_id, name, latitude, longitude"),
        ("hotel_amenities", "place_id, name, latitude, longitude"),
        ("culinary_amenities", "place_id, name, latitude, longitude"),
    ]

    missing_list = list(missing_ids)
    for table_name, cols in tables:
        if not missing_list:
            break
        try:
            def _make_fetch(tbl, c, ids):
                def _fetch():
                    return (
                        supabase_service.client.table(tbl)
                        .select(c)
                        .in_("place_id", ids)
                        .execute()
                    )
                return _fetch

            result = await asyncio.to_thread(_make_fetch(table_name, cols, missing_list))
            for row in (result.data or []):
                pid = row.get("place_id")
                lat = row.get("latitude")
                lng = row.get("longitude")
                if pid and lat is not None and lng is not None:
                    coord_map[pid] = {"latitude": lat, "longitude": lng}
        except Exception as e:
            logger.warning(f"Coordinate enrichment: query '{table_name}' gagal: {e}")

    if not coord_map:
        logger.warning("Coordinate enrichment: tidak menemukan koordinat dari database.")
        return parsed_data

    injected = 0
    for day in parsed_data.itinerary_days:
        for place in (day.places or []):
            if place.place_id and place.place_id in coord_map:
                if place.latitude is None:
                    place.latitude = coord_map[place.place_id]["latitude"]
                    injected += 1
                if place.longitude is None:
                    place.longitude = coord_map[place.place_id]["longitude"]

    if hotel_needs_coords and hotel.place_id in coord_map:
        hotel.latitude = coord_map[hotel.place_id]["latitude"]
        hotel.longitude = coord_map[hotel.place_id]["longitude"]
        injected += 1

    logger.info(f"Coordinate enrichment selesai: {injected} place diinjeksi koordinatnya.")
    return parsed_data



def extract_district_hint(raw_text: str, message: str) -> str:
    """
    Mencoba mendeteksi area/district Bali dari teks yang ada.
    Digunakan sebagai hint untuk pencarian hotel dan restoran.
    """
    bali_districts = [
        "ubud", "kuta", "seminyak", "canggu", "sanur", "nusa dua",
        "jimbaran", "uluwatu", "tabanan", "singaraja", "lovina",
        "amed", "padangbai", "candidasa", "denpasar", "legian",
        "nusa penida", "nusa lembongan", "buleleng", "gianyar",
        "klungkung", "karangasem", "bangli",
    ]
    combined = (raw_text + " " + message).lower()
    for d in bali_districts:
        if d in combined:
            return d.title()
    return "Bali"


# ============================================================
# SANITIZE & VALIDATE JSON OUTPUT
# ============================================================

def sanitize_ai_output(raw_dict: dict) -> dict:
    """
    Membersihkan dan memperbaiki output JSON AI sebelum validasi Pydantic.
    Menangani berbagai edge case dari output AI yang tidak sempurna.
    """
    # 1. Deteksi dan isi response_type jika hilang
    if "response_type" not in raw_dict or not raw_dict["response_type"]:
        if "itinerary_days" in raw_dict and raw_dict["itinerary_days"]:
            raw_dict["response_type"] = "itinerary"
        elif "recommendations" in raw_dict and raw_dict["recommendations"]:
            raw_dict["response_type"] = "recommendation"
        elif "clarifying_questions" in raw_dict and raw_dict["clarifying_questions"]:
            raw_dict["response_type"] = "clarifying"
        else:
            raw_dict["response_type"] = "chat"

    # 2. Pastikan suggested_replies ada dan minimal 3 item
    if "suggested_replies" not in raw_dict or not isinstance(raw_dict["suggested_replies"], list) or not raw_dict["suggested_replies"]:
        raw_dict["suggested_replies"] = [
            "Bagaimana cuaca di Bali sekarang?",
            "Rekomendasikan pantai terdekat",
            "Buatkan itinerary untuk area Ubud"
        ]
    elif len(raw_dict["suggested_replies"]) < 3:
        while len(raw_dict["suggested_replies"]) < 3:
            raw_dict["suggested_replies"].append("Tunjukkan opsi hotel lainnya")

    # 3. Buat/perbaiki message_to_user jika hilang atau terlalu pendek
    if "message_to_user" not in raw_dict or not raw_dict["message_to_user"] or len(str(raw_dict["message_to_user"]).strip()) < 5:
        resp_type = raw_dict["response_type"]
        if resp_type == "itinerary":
            title = raw_dict.get("trip_title") or "Rencana Perjalanan Bali"
            hotel_name = ""
            if "base_hotel" in raw_dict and isinstance(raw_dict["base_hotel"], dict):
                hotel_name = raw_dict["base_hotel"].get("name", "")
            
            msg = f"# ✈️ {title}\n\n"
            msg += "Halo! Rencana perjalanan Anda di Bali telah siap disusun. Berikut adalah ringkasan itinerary harian Anda:\n\n"
            
            if hotel_name:
                msg += f"🏨 **Akomodasi:** Menginap di **{hotel_name}** sebagai home base.\n\n"
            
            msg += "---\n\n"
            
            days = raw_dict.get("itinerary_days") or []
            for d in days:
                d_num = d.get("day", 1)
                theme = d.get("theme") or "Eksplorasi"
                date_str = d.get("date") or ""
                date_part = f" · {date_str}" if date_str else ""
                msg += f"## 🌅 Hari {d_num}{date_part} — {theme}\n"
                
                places = d.get("places") or []
                for p in places:
                    p_name = p.get("name", "Tempat")
                    p_time = p.get("visit_time") or "08:00"
                    p_cat = p.get("category") or "attraction"
                    p_desc = p.get("description") or ""
                    
                    cat_emoji = "🍽️" if p_cat == "restaurant" else "🏨" if p_cat == "hotel" else "📍"
                    msg += f"- **{p_time}** {cat_emoji} **{p_name}**: {p_desc}\n"
                msg += "\n"
                
            msg += "---\n\n"
            
            budget = raw_dict.get("budget_breakdown") or {}
            total = raw_dict.get("total_budget_idr") or 0
            if budget or total:
                msg += "## 💰 Estimasi Budget\n"
                if budget.get("accommodation_idr"):
                    msg += f"- 🏨 Akomodasi: Rp {budget['accommodation_idr']:,}\n".replace(",", ".")
                if budget.get("food_idr"):
                    msg += f"- 🍽️ Makan & Minum: Rp {budget['food_idr']:,}\n".replace(",", ".")
                if budget.get("transport_idr"):
                    msg += f"- 🚗 Transportasi: Rp {budget['transport_idr']:,}\n".replace(",", ".")
                if budget.get("entrance_fee_idr"):
                    msg += f"- 🎟️ Tiket Masuk: Rp {budget['entrance_fee_idr']:,}\n".replace(",", ".")
                if budget.get("miscellaneous_idr"):
                    msg += f"- 🛍️ Lain-lain: Rp {budget['miscellaneous_idr']:,}\n".replace(",", ".")
                if total:
                    msg += f"- **Total Estimasi: Rp {total:,}**\n\n".replace(",", ".")
            
            msg += "Apakah Anda ingin menyesuaikan tempat wisata atau mengganti hotel? Silakan beritahu saya!"
            raw_dict["message_to_user"] = msg
            
        elif resp_type == "recommendation":
            msg = "Berikut adalah beberapa rekomendasi tempat menarik di Bali untuk Anda:\n\n"
            recs = raw_dict.get("recommendations") or []
            for r in recs:
                name = r.get("name", "Tempat")
                desc = r.get("description") or ""
                district = r.get("district") or "Bali"
                rating = r.get("rating")
                rating_str = f" (⭐ {rating})" if rating else ""
                msg += f"### 📍 {name}{rating_str}\n"
                msg += f"Area: *{district}*\n"
                msg += f"{desc}\n\n"
            raw_dict["message_to_user"] = msg
            
        elif resp_type == "clarifying":
            msg = "Untuk membantu menyusun rencana perjalanan terbaik, mohon berikan informasi berikut:\n\n"
            questions = raw_dict.get("clarifying_questions") or []
            for q in questions:
                msg += f"- {q}\n"
            raw_dict["message_to_user"] = msg
            
        else: # chat
            raw_dict["message_to_user"] = "Halo! Saya Heidi, asisten perjalanan AI Anda untuk Bali. Bagaimana saya bisa membantu perjalanan Anda hari ini?"

    # Fix base_hotel description kosong
    if "base_hotel" in raw_dict and isinstance(raw_dict["base_hotel"], dict):
        if not raw_dict["base_hotel"].get("description"):
            raw_dict["base_hotel"]["description"] = "Akomodasi pilihan yang nyaman untuk perjalanan Anda."

    # Fix category yang tidak valid
    category_map = {
        "cafe": "restaurant", "warung": "restaurant", "food": "restaurant",
        "dining": "restaurant", "bar": "restaurant", "rumah makan": "restaurant",
        "lodging": "hotel", "resort": "hotel", "villa": "hotel",
        "guesthouse": "hotel", "hostel": "hotel", "penginapan": "hotel",
        "temple": "attraction", "beach": "attraction", "park": "attraction",
        "museum": "attraction", "zoo": "attraction", "waterfall": "attraction",
    }

    if "itinerary_days" in raw_dict and isinstance(raw_dict["itinerary_days"], list):
        for day in raw_dict["itinerary_days"]:
            if "places" in day and isinstance(day["places"], list):
                for place in day["places"]:
                    cat = str(place.get("category", "")).lower()
                    if cat not in ["attraction", "hotel", "restaurant"]:
                        place["category"] = category_map.get(cat, "attraction")

                    # Pastikan estimated_cost_idr tidak None
                    if not place.get("estimated_cost_idr"):
                        if place.get("category") == "restaurant":
                            place["estimated_cost_idr"] = 75000
                        else:
                            place["estimated_cost_idr"] = 25000

                    # Pastikan visit_duration_mins tidak None
                    if not place.get("visit_duration_mins"):
                        if place.get("category") == "restaurant":
                            place["visit_duration_mins"] = 60
                        else:
                            place["visit_duration_mins"] = 75

                    # Pastikan tags tidak kosong
                    if not place.get("tags"):
                        name = place.get("name", "").lower()
                        if any(k in name for k in ["pura", "temple", "tanah lot", "uluwatu"]):
                            place["tags"] = ["pura", "budaya", "spiritual"]
                        elif any(k in name for k in ["pantai", "beach", "nusa"]):
                            place["tags"] = ["pantai", "sunset", "foto"]
                        elif place.get("category") == "restaurant":
                            place["tags"] = ["kuliner", "makan", "lokal"]
                        else:
                            place["tags"] = ["wisata", "bali", "atraksi"]

    return raw_dict


async def repair_empty_places(
    raw_dict: dict,
    messages_history: list,
) -> dict:
    """
    Targeted repair: AI mengembalikan itinerary dengan places:[],
    tapi narasi di message_to_user sudah benar.

    Strategy:
    1. Ambil narasi dari message_to_user + tool results dari history
    2. Kirim ke LLM dengan instruksi ketat untuk mengisi places saja
    3. Merge hasilnya ke raw_dict
    """
    logger.warning("REPAIR: Semua places kosong! Menjalankan repair_empty_places...")

    # Kumpulkan tool results dari history (data POI yang sudah di-fetch sebelumnya)
    tool_data_parts = []
    for msg in messages_history:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            content = msg.get("content", "")
            tool_name = msg.get("name", "")
            if tool_name in ("get_smart_recommendations", "search_specific_place", "get_nearby_places"):
                tool_data_parts.append(f"[{tool_name} result]:\n{content[:3000]}")

    tool_data_str = "\n\n".join(tool_data_parts) if tool_data_parts else "(Tidak ada data tool tersedia)"

    narrative = raw_dict.get("message_to_user", "")
    days_count = len(raw_dict.get("itinerary_days", []))
    day_themes = [
        f"Hari {d.get('day')}: {d.get('theme', 'Eksplorasi')}"
        for d in raw_dict.get("itinerary_days", [])
    ]
    themes_str = "\n".join(day_themes)

    repair_prompt = f"""
Kamu adalah asisten JSON repair. Tugasmu adalah mengisi array `places` pada itinerary.

Itinerary ini memiliki {days_count} hari:
{themes_str}

Narasi yang sudah dibuat (GUNAKAN INI sebagai referensi tempat yang harus dimasukkan):
{narrative[:4000]}

Data yang tersedia dari tool:
{tool_data_str[:4000]}

Tugasmu: Kembalikan HANYA objek JSON dengan key `itinerary_days`.
Setiap hari WAJIB memiliki array `places` yang berisi minimal 3 attraction + 1 restaurant.

Contoh format PlaceItem (WAJIB LENGKAP, gunakan data dari tool di atas untuk lat/lng):
{{
  "place_id": "ChIJ...",
  "name": "Nama Tempat",
  "category": "attraction",
  "latitude": -8.5,
  "longitude": 115.1,
  "district": "Kuta",
  "description": "Deskripsi singkat.",
  "rating": 4.5,
  "image_url": null,
  "tags": ["wisata", "bali"],
  "visit_time": "09:00",
  "visit_duration_mins": 90,
  "estimated_cost_idr": 50000,
  "tips": "Tip berguna.",
  "route_to_next": null
}}

Aturan:
- Ekstrak SEMUA tempat yang disebut dalam narasi
- Gunakan data dari tool untuk latitude/longitude (WAJIB diisi)
- JANGAN mengarang tempat yang tidak ada di narasi atau data tool
- Semua field route (route_to_next, dll) harus null
- Kembalikan HANYA JSON: {{"itinerary_days": [{{"day": 1, "places": [...]}}, ...]}}
"""

    try:
        client = _get_client()
        repair_response = await client.chat.completions.create(
            model=settings.openai_model_id,
            messages=[
                {"role": "system", "content": "Kamu adalah JSON repair assistant. Kembalikan HANYA JSON murni."},
                {"role": "user", "content": repair_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        repair_json = json.loads(repair_response.choices[0].message.content)
        repaired_days = repair_json.get("itinerary_days", [])

        if not repaired_days:
            logger.warning("REPAIR: Repair call mengembalikan itinerary_days kosong juga.")
            return raw_dict

        # Merge repaired places ke raw_dict
        repaired_map = {d.get("day"): d.get("places", []) for d in repaired_days}
        total_repaired = 0
        for day in raw_dict.get("itinerary_days", []):
            day_num = day.get("day")
            repaired_places = repaired_map.get(day_num, [])
            if repaired_places and not day.get("places"):
                day["places"] = repaired_places
                total_repaired += len(repaired_places)

        logger.info(f"REPAIR: Berhasil mengisi {total_repaired} places ke itinerary.")
        return raw_dict

    except Exception as e:
        logger.error(f"REPAIR: repair_empty_places gagal: {e}")
        return raw_dict


@app.post("/api/chat", response_model=FinalAIResponse, tags=["AI Chat"])
@limiter.limit(RATE_CHAT)
async def chat_with_heidi(
    request: Request,
    req: ChatRequest,
    current_user=Depends(get_optional_user),
):
    """
    Chat dengan Heidi — AI Travel Planner Bali.
    - Guest: Gunakan history dari frontend.
    - Authenticated: Load history dari DB, simpan otomatis.
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="Server belum dikonfigurasi (OPENAI_API_KEY kosong). Hubungi administrator."
        )

    try:
        is_authenticated = current_user is not None
        user_id = current_user.id if is_authenticated else None
        is_editing = req.current_itinerary is not None

        # Expose user_id to the rate limiter key function via request state
        if user_id:
            request.state.user_id = str(user_id)

        # --- SESSION PERSISTENCE: load itinerary jika frontend tidak menyertakannya ---
        # Jika session_id diberikan tapi current_itinerary kosong, coba load dari DB (Hanya untuk logged-in user).
        _session_itinerary: Optional[Dict] = None
        if is_authenticated and req.session_id and not is_editing:
            try:
                _session_itinerary = await supabase_service.get_latest_itinerary_by_session(req.session_id, str(user_id))
                if _session_itinerary:
                    is_editing = True
                    logger.info(
                        f"Session {req.session_id}: itinerary dimuat dari user_itineraries "
                        f"(response_type={_session_itinerary.get('response_type', '?')})."
                    )
            except Exception as _e:
                logger.warning(f"Gagal load itinerary session ({req.session_id}): {_e}")

        # Gunakan itinerary dari session jika frontend tidak mengirimnya
        _effective_itinerary = req.current_itinerary or _session_itinerary
        is_editing = _effective_itinerary is not None

        # --- KALKULASI POI BUDGET (PRE-AI) ---
        db_districts = await supabase_service.get_db_districts()
        trip_params = await extract_trip_parameters_from_message(req.message, db_districts)

        # --- PRE-COMPILE HISTORY FOR DEEP RESEARCH CHECK ---
        active_session_id = None
        db_history = []
        history_for_checking = []
        if is_authenticated:
            active_session_id = await supabase_service.get_or_create_chat_session(user_id, req.session_id)
            db_history = await supabase_service.get_chat_history(active_session_id)
            for msg in db_history:
                oai_role = "assistant" if msg["role"] in ["model", "assistant"] else "user"
                history_for_checking.append({"role": oai_role, "content": msg["content"]})
        else:
            for msg in (req.history or []):
                role = "assistant" if msg["role"] in ["model", "assistant"] else "user"
                content = msg.get("parts", msg.get("content", ""))
                history_for_checking.append({"role": role, "content": content})

        num_days_hint = 2

        # --- DEEP RESEARCH CHECK ---
        dr_vars = None
        if req.mode == "deep_research" and not is_editing:
            dr_result = await check_deep_research_variables(req.message, history_for_checking)
            logger.info(f"Deep Research variable check result: {dr_result}")
            if not dr_result.get("all_filled", False):
                # Format missing variables nicely for the prompt
                missing_vars = dr_result.get("missing_variables", [])
                indonesian_var_names = {
                    "lokasi": "Lokasi (Contoh: Ubud, Kuta)",
                    "tanggal": "Tanggal (Contoh: Besok, 12 Agustus)",
                    "durasi": "Durasi (Contoh: 3 hari)",
                    "budget": "Budget (Contoh: Hemat, Menengah)",
                    "teman_perjalanan": "Teman Perjalanan (Contoh: Sendiri, Keluarga)",
                    "pace": "Pace (Contoh: Santai, Padat)"
                }
                missing_vars_formatted = [indonesian_var_names.get(v, v) for v in missing_vars]
                missing_vars_str = "\n".join([f"- {v}" for v in missing_vars_formatted])
                
                system_prompt = (
                    "Kamu adalah **Heidi**, asisten perjalanan AI spesialis Bali dari SobatNavi.\n"
                    "Kepribadianmu: hangat, informatif, sangat ramah, dan membantu.\n\n"
                    "Saat ini kita sedang dalam mode Deep Research untuk merencanakan liburan terbaik di Bali.\n"
                    "Namun, ada beberapa variabel penting yang masih kosong/belum diisi oleh user.\n\n"
                    "Variabel yang MASIH KOSONG:\n"
                    f"{missing_vars_str}\n\n"
                    "Tugasmu:\n"
                    "Tanyakan kepada user dengan cara yang ramah, hangat, dan interaktif (dalam Bahasa Indonesia) untuk melengkapi variabel yang masih kosong di atas.\n"
                    "JANGAN membuat itinerary sekarang! Fokuslah hanya untuk menanyakan variabel yang belum terisi.\n"
                    "Gunakan format markdown yang indah (seperti bold, list, emoji) di dalam pesanmu.\n\n"
                    "Kamu WAJIB mengembalikan respon dalam format JSON murni dengan skema berikut:\n"
                    "{\n"
                    "  \"response_type\": \"clarifying\",\n"
                    "  \"message_to_user\": \"Pesan ramah dalam markdown yang menanyakan variabel yang kurang.\",\n"
                    "  \"suggested_replies\": [\n"
                    "     \"3 contoh jawaban yang relevan dan singkat bagi user untuk melengkapi variabel tersebut.\"\n"
                    "  ]\n"
                    "}\n"
                    "Kembalikan HANYA JSON objek tersebut."
                )
                
                # Run the LLM to get the response
                llm_messages = [{"role": "system", "content": system_prompt}]
                for msg in history_for_checking:
                    llm_messages.append(msg)
                llm_messages.append({"role": "user", "content": req.message})
                
                client = _get_client()
                response = await client.chat.completions.create(
                    model=settings.openai_model_id,
                    messages=llm_messages,
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                raw_text = response.choices[0].message.content
                raw_dict = json.loads(raw_text)
                raw_dict = sanitize_ai_output(raw_dict)
                parsed_data = FinalAIResponse.model_validate(raw_dict)
                
                # Save messages if authenticated
                if is_authenticated:
                    await supabase_service.save_chat_message(
                        session_id=active_session_id,
                        role="user",
                        content=req.message,
                    )
                    await supabase_service.save_chat_message(
                        session_id=active_session_id,
                        role="assistant",
                        content=raw_text,
                    )
                    if hasattr(parsed_data, "session_id"):
                        parsed_data.session_id = active_session_id
                else:
                    if hasattr(parsed_data, "session_id"):
                        parsed_data.session_id = req.session_id
                
                return parsed_data
            else:
                dr_vars = dr_result.get("variables", {})
                
                # Override parameters using extracted deep research variables
                extracted_days = extract_number_of_days(dr_vars.get("durasi"))
                if extracted_days:
                    num_days_hint = extracted_days
                
                pace_val = dr_vars.get("pace")
                if pace_val:
                    pace_str = str(pace_val).lower()
                    if "santai" in pace_str:
                        trip_params["pace"] = "santai"
                    elif "padat" in pace_str or "cepat" in pace_str:
                        trip_params["pace"] = "padat"
                    else:
                        trip_params["pace"] = "normal"
                
                lokasi_val = dr_vars.get("lokasi")
                if lokasi_val:
                    trip_params["detected_location"] = lokasi_val
                    det_low = lokasi_val.lower()
                    matches = [db_d for db_d in db_districts if det_low in db_d.lower() or db_d.lower() in det_low]
                    if matches:
                        trip_params["normalized_districts"] = matches
                        
                budget_val = dr_vars.get("budget")
                if budget_val:
                    budget_str = str(budget_val).lower()
                    if any(x in budget_str for x in ["hemat", "murah", "backpacker", "budget"]):
                        req.budget_preference = "budget"
                    elif any(x in budget_str for x in ["mewah", "luxury", "sultan", "mahal"]):
                        req.budget_preference = "luxury"
                    else:
                        req.budget_preference = "moderate"
                
                # Force intent to create so it generates the itinerary
                trip_params["intent"] = "create"

        if trip_params.get("intent") in ["add_place", "delete_place", "swap_place"] and not is_editing:
            return FinalAIResponse(
                response_type="chat",
                message_to_user="Maaf, saat ini belum ada itinerary yang sedang kita susun. Kamu harus membuat itinerary liburan terlebih dahulu sebelum bisa menambah atau menghapus tempat. Mau aku buatkan untuk berapa hari?",
                suggested_replies=["Buatkan itinerary 3 hari", "Rekomendasi tempat di Ubud", "Cari pantai bagus"],
                itinerary_days=None,
                base_hotel=None,
                trip_title=None
            )

        if trip_params.get("intent") == "swap_place":
            return FinalAIResponse(
                response_type="chat",
                message_to_user="Maaf, untuk menukar tempat (swapping), kamu harus melakukannya secara manual melalui editor itinerary. Aku hanya bisa membantu menambah atau menghapus tempat lewat chat. 😊",
                suggested_replies=["Hapus tempat di Hari 1", "Tambahkan tempat di Hari 2"],
                itinerary_days=None,
                base_hotel=None,
                trip_title=None
            )

        skip_llm_loop = False
        parsed_data = None
        if trip_params.get("intent") in ["add_place", "delete_place"] and is_editing:
            edit_result = await execute_itinerary_edit(req.message, _effective_itinerary, trip_params["intent"])
            if edit_result:
                if edit_result.response_type == "chat":
                    if is_authenticated:
                        active_session_id = await supabase_service.get_or_create_chat_session(user_id, req.session_id)
                        await supabase_service.save_chat_message(
                            session_id=active_session_id,
                            role="assistant",
                            content=edit_result.message_to_user
                        )
                    return edit_result
                parsed_data = edit_result
                skip_llm_loop = True
        
        # JIKA GAGAL DETEKSI LOKASI & LOKASI PENTING UNTUK ITINERARY
        if not skip_llm_loop and not trip_params["detected_location"] and req.mode == "general" and trip_params.get("intent") == "create":
            # Berhenti sejenak dan minta klarifikasi
            return {
                "response_type": "clarifying",
                "message_to_user": "Untuk menyusun itinerary yang akurat, boleh bantu saya tahu daerah mana di Bali yang ingin Anda jelajahi?",
                "suggested_replies": ["Ubud", "Kuta", "Seminyak", "Nusa Dua"],
                "itinerary_days": None
            }
        
        # Cache the extracted locations so that get_smart_recommendations tool hits the cache
        if trip_params.get("detected_location") and trip_params.get("normalized_districts"):
            raw_loc = trip_params["detected_location"].lower()
            supabase_service._normalization_cache[raw_loc] = trip_params["normalized_districts"]
            for nd in trip_params["normalized_districts"]:
                supabase_service._normalization_cache[nd.lower()] = trip_params["normalized_districts"]
        # Deteksi jumlah hari dari history / pesan untuk kalkulasi budget
        # Default ke 2 hari; AI yang nanti menentukan secara akurat
        if not (req.mode == "deep_research" and dr_vars):
            day_match = re.search(r"(\d+)\s*(hari|days?|malam|night)", req.message.lower())
            if day_match:
                num_days_hint = max(1, min(int(day_match.group(1)), 14))

        poi_budget = calculate_poi_budget(
            num_days=num_days_hint,
            pace=trip_params["pace"],
            user_requested_pois=trip_params["user_requested_pois"],
            is_half_day=trip_params["is_half_day"],
        )
        logger.info(f"POI Budget kalkulasi: {poi_budget}")

        # --- MAP budget_preference → preference_mode ---
        _budget_to_mode = {"budget": "budget", "moderate": "standard", "luxury": "luxury"}
        preference_mode = _budget_to_mode.get(req.budget_preference, "standard")

        # --- DYNAMIC AGENT ROUTING & PROMPT LOADING ---
        intent = trip_params.get("intent", "chat") or "chat"
        today_str = datetime.now().strftime("%Y-%m-%d")
        tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        schema_string = json.dumps(FinalAIResponse.model_json_schema(), indent=2)

        replacements = {
            "TODAY": today_str,
            "TOMORROW": tomorrow_str,
            "SCHEMA_STRING": schema_string,
        }

        active_tools = []
        if intent == "create":
            alert_instruction = ""
            if poi_budget.get("was_capped", False):
                alert_instruction = (
                    "⚠️ PERINGATAN EDGE CASE (PENTING): Permintaan jumlah tempat wisata (atraksi) per hari dari user terlalu besar "
                    "dan tidak realistis untuk diselesaikan dalam satu hari, sehingga sistem telah membatasinya menjadi "
                    "maksimal 7 atraksi per hari. KAMU WAJIB secara eksplisit dan sopan memberi tahu user di dalam "
                    "narasi `message_to_user` bahwa jumlah atraksi telah dikurangi/dibatasi demi kenyamanan perjalanan "
                    "mereka (agar tidak terlalu lelah dan memiliki waktu kunjungan yang cukup)."
                )

            replacements.update({
                "PACE": poi_budget.get('pace_label', 'normal'),
                "ATTRACTIONS_COUNT": poi_budget.get('attractions_per_day', 4),
                "PREFERENCE_MODE": preference_mode,
                "DAILY_POI_TARGETS": poi_budget.get('daily_targets', 'Default'),
                "NUM_DAYS_HINT": num_days_hint,
                "CAPPED_POI_ALERT_INSTRUCTION": alert_instruction
            })
            system_prompt = load_prompt_from_md("creator_agent.md", replacements)
            
            # Inject deep research context if applicable
            if req.mode == "deep_research" and dr_vars:
                deep_research_context = (
                    "\n\n## DEEP RESEARCH CONTEXT (PENTING)\n"
                    "Kamu sedang membuat itinerary dalam mode **Deep Research**. Gunakan preferensi perjalanan berikut "
                    "yang sudah dikonfirmasi oleh user untuk menyusun itinerary secara lebih detail, relevan, dan terpersonalisasi:\n"
                    f"- **Lokasi Utama**: {dr_vars.get('lokasi')}\n"
                    f"- **Tanggal Mulai**: {dr_vars.get('tanggal')}\n"
                    f"- **Durasi**: {dr_vars.get('durasi')}\n"
                    f"- **Budget Preference**: {dr_vars.get('budget')}\n"
                    f"- **Teman Perjalanan**: {dr_vars.get('teman_perjalanan')}\n"
                    f"- **Pace**: {dr_vars.get('pace')}\n\n"
                    "Pastikan pilihan tempat wisata, restoran, dan hotel benar-benar sesuai dengan teman perjalanan "
                    f"({dr_vars.get('teman_perjalanan')}), budget ({dr_vars.get('budget')}), dan pace ({dr_vars.get('pace')}). "
                    "Jelaskan di narasi `message_to_user` mengapa pilihan ini cocok untuk kelompok perjalanan tersebut."
                )
                system_prompt += deep_research_context

            allowed_names = ["get_bali_context", "get_smart_recommendations", "validate_itinerary_safety", "get_nearby_places"]
            active_tools = [tool for tool in OPENAI_TOOLS if tool["function"]["name"] in allowed_names]
        else:
            if is_editing:
                edit_context = json.dumps(_effective_itinerary, ensure_ascii=False, indent=2)
                mode_instruction = f"""
## MODE: EDIT ITINERARY (via Chat)
Kamu sedang memodifikasi itinerary berikut:
{edit_context}

## ATURAN EDIT:
1. HAPUS TEMPAT BERDASARKAN NAMA: Cari item di array `places` yang namanya cocok, keluarkan dari array.
2. HAPUS TEMPAT BERDASARKAN POSISI: Gunakan indeks array (0-based).
3. TAMBAH TEMPAT SPESIFIK: WAJIB panggil tool `search_specific_place_nearby`. Untuk parameter `lat` dan `lng`, kamu WAJIB mengambil latitude dan longitude dari salah satu tempat wisata (POI) yang SUDAH ADA di dalam array hari tersebut. Ini untuk memastikan tempat yang baru ditambahkan jaraknya berdekatan dan tidak merusak rute.
4. TAMBAH TEMPAT TANPA NAMA (tema/vibe): Panggil `get_nearby_places` atau `get_smart_recommendations`.
5. JANGAN menghitung atau mengisi field rute apapun.
6. Kembalikan SELURUH struktur itinerary yang sudah dimodifikasi.
7. Hotel (base_hotel) TIDAK BOLEH berubah kecuali user eksplisit minta ganti hotel.
8. Field `route_to_next`, `day_full_polyline`, `day_total_distance_km`, `day_total_travel_time_mins` HARUS null.
"""
                rule_12_dynamic = (
                    "12. **ABAIKAN KUOTA MINIMUM & HANDLING HARI KOSONG**: Karena ini mode EDIT, kamu WAJIB "
                    "MENGABAIKAN aturan batas minimum atraksi harian. Biarkan array `places` pada hari tersebut "
                    "berkurang atau kosong jika user memang memintanya.\n"
                    "⚠️ **[CRITICAL EDGE CASE - HARI KOSONG]**: Jika penghapusan tempat oleh user mengakibatkan "
                    "array `places` pada suatu hari menjadi KOSONG TOTAL, kamu WAJIB mendeteksinya secara sadar "
                    "(self-aware). Jangan diam saja! Di dalam `message_to_user`, beritahu user secara ramah bahwa "
                    "jadwal untuk hari tersebut sekarang kosong, lalu berikan saran ide/opsi aktivitas menarik "
                    "untuk mengisinya kembali."
                )
                workflow_instruction = """
## ALUR KERJA EDIT ITINERARY (PARTIAL UPDATE ONLY)
STEP 1 → Analisis pesan user: Apakah meminta HAPUS (Delete) atau TAMBAH (Add)?
STEP 2 → JIKA USER MINTA HAPUS: Cukup hilangkan objek tempat tersebut dari array `places` pada hari yang dimaksud. JANGAN panggil tool pencarian global atau nearby apa pun! Cukup eliminasi item dari array JSON yang sudah ada.
STEP 3 → JIKA USER MINTA TAMBAH TEMPAT SPESIFIK: Kamu WAJIB memanggil tool `search_specific_place_nearby`. Untuk parameter `lat` dan `lng`, kamu WAJIB mengambil koordinat dari salah satu tempat wisata (POI) yang SUDAH ADA di dalam array hari tersebut. Ini untuk memastikan tempat baru dikunci dalam radius berdekatan (~15km) dan tidak membuat rute hari itu melompat jauh.
STEP 4 → JIKA USER MINTA TAMBAH TEMPAT SECARA UMUM/GENERIK: Kamu WAJIB memanggil `get_smart_recommendations` atau `get_nearby_places`. DILARANG KERAS bertanya kembali kepada user atau menawarkan pilihan! Pilih 1 tempat terbaik yang paling relevan dengan tema yang diminta (misal: "air terjun di Bali"), masukkan tempat tersebut ke dalam itinerary, dan langsung konfirmasi perubahannya di `message_to_user`. Tugasmu adalah eksekusi, bukan berdiskusi.
STEP 5 → IMMUTABLE CLONING (MANDATORY): Untuk hari-hari (Day) atau data lain yang TIDAK diminta untuk diubah oleh user, kamu WAJIB menyalin (clone) seluruh strukturnya secara persis 100%, termasuk `latitude`, `longitude`, `place_id`, `name`, dan semua field lainnya. JANGAN mengubah, menghapus, atau mempersingkat data apapun yang tidak diminta user.
STEP 6 → KOORDINAT WAJIB DIJAGA: Setiap PlaceItem di semua hari HARUS memiliki `latitude` and `longitude` yang tidak null. Saat cloning, pastikan kamu menyalin nilai numerik koordinat persis seperti yang ada di `current_itinerary` yang diberikan.
STEP 7 → LARANGAN ROUTING: Biarkan semua field rute harian (`route_to_next`, `day_full_polyline`, `day_total_distance_km`, `day_total_travel_time_mins`) selalu bernilai `null` karena backend TomTom yang akan menghitung ulang jalurnya secara otomatis setelah JSON divalidasi.
STEP 8 → Tulis narasi konfirmasi hangat di `message_to_user` dan kembalikan struktur JSON penuh. Pastikan `response_type` tetap `"itinerary"`.
ATURAN KRITIKAL: DILARANG KERAS menggunakan frasa seperti "Tunggu sebentar", "Sedang diproses", atau "Aku akan segera kirimkan". Langsung berikan konfirmasi tegas bahwa jadwal SUDAH berhasil diperbarui!
"""
            elif req.mode == "deep_research":
                mode_instruction = """
## MODE: DEEP RESEARCH (Riset Mendalam)
Sebelum membuat itinerary, pastikan 5 VARIABEL berikut sudah terpenuhi dari chat history:
  ① LOKASI: Kabupaten/area di Bali yang ingin dikunjungi
  ② TANGGAL: Tanggal mulai (format YYYY-MM-DD)
  ③ DURASI: Berapa hari perjalanan
  ④ BUDGET: Kisaran anggaran (low/medium/high atau nominal IDR)
  ⑤ COMPANION: Pergi sendiri, berdua, keluarga, atau rombongan
  ⑥ PACE: Santai (sedikit tempat) atau padat (banyak tempat)

- Jika SEMUA variabel sudah ada → lanjut buat itinerary.
- Jika ADA YANG KURANG → response_type="clarifying". Isi `clarifying_questions`.
- Jika user hanya menyapa → response_type="chat".
"""
                rule_12_dynamic = (
                    f"12. **ATRAKSI MINIMUM**: Setiap hari WAJIB memiliki minimal "
                    f"{poi_budget.get('min_attractions', 2)} atraksi wisata (di luar restoran). "
                    "Jika bahan dari tool kurang, panggil tool kembali dengan query berbeda."
                )
                workflow_instruction = f"""
## ALUR KERJA PEMBUATAN ITINERARY BARU (FULL GENERATION)
STEP 1 → Panggil `get_bali_context(date_start, date_end, district)` untuk mengambil info cuaca & avoid_zones.
STEP 2 → Panggil `get_smart_recommendations(query, num_days=N, category="poi", preference_mode="{preference_mode}")` untuk menarik data POI + Restoran terkluster.
STEP 3 → Panggil `get_nearby_places(lat, lng, category="hotel")` menggunakan anchor koordinat Hari 1 untuk menentukan `base_hotel`.
STEP 4 → Panggil `validate_itinerary_safety(poi_ids, date_start, date_end)` untuk memastikan keamanan ritual adat.
STEP 5 → Susun `places` harian: gabungkan atraksi dan restoran, lalu URUTKAN KRONOLOGIS berdasarkan `visit_time`.
STEP 6 → Tulis `message_to_user` berupa narasi storytelling yang hangat minimal 250 kata.
STEP 7 → Lengkapi `trip_title` dan `suggested_replies`.
"""
            else:
                mode_instruction = f"""
## MODE: GENERAL (Langsung Proses)
- Jika user MENYAPA atau NGOBROL BIASA: response_type="chat". JANGAN panggil tool.
- Jika user HANYA MINTA REKOMENDASI tanpa jadwal: panggil get_general_recommendations, response_type="recommendation".
- Jika user MINTA ITINERARY LENGKAP → lanjut ke alur pembuatan.
  Jika tidak ada tanggal → asumsikan besok ({tomorrow_str}).
  Jika tidak ada durasi → asumsikan 1-2 hari.
  Jika tidak ada budget → asumsikan menengah.
  JANGAN BERTANYA jika data kurang! Langsung buatkan dengan asumsi!
"""
                rule_12_dynamic = (
                    f"12. **ATRAKSI MINIMUM**: Setiap hari WAJIB memiliki minimal "
                    f"{poi_budget.get('min_attractions', 2)} atraksi wisata (di luar restoran). "
                    "Jika bahan dari tool kurang, panggil tool kembali dengan query berbeda."
                )
                workflow_instruction = f"""
## ALUR KERJA PEMBUATAN ITINERARY BARU (FULL GENERATION)
STEP 1 → Panggil `get_bali_context(date_start, date_end, district)` untuk mengambil info cuaca & avoid_zones.
STEP 2 → Panggil `get_smart_recommendations(query, num_days=N, category="poi", preference_mode="{preference_mode}")` untuk menarik data POI + Restoran terkluster.
STEP 3 → Panggil `get_nearby_places(lat, lng, category="hotel")` menggunakan anchor koordinat Hari 1 untuk menentukan `base_hotel`.
STEP 4 → Panggil `validate_itinerary_safety(poi_ids, date_start, date_end)` untuk memastikan keamanan ritual adat.
STEP 5 → Susun `places` harian: gabungkan atraksi dan restoran, lalu URUTKAN KRONOLOGIS berdasarkan `visit_time`.
STEP 6 → Tulis `message_to_user` berupa narasi storytelling yang hangat minimal 250 kata.
STEP 7 → Lengkapi `trip_title` dan `suggested_replies`.
"""

            replacements.update({
                "MODE_INSTRUCTION": mode_instruction,
                "RULE_12_DYNAMIC": rule_12_dynamic,
                "WORKFLOW_INSTRUCTION": workflow_instruction,
            })
            system_prompt = load_prompt_from_md("chatter_agent.md", replacements)
            allowed_names = ["search_specific_place", "get_inspiration_narration", "get_general_recommendations"]
            active_tools = [tool for tool in OPENAI_TOOLS if tool["function"]["name"] in allowed_names]

        generated_schedule = None
        captured_date_start = None
        messages = [{"role": "system", "content": system_prompt}]
        active_session_id = None

        # --- MANAJEMEN HISTORY ---
        if is_authenticated:
            active_session_id = await supabase_service.get_or_create_chat_session(user_id, req.session_id)
            for msg in db_history:
                oai_role = "assistant" if msg["role"] in ["model", "assistant"] else "user"
                messages.append({"role": oai_role, "content": msg["content"]})
            await supabase_service.save_chat_message(
                session_id=active_session_id,
                role="user",
                content=req.message,
            )
        else:
            for msg in (req.history or []):
                role = "assistant" if msg["role"] in ["model", "assistant"] else "user"
                content = msg.get("parts", msg.get("content", ""))
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": req.message})

        if not skip_llm_loop:
            # --- TOOL CALLING LOOP ---
            # Wrapped in a 120-second hard timeout to prevent runaway requests
            # from holding connections indefinitely.
            MAX_LOOPS = 20
            AI_TIMEOUT_SECONDS = 120

            async def _run_ai_loop() -> str:
                nonlocal generated_schedule, captured_date_start
                loop_count = 0
                _raw_text = ""

                while loop_count < MAX_LOOPS:
                    loop_count += 1
                    logger.info(f"AI loop {loop_count}/{MAX_LOOPS}")
                    response = await _get_client().chat.completions.create(
                        model=settings.openai_model_id,
                        messages=messages,
                        tools=active_tools if active_tools else None,
                        response_format={"type": "json_object"},
                        temperature=0.1,
                    )

                    response_message = response.choices[0].message

                    if response_message.tool_calls:
                        messages.append(response_message.model_dump(exclude_none=True))
                        for tool_call in response_message.tool_calls:
                            func_name = tool_call.function.name
                            func_to_call = AVAILABLE_FUNCTIONS.get(func_name)

                            if not func_to_call:
                                logger.warning(f"Tool tidak dikenal: {func_name}")
                                messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": func_name,
                                    "content": json.dumps({"error": f"Tool '{func_name}' tidak tersedia."})
                                })
                                continue

                            try:
                                func_args = json.loads(tool_call.function.arguments)
                                
                                # Capture date_start from get_bali_context to align calendar dates
                                if func_name == "get_bali_context" and "date_start" in func_args:
                                    captured_date_start = func_args["date_start"]
                                    logger.info(f"Captured start date from get_bali_context: {captured_date_start}")
                                
                                # Inject user_requested_pois if calling get_smart_recommendations for POI category
                                if func_name == "get_smart_recommendations" and func_args.get("category", "poi") == "poi":
                                    if "user_requested_pois" not in func_args and poi_budget.get("full_targets"):
                                        func_args["user_requested_pois"] = poi_budget["full_targets"]
                                
                                logger.info(f"OpenAI panggil: {func_name}({func_args})")
                                func_result = await func_to_call(**func_args)
                                
                                # Capture the generated schedule and convert it to stringified summary for LLM
                                if func_name == "get_smart_recommendations" and func_args.get("category", "poi") == "poi":
                                    generated_schedule = func_result  # Keep the raw built days schedule list
                                    
                                    # Create a clean text summary of the generated schedule
                                    summary_lines = ["Berikut adalah jadwal harian yang telah disusun secara otomatis oleh backend Python untuk kamu:"]
                                    for day_data in func_result:
                                        summary_lines.append(f"\nHari {day_data.get('day')} (Tema: {day_data.get('theme', 'Eksplorasi')}):")
                                        for p in day_data.get("places", []):
                                            summary_lines.append(
                                                f"  - {p.get('visit_time')} ({p.get('visit_duration_mins')} mnt): {p.get('name')} "
                                                f"[{p.get('category')}]. Deskripsi: {p.get('description')}. Tips: {p.get('tips')}. Cost: Rp {p.get('estimated_cost_idr'):,} per orang."
                                            )
                                    summary_text = "\n".join(summary_lines)
                                    logger.info("Interception: get_smart_recommendations output summarized for LLM.")
                                    tool_content = summary_text
                                else:
                                    tool_content = json.dumps(func_result, default=str)
                                    
                            except Exception as e:
                                logger.error(f"Error pada {func_name}: {e}", exc_info=True)
                                tool_content = json.dumps({"error": "Tool execution failed."})

                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": func_name,
                                "content": tool_content
                            })
                    else:
                        _raw_text = response_message.content
                        break

                if loop_count >= MAX_LOOPS and not _raw_text:
                    raise HTTPException(
                        status_code=504,
                        detail="Proses AI terlalu lama. Silakan coba lagi."
                    )
                return _raw_text

            try:
                raw_text = await asyncio.wait_for(
                    _run_ai_loop(),
                    timeout=AI_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"AI loop timeout setelah {AI_TIMEOUT_SECONDS}s untuk user={user_id}"
                )
                raise HTTPException(
                    status_code=504,
                    detail="Permintaan AI melebihi batas waktu. Silakan coba lagi."
                )

            # --- PARSE & VALIDASI JSON ---
            try:
                raw_dict = json.loads(raw_text)
                raw_dict = sanitize_ai_output(raw_dict)
                parsed_data = FinalAIResponse.model_validate(raw_dict)
            except Exception as e:
                logger.warning(f"Sanitasi JSON gagal, coba validasi mentah: {e}")
                try:
                    parsed_data = FinalAIResponse.model_validate_json(raw_text)
                except Exception as e2:
                    logger.error(f"Validasi JSON total gagal: {e2}. raw_text={raw_text}")
                    raise HTTPException(status_code=500, detail=f"AI menghasilkan format JSON tidak valid: {e2}")
        else:
            raw_text = json.dumps(parsed_data.model_dump(), default=str)

        # --- POST-PROCESSING (hanya untuk itinerary) ---
        if parsed_data.response_type == "itinerary":

            if generated_schedule:
                logger.info("INJECTION: Injecting generated_schedule into parsed_data.itinerary_days")
                
                # Align dates based on captured_date_start
                start_date_val = datetime.now() + timedelta(days=1)
                if captured_date_start:
                    try:
                        start_date_val = datetime.strptime(captured_date_start, "%Y-%m-%d")
                    except Exception:
                        pass
                
                injected_days = []
                for day_data in generated_schedule:
                    day_num = day_data.get("day", 1)
                    formatted_date = (start_date_val + timedelta(days=day_num - 1)).strftime("%Y-%m-%d")
                    
                    places_list = []
                    for p in day_data.get("places", []):
                        places_list.append(PlaceItem(**p))
                        
                    injected_days.append(DailyItinerary(
                        day=day_num,
                        date=formatted_date,
                        theme=day_data.get("theme"),
                        places=places_list,
                        day_total_distance_km=None,
                        day_total_travel_time_mins=None,
                        day_full_polyline=None,
                        route_from_hotel=None,
                        odalan_warning=None,
                        weather_note=None
                    ))
                parsed_data.itinerary_days = injected_days

            # REPAIR: Deteksi jika semua hari punya places kosong (model skip JSON population)
            if parsed_data.itinerary_days:
                all_empty = all(
                    not day.places
                    for day in parsed_data.itinerary_days
                )
                if all_empty:
                    logger.warning("REPAIR TRIGGERED: Semua days punya places:[] kosong. Menjalankan repair...")
                    raw_dict = json.loads(raw_text)
                    raw_dict = sanitize_ai_output(raw_dict)
                    raw_dict = await repair_empty_places(raw_dict, messages)
                    raw_dict = sanitize_ai_output(raw_dict)  # sanitize ulang setelah repair
                    try:
                        parsed_data = FinalAIResponse.model_validate(raw_dict)
                    except Exception as val_err:
                        logger.error(f"REPAIR: Validasi setelah repair gagal: {val_err}")
                        # Lanjut dengan data asli (lebih baik kosong daripada error)

            # Deteksi district dari raw_text dan pesan user
            district_hint = extract_district_hint(raw_text, req.message)
            logger.info(f"District hint terdeteksi: {district_hint}")

            # 1. HOTEL GUARANTEE — Pastikan base_hotel selalu ada
            parsed_data = await guarantee_base_hotel(parsed_data, district_hint, preference_mode)

            # 2. BUDGET RECALCULATION — Calculate deterministically based on injected itinerary and guaranteed hotel
            if parsed_data.itinerary_days:
                num_days = len(parsed_data.itinerary_days)
                num_nights = max(1, num_days - 1)
                hotel_price = 850000
                if parsed_data.base_hotel and parsed_data.base_hotel.price_per_night_idr:
                    hotel_price = parsed_data.base_hotel.price_per_night_idr
                
                accommodation_idr = hotel_price * num_nights
                
                food_resto_cost = 0
                for day in parsed_data.itinerary_days:
                    for p in day.places:
                        if p.category == "restaurant" and p.estimated_cost_idr:
                            food_resto_cost += p.estimated_cost_idr
                food_idr = food_resto_cost * 2 + 50000 * 2 * num_days
                
                transport_idr = 150000 * num_days
                
                entrance_fee_cost = 0
                for day in parsed_data.itinerary_days:
                    for p in day.places:
                        if p.category == "attraction" and p.estimated_cost_idr:
                            entrance_fee_cost += p.estimated_cost_idr
                entrance_fee_idr = entrance_fee_cost * 2
                
                miscellaneous_idr = int((accommodation_idr + food_idr + transport_idr + entrance_fee_idr) * 0.12)
                total_budget_idr = accommodation_idr + food_idr + transport_idr + entrance_fee_idr + miscellaneous_idr
                
                parsed_data.budget_breakdown = BudgetBreakdown(
                    accommodation_idr=accommodation_idr,
                    food_idr=food_idr,
                    transport_idr=transport_idr,
                    entrance_fee_idr=entrance_fee_idr,
                    miscellaneous_idr=miscellaneous_idr
                )
                parsed_data.total_budget_idr = total_budget_idr
                logger.info(f"Budget recalculated: Total={total_budget_idr}")

            # 3. COORDINATE ENRICHMENT — Jika AI lupa isi lat/lng, backend ambil dari DB via place_id
            parsed_data = await enrich_place_coordinates(parsed_data)

            # 4. ROUTING INJECTION — Backend hitung rute (tidak berubah dari v7)
            hotel = parsed_data.base_hotel
            hotel_lat = hotel.latitude if hotel else None
            hotel_lng = hotel.longitude if hotel else None

            for day in (parsed_data.itinerary_days or []):
                waypoints = []
                if hotel_lat is not None and hotel_lng is not None:
                    waypoints.append({"lat": hotel_lat, "lng": hotel_lng, "name": hotel.name})

                for place in (day.places or []):
                    if place.latitude is not None and place.longitude is not None:
                        waypoints.append({
                            "lat": place.latitude,
                            "lng": place.longitude,
                            "name": place.name,
                        })

                if hotel_lat is not None and hotel_lng is not None:
                    waypoints.append({"lat": hotel_lat, "lng": hotel_lng, "name": hotel.name})

                if len(waypoints) < 2:
                    logger.warning(f"Hari {day.day}: waypoints tidak cukup, skip routing.")
                    continue

                try:
                    route_result = await tomtom_service.get_full_day_route(waypoints, [])
                    day.day_total_distance_km = route_result.get("total_distance_km")
                    day.day_total_travel_time_mins = route_result.get("total_travel_time_mins")
                    day.day_full_polyline = route_result.get("full_day_polyline")

                    segments = route_result.get("segments", [])

                    # Assign route_from_hotel (hotel → first place) = segments[0]
                    if segments:
                        seg0 = segments[0]
                        day.route_from_hotel = RouteSegment(
                            distance_km=seg0.get("distance_km"),
                            travel_time_mins=seg0.get("travel_time_mins"),
                            traffic_delay_mins=seg0.get("traffic_delay_mins"),
                            polyline=seg0.get("polyline"),
                            status=seg0.get("status"),
                        )

                    # Assign route_to_next per place.
                    # IMPORTANT: waypoints were built skipping places with null coords,
                    # so we track a separate waypoint_place_idx that only advances for
                    # places that actually made it into the waypoints array.
                    # seg_index = waypoint_place_idx + 1  (offset by 1 because waypoints[0] = hotel)
                    waypoint_place_idx = 0
                    for place in (day.places or []):
                        if place.latitude is not None and place.longitude is not None:
                            seg_index = waypoint_place_idx + 1
                            if seg_index < len(segments):
                                seg = segments[seg_index]
                                place.route_to_next = RouteSegment(
                                    distance_km=seg.get("distance_km"),
                                    travel_time_mins=seg.get("travel_time_mins"),
                                    traffic_delay_mins=seg.get("traffic_delay_mins"),
                                    polyline=seg.get("polyline"),
                                    status=seg.get("status"),
                                )
                            waypoint_place_idx += 1
                    logger.info(
                        f"Hari {day.day}: rute diinjeksi — "
                        f"{route_result.get('total_distance_km')} km, "
                        f"{route_result.get('total_travel_time_mins')} menit, "
                        f"{len(segments)} segmen."
                    )
                except Exception as e:
                    logger.error(f"Gagal menghitung rute untuk hari {day.day}: {e}")

        # --- AUTO-SAVE ---
        if is_authenticated:
            await supabase_service.save_chat_message(
                session_id=active_session_id,
                role="assistant",
                content=raw_text,
                itinerary_data=parsed_data.model_dump() if parsed_data.response_type == "itinerary" else None
            )
            if hasattr(parsed_data, "session_id"):
                parsed_data.session_id = active_session_id

            if parsed_data.response_type == "itinerary":
                if not is_editing:
                    saved = await supabase_service.save_user_itinerary(
                        user_id=user_id,
                        itinerary_data=parsed_data.model_dump(),
                        chat_session_id=active_session_id,
                    )
                    if saved.get("id"):
                        parsed_data.itinerary_id = saved["id"]
                        logger.info(f"Itinerary disimpan: {saved['id']} untuk user {user_id}")
                elif req.itinerary_id:
                    days = parsed_data.itinerary_days or []
                    await supabase_service.update_itinerary_full(
                        itinerary_id=req.itinerary_id,
                        user_id=user_id,
                        update_data={
                            "itinerary_data": parsed_data.model_dump(),
                            "total_budget_idr": parsed_data.total_budget_idr,
                            "days_count": len(days),
                        },
                    )
                    parsed_data.itinerary_id = req.itinerary_id
        else:
            logger.info("Guest mode: history dibaca dari frontend dan tidak disimpan.")
            if hasattr(parsed_data, "session_id"):
                parsed_data.session_id = req.session_id
            if hasattr(parsed_data, "itinerary_id"):
                parsed_data.itinerary_id = req.itinerary_id


        return parsed_data

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        # Already handled inside the try block, but caught here as safety net
        logger.error("asyncio.TimeoutError escaped AI loop.", exc_info=True)
        raise HTTPException(status_code=504, detail="Permintaan AI melebihi batas waktu.")
    except Exception as e:
        # Log full detail internally — DO NOT leak internal exception messages to client
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal. Silakan coba lagi.")


# ============================================================
# ITINERARY CRUD ENDPOINTS (tidak berubah dari v7)
# ============================================================

@app.get("/api/itineraries", response_model=List[ItinerarySummary], tags=["Itinerary CRUD"])
@limiter.limit(RATE_ITINERARY_READ)
async def list_itineraries(
    request: Request,
    include_public: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
):
    try:
        data = await supabase_service.list_user_itineraries(
            user_id=current_user.id,
            include_public=include_public,
            limit=limit,
            offset=offset,
        )
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"list_itineraries error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal mengambil daftar itinerary.")


@app.get("/api/itinerary/{itinerary_id}", tags=["Itinerary CRUD"])
@limiter.limit(RATE_ITINERARY_READ)
async def get_itinerary(
    request: Request,
    itinerary_id: str,
    current_user=Depends(get_current_user),
):
    # Validate UUID format to prevent probing with arbitrary strings
    try:
        uuid.UUID(itinerary_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Format itinerary ID tidak valid.")
    try:
        data = await supabase_service.get_itinerary_by_id(itinerary_id, current_user.id)
        if data is None:
            raise HTTPException(status_code=404, detail="Itinerary tidak ditemukan atau kamu tidak memiliki akses.")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_itinerary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal mengambil itinerary.")


@app.put("/api/itinerary/{itinerary_id}", tags=["Itinerary CRUD"])
@limiter.limit(RATE_ITINERARY_WRITE)
async def update_itinerary_manual(
    request: Request,
    itinerary_id: str,
    body: UpdateItineraryRequest,
    current_user=Depends(get_current_user),
):
    try:
        uuid.UUID(itinerary_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Format itinerary ID tidak valid.")
    try:
        days = body.itinerary_data.get("itinerary_days", [])
        update_payload = {
            "itinerary_data": body.itinerary_data,
            "days_count": len(days),
        }
        if body.title is not None:
            update_payload["title"] = body.title
        if body.total_budget_idr is not None:
            update_payload["total_budget_idr"] = body.total_budget_idr
        if body.is_public is not None:
            update_payload["is_public"] = body.is_public

        result = await supabase_service.update_itinerary_full(itinerary_id, current_user.id, update_payload)
        if result is None:
            raise HTTPException(status_code=404, detail="Itinerary tidak ditemukan atau kamu bukan pemiliknya.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_itinerary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal memperbarui itinerary.")


@app.patch("/api/itinerary/{itinerary_id}/visibility", tags=["Itinerary CRUD"])
@limiter.limit(RATE_ITINERARY_WRITE)
async def toggle_itinerary_visibility(
    request: Request,
    itinerary_id: str,
    body: ToggleVisibilityRequest,
    current_user=Depends(get_current_user),
):
    try:
        uuid.UUID(itinerary_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Format itinerary ID tidak valid.")
    try:
        result = await supabase_service.update_itinerary_visibility(itinerary_id, current_user.id, body.is_public)
        if result is None:
            raise HTTPException(status_code=404, detail="Itinerary tidak ditemukan atau kamu bukan pemiliknya.")
        return {"status": "updated", "itinerary_id": itinerary_id, "is_public": body.is_public}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"toggle_visibility error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal mengubah visibilitas itinerary.")


@app.post("/api/itinerary/{itinerary_id}/copy", tags=["Itinerary CRUD"])
@limiter.limit(RATE_ITINERARY_WRITE)
async def copy_public_itinerary(
    request: Request,
    itinerary_id: str,
    current_user=Depends(get_current_user),
):
    try:
        uuid.UUID(itinerary_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Format itinerary ID tidak valid.")
    try:
        result = await supabase_service.copy_public_itinerary(itinerary_id, current_user.id)
        if result is None:
            raise HTTPException(status_code=404, detail="Itinerary tidak ditemukan atau bukan itinerary publik.")
        return {"status": "copied", "new_itinerary_id": result.get("id"), "title": result.get("title")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"copy_itinerary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal menyalin itinerary.")


@app.delete("/api/itinerary/{itinerary_id}", tags=["Itinerary CRUD"])
@limiter.limit(RATE_ITINERARY_WRITE)
async def delete_itinerary(
    request: Request,
    itinerary_id: str,
    current_user=Depends(get_current_user),
):
    try:
        uuid.UUID(itinerary_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Format itinerary ID tidak valid.")
    try:
        success = await supabase_service.delete_itinerary(itinerary_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Itinerary tidak ditemukan atau kamu bukan pemiliknya.")
        return {"status": "deleted", "itinerary_id": itinerary_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_itinerary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal menghapus itinerary.")


# ============================================================
# CHAT MESSAGES ENDPOINT
# ============================================================

@app.get("/api/chat/sessions/{session_id}/messages", tags=["AI Chat"])
async def get_chat_messages(session_id: str):
    history = await supabase_service.get_chat_history(session_id)
    if not history:
        return []
    return history


# ============================================================
# PLACE SEARCH ENDPOINTS
# ============================================================

@app.get("/api/place/search", tags=["Place Search"])
@limiter.limit(RATE_SEARCH)
async def manual_search_place(
    request: Request,
    # max_length=200 prevents oversized queries being sent to the database
    query: str = Query(..., max_length=200, min_length=1),
    category: Literal["attraction", "hotel", "restaurant"] = Query("attraction"),
    current_user=Depends(get_current_user),
):
    try:
        results = await supabase_service.search_specific_place(query.strip(), category)
        return {"results": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"place_search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal mencari tempat.")


@app.get("/api/place/recommendations", tags=["Place Search"])
@limiter.limit(RATE_SEARCH)
async def get_place_recommendations(
    request: Request,
    query: str = Query(..., max_length=200, min_length=1),
    category: Literal["poi", "hotel", "restaurant"] = Query("poi"),
    limit: int = Query(10, ge=1, le=30),
    current_user=Depends(get_current_user),
):
    raw = await supabase_service.search_pois_semantic(query=query, limit=limit * 2)
    ranked = rank_pois_by_topsis(raw, category=category, top_n=limit)
    return {"results": ranked, "count": len(ranked), "query": query}


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", tags=["System"])
@limiter.limit(RATE_HEALTH)
async def health_check(request: Request):
    return {
        "status": "healthy",
        "version": "9.0",
        "service": "SobatNavi AI Agent",
        "features": [
            "radius_anchor_first_clustering",
            "topsis_ranking",
            "ai_meal_scheduling",
            "hotel_guarantee",
            "odalan_safety_check",
            "tomtom_routing",
            "rate_limiting",
            "security_headers",
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)