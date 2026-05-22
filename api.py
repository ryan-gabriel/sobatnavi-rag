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

from app.schemas.response_schema import BaseHotel, FinalAIResponse, ItinerarySummary, RouteSegment
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
    user_requested_pois: int = None,
    is_half_day: bool = False,
) -> dict:
    """
    Menghitung 'anggaran' POI per hari secara programatik.

    Returns dict:
        - attractions_per_day: jumlah atraksi (bukan restoran) per hari
        - meals_per_day: berapa sesi makan yang akan disisipkan
        - total_places_per_day: total places (atraksi + restoran)
        - fetch_buffer_multiplier: multiplier untuk fetch ke DB (selalu lebih banyak dari yang ditampilkan)
    """
    if is_half_day:
        # Trip setengah hari: maksimal 2 atraksi + 1 makan
        return {
            "attractions_per_day": 2,
            "meals_per_day": 1,
            "total_places_per_day": 3,
            "fetch_buffer_multiplier": 3,
            "pace_label": "setengah hari",
        }

    if user_requested_pois is not None:
        # User secara eksplisit meminta jumlah tempat tertentu
        user_req = max(1, min(user_requested_pois, 8))  # Clamp: 1-8
        meals = 2 if user_req >= 4 else 1
        return {
            "attractions_per_day": user_req,
            "meals_per_day": meals,
            "total_places_per_day": user_req + meals,
            "fetch_buffer_multiplier": 3,
            "pace_label": f"custom ({user_req} tempat)",
        }

    pace_key = pace.lower()
    if pace_key not in PACE_CONFIG:
        pace_key = "normal"

    min_poi, ideal_poi, max_poi = PACE_CONFIG[pace_key]

    # Untuk trip panjang (5+ hari), kurangi sedikit agar tidak terlalu melelahkan
    if num_days >= 5:
        ideal_poi = max(min_poi, ideal_poi - 1)

    # Jumlah sesi makan:
    # - pace santai: 1 makan siang (sarapan dan makan malam di hotel)
    # - pace normal/padat: 1 makan siang + 1 makan malam (atau sarapan luar)
    meals = 1 if pace_key == "santai" else 2

    return {
        "attractions_per_day": ideal_poi,
        "meals_per_day": meals,
        "total_places_per_day": ideal_poi + meals,
        "fetch_buffer_multiplier": 3,  # Fetch 3x lebih banyak dari DB untuk ranking
        "pace_label": pace_key,
        "min_attractions": min_poi,
        "max_attractions": max_poi,
    }


async def extract_trip_parameters_from_message(message: str, db_districts: list[str]) -> dict:
    try:
        logger.info(">>> MENJALANKAN EXTRACT PARAMS <<<")
        client = _get_client()
        prompt = (
            "Analyze the travel planning message and return a JSON object containing exactly these fields:\n"
            "- intent: strictly one of 'edit', 'create', or 'chat'.\n"
            "   * Choose 'edit' if the user explicitly wants to add, delete, replace, or modify places/days.\n"
            "   * Choose 'create' if the user wants to plan a new itinerary.\n"
            "   * Choose 'chat' for general greetings.\n"
            "- pace: string, one of 'santai', 'padat', 'normal'.\n"
            "- is_half_day: boolean, true if half-day trip.\n"
            "- user_requested_pois: integer or null.\n"
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
        if user_requested_pois is not None:
            try: user_requested_pois = int(user_requested_pois)
            except: user_requested_pois = None
                
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
    # Jika LLM gagal paham, tapi user jelas-jelas bilang "hapus/tambah",
    # kita paksa intent menjadi "edit".
    msg_lower = message.lower()
    if any(k in msg_lower for k in ["hapus", "tambah", "ganti", "ubah", "delete", "remove"]):
        logger.info(f"Hybrid Override: Kata kunci edit terdeteksi, memaksa intent='edit'")
        intent = "edit"
        
    if intent not in ["edit", "create", "chat"]:
        intent = "chat"
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
        )


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
                    "query": {"type": "string", "description": "Kata kunci tema wisata (misal: 'pantai sunset', 'budaya ubud')"},
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
# HEIDI SYSTEM PROMPT
# ============================================================

def build_heidi_prompt(
    mode: str,
    is_editing: bool,
    current_itinerary: Optional[Dict],
    poi_budget: dict,
    preference_mode: str = "standard",
) -> str:
    schema_string = json.dumps(FinalAIResponse.model_json_schema(), indent=2)
    today_str = datetime.now().strftime("%Y-%m-%d")
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Map preference_mode ke label yang mudah dibaca
    _budget_label_map = {
        "budget": "Hemat (budget) — prioritaskan tempat terjangkau & value for money",
        "luxury": "Mewah (luxury) — prioritaskan tempat premium & eksklusif",
        "standard": "Menengah (moderate) — keseimbangan harga dan kualitas",
    }
    _budget_label = _budget_label_map.get(preference_mode, _budget_label_map["standard"])

    # Informasikan AI tentang perubahan arsitektur v9.0
    poi_budget_context = f"""
## KONTEKS: RADIUS & ANCHOR-FIRST CLUSTERING
Backend menggunakan metode clustering baru untuk menghasilkan
pool POI + restoran yang geographically tight per hari.

  • Pace perjalanan        : {poi_budget.get('pace_label', 'normal')}
  • Atraksi per hari       : {poi_budget.get('attractions_per_day', 4)} tempat wisata (default, bisa dinamis 2-4)
  • Preferensi anggaran    : {_budget_label}

PENTING:
  • Tool get_smart_recommendations kini mengembalikan data PER HARI:
     [{{"day": 1, "anchor": {{...}}, "pois": [...], "restaurants": [...]}}, ...]
  • Restoran SUDAH TERMASUK di output tool — KAMU WAJIB memasukkannya ke places[]
  • Sisipkan restoran di slot waktu makan: Lunch (12:00-13:30), Dinner (18:30-20:00)
  • Urutkan semua places berdasarkan visit_time secara kronologis
  • Saat memanggil get_smart_recommendations, WAJIB sertakan preference_mode="{preference_mode}"
"""

    if is_editing:
        edit_context = json.dumps(current_itinerary, ensure_ascii=False, indent=2)
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
    elif mode == "deep_research":
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
    else:
        mode_instruction = f"""
## MODE: GENERAL (Langsung Proses)
- Jika user MENYAPA atau NGOBROL BIASA: response_type="chat". JANGAN panggil tool.
- Jika user HANYA MINTA REKOMENDASI tanpa jadwal: panggil get_smart_recommendations, response_type="recommendation".
- Jika user MINTA ITINERARY LENGKAP → lanjut ke alur pembuatan.
  Jika tidak ada tanggal → asumsikan besok ({tomorrow_str}).
  Jika tidak ada durasi → asumsikan 1-2 hari.
  Jika tidak ada budget → asumsikan menengah.
  JANGAN BERTANYA jika data kurang! Langsung buatkan dengan asumsi!
"""

    # ── Dynamic rule 12 & workflow: completely different in edit vs. create mode ──
    if is_editing:
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
STEP 4 → JIKA USER MINTA TAMBAH TEMPAT SECARA UMUM/GENERIK: Kamu WAJIB memanggil `get_smart_recommendations` atau `get_nearby_places`. DILARANG KERAS bertanya kembali kepada user atau menawarkan pilihan! Pilih 1 tempat terbaik yang paling relevan dengan tema yang diminta (misal: "air terjun di Bangli"), masukkan tempat tersebut ke dalam itinerary, dan langsung konfirmasi perubahannya di `message_to_user`. Tugasmu adalah eksekusi, bukan berdiskusi.
STEP 5 → IMMUTABLE CLONING (MANDATORY): Untuk hari-hari (Day) atau data lain yang TIDAK diminta untuk diubah oleh user, kamu WAJIB menyalin (clone) seluruh strukturnya secara persis 100% tanpa mengubah data, urutan jam, nama, atau narasinya sedikit pun!
STEP 6 → LARANGAN ROUTING: Biarkan semua field rute harian (`route_to_next`, `day_full_polyline`, `day_total_distance_km`, `day_total_travel_time_mins`) selalu bernilai `null` karena backend TomTom yang akan menghitung ulang jalurnya secara otomatis setelah JSON divalidasi.
STEP 7 → Tulis narasi konfirmasi hangat di `message_to_user` dan kembalikan struktur JSON penuh. Pastikan `response_type` tetap `"itinerary"`.
ATURAN KRITIKAL: DILARANG KERAS menggunakan frasa seperti "Tunggu sebentar", "Sedang diproses", atau "Aku akan segera kirimkan". Langsung berikan konfirmasi tegas bahwa jadwal SUDAH berhasil diperbarui!
"""
    else:
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

    return f"""
Kamu adalah **Heidi**, asisten perjalanan AI spesialis Bali dari SobatNavi.
Kepribadianmu: hangat, informatif, dan sangat paham budaya Bali.
Hari ini: {today_str}. Asumsi keberangkatan jika tidak disebutkan: besok ({tomorrow_str}).

{poi_budget_context}

{mode_instruction}

## ATURAN MUTLAK (WAJIB DIPATUHI)
1. **FORMAT JSON**: Balas HANYA dengan JSON murni (tidak ada teks di luar JSON, tidak ada ```json```)
2. **MARKDOWN WAJIB di `message_to_user`**: Gunakan **bold**, *italic*, ## heading, - list, emoji.
3. **ANTI-HALUSINASI**: DILARANG mengarang nama tempat, place_id, latitude, longitude. Semua dari Tool.
4. **SATU HOTEL**: Pilih SATU `base_hotel` untuk SEMUA hari. Hotel TIDAK BOLEH muncul di dalam `places` harian.
5. **LARANGAN ROUTING**: JANGAN isi field rute apapun. `route_to_next`, `day_full_polyline`, `day_total_distance_km`, `day_total_travel_time_mins` WAJIB null.
6. **DATA WAJIB DI SETIAP PlaceItem (TIDAK BOLEH NULL)**:
   - `rating` & `district`: Dari data tool.
   - `tags`: 3-5 label dari deskripsi. WAJIB DIISI.
   - `estimated_cost_idr`: Estimasi biaya per orang (pura ~15000, pantai ~25000, museum ~50000, makan ~75000). WAJIB DIISI.
   - `visit_duration_mins`: Durasi estimasi (pura kecil=45, pantai=75, museum=90, restoran=50). WAJIB DIISI.
   - `visit_time`: HH:MM 24-jam sesuai urutan (pagi mulai 08:00). WAJIB DIISI.
   - `tips`: Satu kalimat tip berguna. WAJIB DIISI.
   - `image_url`: Dari tool jika ada.
7. **HARI WAJIB SESUAI PERMINTAAN (CRITICAL)**:
   Jika user meminta N hari, kamu WAJIB menghasilkan TEPAT N objek `day` di dalam `itinerary_days`.
   DILARANG KERAS mengurangi jumlah hari. Jika pool POI kurang, variasikan query ke tool.
8. **POI BUDGETING DINAMIS**:
   Default: 2-4 atraksi per hari. TAPI jika user eksplisit minta jumlah tertentu
   (misal "buat padat 5 tempat"), PATUHI permintaan user tersebut.
9. **RESTORAN WAJIB DIMASUKKAN KE PLACES**:
   Tool get_smart_recommendations menyediakan pool `restaurants[]` per hari.
   KAMU WAJIB memasukkan restoran dari pool ini ke dalam array `places` pada waktu makan:
   - Makan Siang (Lunch): visit_time antara 12:00 - 13:30
   - Makan Malam (Dinner): visit_time antara 18:30 - 20:00
   Urutkan semua places (atraksi + restoran) secara kronologis berdasarkan `visit_time`.
10. **KATEGORI TEMPAT**: Untuk field `category` di dalam objek tempat: HANYA "attraction", "hotel", atau "restaurant".
11. **SUGGESTED REPLIES**: Selalu isi `suggested_replies` dengan 3 saran pertanyaan relevan.
{rule_12_dynamic}

{workflow_instruction}


SANGAT PENTING: Untuk SETIAP tempat yang kamu sebutkan dalam message_to_user, kamu WAJIB memasukkan data tempat tersebut ke dalam array itinerary_days dengan struktur JSON yang lengkap. Jika kamu tidak memasukkannya ke JSON, maka itinerary dianggap tidak valid.

SKEMA JSON OUTPUT (WAJIB IKUTI PERSIS)
{schema_string}
"""

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
# EXTRACT DISTRICT HINT dari raw text / itinerary
# ============================================================

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


# ============================================================
# MAIN CHAT ENDPOINT
# ============================================================

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
        
        # Ekstrak itinerary_id dari state jika frontend tidak mengirimkannya secara eksplisit
        if not req.itinerary_id and _effective_itinerary:
            req.itinerary_id = _effective_itinerary.get("itinerary_id")
            
        is_editing = _effective_itinerary is not None

        # --- KALKULASI POI BUDGET (PRE-AI) ---
        db_districts = await supabase_service.get_db_districts()
        trip_params = await extract_trip_parameters_from_message(req.message, db_districts)

        if trip_params.get("intent") == "edit" and not is_editing:
            return FinalAIResponse(
                response_type="chat",
                message_to_user="Maaf, saat ini belum ada itinerary yang sedang kita susun. Kamu harus membuat itinerary liburan terlebih dahulu sebelum bisa mengubah tempat.",
                suggested_replies=["Buatkan itinerary 3 hari", "Rekomendasi tempat di Ubud"],
                itinerary_days=None,
                base_hotel=None,
                trip_title=None
            )

        if trip_params.get("intent") == "edit" and not is_editing:
            logger.info(f"Early return: Menolak request edit '{req.message}' karena itinerary kosong.")
            return FinalAIResponse(
                response_type="chat",
                message_to_user="Maaf, saat ini belum ada itinerary yang sedang kita susun. Kamu harus membuat itinerary liburan terlebih dahulu sebelum bisa mengubahnya. Mau aku buatkan untuk berapa hari?",
                suggested_replies=["Buatkan itinerary 3 hari", "Rekomendasi tempat di Ubud", "Cari pantai bagus"],
                itinerary_days=None,
                base_hotel=None,
                trip_title=None
            )
        
        # JIKA GAGAL DETEKSI LOKASI & LOKASI PENTING UNTUK MEMBUAT ITINERARY BARU
        if not trip_params.get("detected_location") and req.mode == "general" and trip_params.get("intent") == "create" and not is_editing:
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
        num_days_hint = 2
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

        # --- BUILD SYSTEM PROMPT ---
        system_prompt = build_heidi_prompt(req.mode, is_editing, _effective_itinerary, poi_budget, preference_mode)
        messages = [{"role": "system", "content": system_prompt}]
        active_session_id = None

        # --- MANAJEMEN HISTORY ---
        if is_authenticated:
            active_session_id = await supabase_service.get_or_create_chat_session(user_id, req.session_id)
            db_history = await supabase_service.get_chat_history(active_session_id)
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

        # --- TOOL CALLING LOOP ---
        # Wrapped in a 120-second hard timeout to prevent runaway requests
        # from holding connections indefinitely.
        MAX_LOOPS = 20
        AI_TIMEOUT_SECONDS = 120

        async def _run_ai_loop() -> str:
            loop_count = 0
            _raw_text = ""

            while loop_count < MAX_LOOPS:
                loop_count += 1
                logger.info(f"AI loop {loop_count}/{MAX_LOOPS}")
                response = await _get_client().chat.completions.create(
                    model=settings.openai_model_id,
                    messages=messages,
                    tools=OPENAI_TOOLS,
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
                            logger.info(f"OpenAI panggil: {func_name}({func_args})")
                            func_result = await func_to_call(**func_args)
                        except Exception as e:
                            # Log full detail server-side; return sanitized error to AI
                            logger.error(f"Error pada {func_name}: {e}", exc_info=True)
                            func_result = {"error": "Tool execution failed."}

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": func_name,
                            "content": json.dumps(func_result, default=str)
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

        # --- POST-PROCESSING (hanya untuk itinerary) ---
        if parsed_data.response_type == "itinerary" and parsed_data.itinerary_days:

            # Deteksi district dari raw_text dan pesan user
            district_hint = extract_district_hint(raw_text, req.message)
            logger.info(f"District hint terdeteksi: {district_hint}")

            # 1. HOTEL GUARANTEE — Pastikan base_hotel selalu ada
            if is_editing and _effective_itinerary and _effective_itinerary.get("base_hotel") and not parsed_data.base_hotel:
                from app.schemas.response_schema import BaseHotel
                parsed_data.base_hotel = BaseHotel(**_effective_itinerary["base_hotel"])
                logger.info("Hotel dikembalikan dari state sebelumnya karena AI lupa menyertakannya saat edit.")

            parsed_data = await guarantee_base_hotel(parsed_data, district_hint, preference_mode)

            # 2. MEAL SCHEDULING — v9.0: Handled by Heidi AI
            # Restaurants are delivered in the per-day pool from
            # generate_clustered_pool_delivery. AI places them at
            # Lunch (12:00-13:30) and Dinner (18:30-20:00).

            # 3. ROUTING INJECTION — Backend hitung rute (tidak berubah dari v7)
            hotel = parsed_data.base_hotel
            hotel_lat = hotel.latitude if hotel else None
            hotel_lng = hotel.longitude if hotel else None

            for day in parsed_data.itinerary_days:
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