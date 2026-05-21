# api.py — SobatNavi AI Agent v8.0
# ============================================================
# PERUBAHAN UTAMA dari v7:
#   1. POI_BUDGET: Kalkulasi jumlah POI per hari secara programatik
#      (mempertimbangkan pace, durasi, dan preferensi user)
#   2. MEAL INJECTION: Backend secara aktif menyisipkan restoran
#      untuk sarapan/makan siang/makan malam — tidak lagi mengandalkan AI
#   3. HOTEL GUARANTEE: Jika AI tidak menghasilkan base_hotel,
#      backend mencarinya secara otomatis dari database
#   4. EDGE CASES: Trip setengah hari, trip sangat panjang (7+ hari),
#      database kosong/sedikit hasil, restoran tidak ditemukan di sekitar POI
# ============================================================

from server import validate_itinerary_safety
import asyncio
import os
import json
import logging
import math
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

from app.schemas.response_schema import FinalAIResponse, ItinerarySummary
from app.services.supabase_service import supabase_service
from app.services.tomtom_service import tomtom_service
from app.services.live_intel_service import live_intel_service
from app.engine.odalan_checker import extract_global_avoid_zones
from app.engine.recommender import cluster_and_rank_pois
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


def extract_trip_parameters_from_message(message: str) -> dict:
    """
    Parsing sederhana pesan user untuk mendeteksi:
    - Pace (santai, padat, dll)
    - Apakah user menyebut jumlah tempat spesifik
    - Apakah trip setengah hari

    Ini hanya fallback. Heidi (AI) tetap menginterpretasi pesan
    secara lebih akurat. Hasilnya dikirimkan ke AI sebagai konteks tambahan.
    """
    msg_lower = message.lower()

    pace = "normal"
    if any(k in msg_lower for k in ["santai", "slow", "rileks", "bersantai", "tenang"]):
        pace = "santai"
    elif any(k in msg_lower for k in ["padat", "penuh", "banyak tempat", "maksimal", "semua"]):
        pace = "padat"

    is_half_day = any(k in msg_lower for k in ["setengah hari", "half day", "beberapa jam", "sore aja", "pagi aja"])

    # Deteksi angka eksplisit: "3 tempat", "5 wisata", dll
    import re
    user_requested_pois = None
    patterns = [
        r"(\d+)\s*(tempat|lokasi|wisata|destinasi|spot|poi)",
        r"kunjungi\s+(\d+)",
        r"hanya\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg_lower)
        if match:
            user_requested_pois = int(match.group(1))
            break

    return {
        "pace": pace,
        "is_half_day": is_half_day,
        "user_requested_pois": user_requested_pois,
    }


# ============================================================
# MEAL SLOT DEFINITIONS
# ============================================================

MEAL_SLOTS = {
    # meal_type → (visit_time_24h, estimated_duration_mins, estimated_cost_idr, tags)
    "sarapan": {
        "visit_time": "08:00",
        "visit_duration_mins": 45,
        "estimated_cost_idr": 50000,
        "tags": ["sarapan", "kuliner", "pagi"],
        "label": "☕ Sarapan",
    },
    "makan_siang": {
        "visit_time": "12:30",
        "visit_duration_mins": 60,
        "estimated_cost_idr": 75000,
        "tags": ["makan siang", "kuliner", "restoran"],
        "label": "🍽️ Makan Siang",
    },
    "makan_malam": {
        "visit_time": "19:00",
        "visit_duration_mins": 75,
        "estimated_cost_idr": 100000,
        "tags": ["makan malam", "kuliner", "dinner"],
        "label": "🌙 Makan Malam",
    },
}


def assign_meal_slots(meals_per_day: int) -> list[str]:
    """
    Menentukan sesi makan mana yang akan disisipkan berdasarkan jumlah meals_per_day.
    """
    if meals_per_day <= 0:
        return []
    elif meals_per_day == 1:
        return ["makan_siang"]
    elif meals_per_day == 2:
        return ["makan_siang", "makan_malam"]
    else:
        return ["sarapan", "makan_siang", "makan_malam"]


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
    limit_per_day: int = 4,
    category: str = "poi",
    preference_mode: str = "standard",
) -> list[dict]:
    """
    Cari tempat wisata menggunakan Semantic Search (RAG), cluster per hari (DBSCAN),
    dan ranking berdasarkan TOPSIS multi-dimensi.

    limit_per_day akan di-override oleh poi_budget yang dikalkulasi backend.
    Namun AI tetap bisa menyesuaikan jika user minta secara eksplisit.
    """
    # Selalu fetch 3x lebih banyak dari yang dibutuhkan untuk memastikan
    # TOPSIS + DBSCAN punya cukup data untuk memilih yang terbaik
    fetch_count = limit_per_day * 3 * num_days
    fetch_count = max(30, min(fetch_count, 80))  # Clamp: 30-80

    if category == "hotel":
        raw = await supabase_service.search_amenities_semantic(query, "hotel", limit=fetch_count)
    elif category == "restaurant":
        raw = await supabase_service.search_amenities_semantic(query, "restaurant", limit=fetch_count)
    else:
        raw = await supabase_service.search_pois_semantic(query=query, limit=fetch_count)

    return cluster_and_rank_pois(
        raw,
        num_clusters=num_days,
        top_n_per_cluster=limit_per_day,
        category=category,
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
                "Cari tempat wisata Semantic Search, cluster per hari & TOPSIS. "
                "Parameter limit_per_day SUDAH DIKALKULASI oleh sistem (lihat poi_budget di context). "
                "Gunakan nilai dari poi_budget.attractions_per_day untuk parameter ini. "
                "Gunakan preference_mode='hidden_gem' untuk tempat tersembunyi, "
                "'luxury' untuk premium, 'budget' untuk hemat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_days": {"type": "integer"},
                    "limit_per_day": {
                        "type": "integer",
                        "description": "Jumlah atraksi per hari. WAJIB ambil dari poi_budget.attractions_per_day di context system."
                    },
                    "category": {"type": "string", "enum": ["poi", "hotel", "restaurant"]},
                    "preference_mode": {
                        "type": "string",
                        "enum": ["standard", "hidden_gem", "luxury", "budget"],
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

    # Informasikan AI tentang POI budget yang sudah dikalkulasi backend
    poi_budget_context = f"""
═══════════════════════════════════════════════
KONTEKS SISTEM: POI BUDGET (SUDAH DIKALKULASI — IKUTI INI)
═══════════════════════════════════════════════
Backend telah menghitung jumlah POI yang masuk akal berdasarkan pesan user:

  • Pace perjalanan        : {poi_budget.get('pace_label', 'normal')}
  • Atraksi per hari       : {poi_budget.get('attractions_per_day', 4)} tempat wisata
  • Sesi makan per hari    : {poi_budget.get('meals_per_day', 2)} sesi makan (diisi backend, JANGAN kamu isi)
  • Total places per hari  : {poi_budget.get('total_places_per_day', 6)} (atraksi + makan)
  • Preferensi anggaran    : {_budget_label}

INSTRUKSI TERKAIT:
  ✅ Saat memanggil get_smart_recommendations, gunakan limit_per_day={poi_budget.get('attractions_per_day', 4)}
  ✅ Saat memanggil get_smart_recommendations, WAJIB sertakan preference_mode="{preference_mode}"
  ✅ JANGAN sertakan restoran/warung makan di dalam array places — backend akan menyisipkannya otomatis
  ✅ Jika user bertanya jumlah tempat dan terasa sedikit, jelaskan bahwa sesi makan sudah disisipkan terpisah
  ✅ Fetch dari DB: backend menggunakan multiplier 3x untuk seleksi TOPSIS yang lebih baik
"""

    if is_editing:
        edit_context = json.dumps(current_itinerary, ensure_ascii=False, indent=2)
        mode_instruction = f"""
═══════════════════════════════════════════════
MODE: EDIT ITINERARY (via Chat)
═══════════════════════════════════════════════
Kamu sedang memodifikasi itinerary berikut:
{edit_context}

ATURAN EDIT:
1. HAPUS TEMPAT BERDASARKAN NAMA: Cari item di array `places` yang namanya cocok, keluarkan dari array.
2. HAPUS TEMPAT BERDASARKAN POSISI: Gunakan indeks array (0-based).
3. TAMBAH TEMPAT DENGAN NAMA SPESIFIK: WAJIB panggil `search_specific_place`. Jika tidak ditemukan, JANGAN mengarang data.
4. TAMBAH TEMPAT TANPA NAMA (tema/vibe): Panggil `get_nearby_places` atau `get_smart_recommendations`.
5. JANGAN menghitung atau mengisi field rute apapun.
6. Kembalikan SELURUH struktur itinerary yang sudah dimodifikasi.
7. Hotel (base_hotel) TIDAK BOLEH berubah kecuali user eksplisit minta ganti hotel.
8. Field `route_to_next`, `day_full_polyline`, `day_total_distance_km`, `day_total_travel_time_mins` HARUS null.
"""
    elif mode == "deep_research":
        mode_instruction = """
═══════════════════════════════════════════════
MODE: DEEP RESEARCH (Riset Mendalam)
═══════════════════════════════════════════════
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
═══════════════════════════════════════════════
MODE: GENERAL (Langsung Proses)
═══════════════════════════════════════════════
- Jika user MENYAPA atau NGOBROL BIASA: response_type="chat". JANGAN panggil tool.
- Jika user HANYA MINTA REKOMENDASI tanpa jadwal: panggil get_smart_recommendations, response_type="recommendation".
- Jika user MINTA ITINERARY LENGKAP → lanjut ke alur pembuatan.
  Jika tidak ada tanggal → asumsikan besok ({tomorrow_str}).
  Jika tidak ada durasi → asumsikan 1-2 hari.
  Jika tidak ada budget → asumsikan menengah.
  JANGAN BERTANYA jika data kurang! Langsung buatkan dengan asumsi!
"""

    return f"""
Kamu adalah **Heidi**, asisten perjalanan AI spesialis Bali dari SobatNavi.
Kepribadianmu: hangat, informatif, dan sangat paham budaya Bali.
Hari ini: {today_str}. Asumsi keberangkatan jika tidak disebutkan: besok ({tomorrow_str}).

{poi_budget_context}

{mode_instruction}

═══════════════════════════════════════════════
ATURAN MUTLAK (WAJIB DIPATUHI)
═══════════════════════════════════════════════
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
7. **RESTORAN JANGAN DIISI DI ARRAY PLACES**: Backend akan menyisipkan restoran secara otomatis berdasarkan poi_budget.meals_per_day. KAMU TIDAK PERLU dan TIDAK BOLEH menambahkan restoran ke dalam array places.
8. **KATEGORI TEMPAT**: Untuk field `category` di dalam objek tempat: HANYA "attraction", "hotel", atau "restaurant".
9. **SUGGESTED REPLIES**: Selalu isi `suggested_replies` dengan 3 saran pertanyaan relevan.
10. **ATRAKSI MINIMUM**: Setiap hari WAJIB memiliki minimal {poi_budget.get('min_attractions', 2)} atraksi. Jika kurang, panggil tool lagi dengan query berbeda.

═══════════════════════════════════════════════
ALUR KERJA PEMBUATAN ITINERARY
═══════════════════════════════════════════════
STEP 1 → get_bali_context(date_start, date_end, district)
STEP 2 → get_smart_recommendations(query, num_days, limit_per_day={poi_budget.get('attractions_per_day', 4)}, category="poi")
         ⚠️ JANGAN panggil get_smart_recommendations untuk category="restaurant" — backend handle ini.
STEP 3 → get_nearby_places(lat, lng, category="hotel") untuk dapatkan base_hotel
STEP 4 → validate_itinerary_safety(poi_ids, date_start, date_end)
STEP 5 → Susun urutan kunjungan yang logis secara geografis (JANGAN hitung rute)
STEP 6 → Tulis message_to_user Markdown (narasi storytelling min. 250 kata untuk itinerary)
STEP 7 → Lengkapi trip_title, suggested_replies

═══════════════════════════════════════════════
SKEMA JSON OUTPUT (WAJIB IKUTI PERSIS)
═══════════════════════════════════════════════
{schema_string}
"""


# ============================================================
# BACKEND MEAL INJECTION
# ============================================================

async def inject_meals_to_itinerary(
    parsed_data: "FinalAIResponse",
    poi_budget: dict,
    district_hint: str = "Bali",
    preference_mode: str = "standard",
) -> "FinalAIResponse":
    """
    Menyisipkan restoran ke dalam setiap hari itinerary secara programatik.
    Backend yang bertanggung jawab penuh atas meal slots — bukan AI.

    Strategi:
    1. Ambil koordinat sentroid dari semua atraksi di hari tersebut
    2. Cari restoran terdekat dari sentroid menggunakan search_amenities_nearby
    3. Fallback: semantic search dengan query district/tema jika nearby tidak ada hasil
    4. Sisipkan restoran di slot waktu yang tepat (siang / malam)
    """
    if not parsed_data.itinerary_days:
        return parsed_data

    meals_per_day = poi_budget.get("meals_per_day", 2)
    meal_slot_types = assign_meal_slots(meals_per_day)

    if not meal_slot_types:
        logger.info("meals_per_day=0, tidak ada restoran yang disisipkan.")
        return parsed_data

    for day in parsed_data.itinerary_days:
        if not day.places:
            continue

        # Kumpulkan koordinat atraksi yang valid di hari ini
        attraction_coords = [
            (p.latitude, p.longitude)
            for p in day.places
            if p.latitude is not None and p.longitude is not None and p.category == "attraction"
        ]

        # Hitung sentroid
        if attraction_coords:
            centroid_lat = sum(c[0] for c in attraction_coords) / len(attraction_coords)
            centroid_lng = sum(c[1] for c in attraction_coords) / len(attraction_coords)
        else:
            # Fallback: pakai koordinat Bali tengah jika tidak ada atraksi dengan koordinat
            centroid_lat = -8.4095
            centroid_lng = 115.1889
            logger.warning(f"Hari {day.day}: Tidak ada koordinat atraksi valid, pakai sentroid Bali.")

        # Cari restoran unik untuk setiap meal slot
        already_used_ids = set()
        restaurants_found = []

        try:
            # Pertama: coba nearby dalam radius 5km
            nearby_restaurants = await supabase_service.search_amenities_nearby(
                amenity_type="restaurant",
                lat=centroid_lat,
                lng=centroid_lng,
                radius_m=5000,
                limit=meals_per_day * 3,  # Ambil lebih banyak untuk fallback
            )
            restaurants_found = nearby_restaurants
        except Exception as e:
            logger.warning(f"Hari {day.day}: nearby restaurant gagal ({e}), coba semantic.")

        # Fallback: semantic search jika nearby tidak cukup
        if len(restaurants_found) < meals_per_day:
            try:
                _restaurant_query_map = {
                    "budget": f"warung makan murah {district_hint} Bali harga terjangkau",
                    "luxury": f"restoran fine dining mewah {district_hint} Bali premium",
                }
                semantic_query = _restaurant_query_map.get(preference_mode, f"restoran {district_hint} Bali")
                semantic_restaurants = await supabase_service.search_amenities_semantic(
                    query=semantic_query,
                    amenity_type="restaurant",
                    limit=meals_per_day * 3,
                )
                # Gabungkan, prioritaskan yang belum ada
                existing_ids = {r.get("place_id") for r in restaurants_found}
                for r in semantic_restaurants:
                    if r.get("place_id") not in existing_ids:
                        restaurants_found.append(r)
                        existing_ids.add(r.get("place_id"))
            except Exception as e:
                logger.warning(f"Hari {day.day}: semantic restaurant fallback juga gagal ({e}).")

        # Sisipkan restoran ke dalam places di posisi yang tepat
        for i, meal_type in enumerate(meal_slot_types):
            if not restaurants_found:
                logger.warning(f"Hari {day.day}: Tidak ada restoran ditemukan untuk slot {meal_type}.")
                break

            # Pilih restoran yang belum dipakai
            chosen = None
            for r in restaurants_found:
                rid = r.get("place_id", r.get("name", ""))
                if rid not in already_used_ids:
                    chosen = r
                    already_used_ids.add(rid)
                    break

            if not chosen:
                logger.warning(f"Hari {day.day}: Semua restoran sudah dipakai, skip slot {meal_type}.")
                break

            slot_config = MEAL_SLOTS[meal_type]
            metadata = chosen.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}

            # Buat PlaceItem untuk restoran ini
            from app.schemas.response_schema import PlaceItem
            restaurant_place = PlaceItem(
                place_id=chosen.get("place_id"),
                name=chosen.get("name", f"Restoran {meal_type.replace('_', ' ').title()}"),
                category="restaurant",
                latitude=chosen.get("latitude"),
                longitude=chosen.get("longitude"),
                district=chosen.get("district", district_hint),
                rating=chosen.get("rating"),
                image_url=chosen.get("image_url"),
                visit_time=slot_config["visit_time"],
                visit_duration_mins=slot_config["visit_duration_mins"],
                estimated_cost_idr=slot_config["estimated_cost_idr"],
                tags=slot_config["tags"],
                tips=f"Nikmati {slot_config['label']} di sini. Cocok untuk pengunjung yang ingin cita rasa lokal.",
                description=chosen.get("content", metadata.get("description", "")),
                route_to_next=None,
            )

            # Tentukan posisi sisipan berdasarkan meal slot
            current_places = list(day.places)
            if meal_type == "sarapan":
                # Sarapan di awal hari (index 0)
                current_places.insert(0, restaurant_place)
            elif meal_type == "makan_siang":
                # Makan siang di tengah (setelah ~50% atraksi)
                total_attractions = len([p for p in current_places if p.category == "attraction"])
                mid_idx = max(1, total_attractions // 2)
                # Cari posisi setelah mid_idx atraksi
                attraction_count = 0
                insert_pos = len(current_places)
                for idx, p in enumerate(current_places):
                    if p.category == "attraction":
                        attraction_count += 1
                        if attraction_count >= mid_idx:
                            insert_pos = idx + 1
                            break
                current_places.insert(insert_pos, restaurant_place)
            else:  # makan_malam
                # Makan malam di akhir hari
                current_places.append(restaurant_place)

            day.places = current_places
            logger.info(
                f"Hari {day.day}: Sisip restoran '{restaurant_place.name}' untuk slot {meal_type} "
                f"(lat={restaurant_place.latitude}, lng={restaurant_place.longitude})"
            )

    return parsed_data


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
        from app.schemas.response_schema import HotelItem
        metadata = hotel_data.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        parsed_data.base_hotel = HotelItem(
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
            price_level=hotel_data.get("price_level"),
            estimated_cost_per_night_idr=None,  # Akan diisi AI atau dibiarkan null
        )
        logger.info(f"Hotel otomatis dipilih backend: {parsed_data.base_hotel.name}")
    else:
        # Final fallback: jika benar-benar tidak ada hotel di database
        # Isi dengan placeholder agar tidak crash
        logger.error("Tidak ada hotel ditemukan di database! Mengisi placeholder.")
        from app.schemas.response_schema import HotelItem
        parsed_data.base_hotel = HotelItem(
            place_id=None,
            name="Hotel (Hubungi Admin)",
            latitude=ref_lat,
            longitude=ref_lng,
            district=district_hint,
            rating=None,
            image_url=None,
            description="Sistem tidak dapat menemukan hotel di database. Silakan pilih hotel manual.",
            price_level=None,
            estimated_cost_per_night_idr=None,
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

        # --- KALKULASI POI BUDGET (PRE-AI) ---
        trip_params = extract_trip_parameters_from_message(req.message)
        # Deteksi jumlah hari dari history / pesan untuk kalkulasi budget
        # Default ke 2 hari; AI yang nanti menentukan secara akurat
        num_days_hint = 2
        import re
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
        system_prompt = build_heidi_prompt(req.mode, is_editing, req.current_itinerary, poi_budget, preference_mode)
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
                logger.error(f"Validasi JSON total gagal: {e2}. raw_text[:500]={raw_text[:500]}")
                raise HTTPException(status_code=500, detail=f"AI menghasilkan format JSON tidak valid: {e2}")

        # --- POST-PROCESSING (hanya untuk itinerary) ---
        if parsed_data.response_type == "itinerary" and parsed_data.itinerary_days:

            # Deteksi district dari raw_text dan pesan user
            district_hint = extract_district_hint(raw_text, req.message)
            logger.info(f"District hint terdeteksi: {district_hint}")

            # 1. HOTEL GUARANTEE — Pastikan base_hotel selalu ada
            parsed_data = await guarantee_base_hotel(parsed_data, district_hint, preference_mode)

            # 2. MEAL INJECTION — Sisipkan restoran ke setiap hari
            # Lewati jika mode editing untuk menghindari duplikasi
            if not is_editing:
                parsed_data = await inject_meals_to_itinerary(
                    parsed_data, poi_budget, district_hint, preference_mode
                )

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
                    from app.schemas.response_schema import RouteSegment
                    for idx, place in enumerate(day.places or []):
                        seg_index = idx + 1
                        if seg_index < len(segments):
                            seg = segments[seg_index]
                            place.route_to_next = RouteSegment(
                                distance_km=seg.get("distance_km"),
                                travel_time_mins=seg.get("travel_time_mins"),
                                traffic_delay_mins=seg.get("traffic_delay_mins"),
                                polyline=seg.get("polyline"),
                                status=seg.get("status"),
                            )
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
    try:
        raw = await supabase_service.search_pois_semantic(query=query.strip(), limit=limit * 2)
        ranked = cluster_and_rank_pois(raw, num_clusters=1, top_n_per_cluster=limit, category=category)
        return {"results": ranked, "count": len(ranked), "query": query}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"place_recommendations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal mengambil rekomendasi tempat.")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", tags=["System"])
@limiter.limit(RATE_HEALTH)
async def health_check(request: Request):
    return {
        "status": "healthy",
        "version": "8.0",
        "service": "SobatNavi AI Agent",
        "features": [
            "poi_budget_calculator",
            "meal_injection",
            "hotel_guarantee",
            "odalan_safety_check",
            "tomtom_routing",
            "rate_limiting",
            "security_headers",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)