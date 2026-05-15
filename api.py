import asyncio
import os
import json
import logging
from fastapi import FastAPI, HTTPException, Depends, Security, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict, Optional, Literal

from app.schemas.response_schema import FinalAIResponse, ItinerarySummary
from app.services.supabase_service import supabase_service
from app.services.tomtom_service import tomtom_service
from app.services.live_intel_service import live_intel_service
from app.engine.odalan_checker import extract_global_avoid_zones
from app.engine.recommender import cluster_and_rank_pois
from app.core.config import settings

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SobatNavi AI Agent API",
    version="7.0",
    description="AI Travel Assistant API untuk Bali — Heidi",
)

# Validasi OPENAI_API_KEY saat startup (warn, tidak crash)
if not settings.openai_api_key:
    logger.warning(
        "OPENAI_API_KEY belum diset di .env! "
        "Endpoint /api/chat akan gagal saat dipanggil."
    )

# Lazy OpenAI client — dibuat saat pertama kali dipakai
_openai_client = None

def _get_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY belum dikonfigurasi di .env")
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client

# HTTPBearer dengan auto_error=False → tidak lempar 403 jika header kosong
security = HTTPBearer(auto_error=False)


# =====================================================================
# AUTH MIDDLEWARE
# =====================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    """
    Auth WAJIB — untuk endpoint CRUD itinerary.
    Lempar 401 jika token tidak ada atau tidak valid.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Token autentikasi diperlukan.")
    token = credentials.credentials
    try:
        user_response = supabase_service.client.auth.get_user(token)
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
    """
    Auth OPSIONAL — untuk /api/chat.
    Jika tidak ada token → return None (guest mode, tidak disimpan).
    Jika ada token → validasi dan return user object.
    """
    if not credentials or not credentials.credentials:
        return None
    try:
        user_response = supabase_service.client.auth.get_user(credentials.credentials)
        if user_response and user_response.user:
            return user_response.user
        return None
    except Exception:
        # Token invalid → treat as guest, jangan crash
        return None


# =====================================================================
# REQUEST SCHEMAS
# =====================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="Pesan user ke Heidi")
    mode: Literal["general", "deep_research"] = Field(
        "general",
        description="'general': langsung buat itinerary. 'deep_research': tanya parameter dulu."
    )
    history: Optional[List[Dict]] = Field(
        default_factory=list,
        description="Riwayat chat sebelumnya. Format: [{role: 'user'|'model', parts: '...'}]"
    )
    current_itinerary: Optional[Dict] = Field(
        None,
        description="Data itinerary saat ini (wajib diisi untuk mode edit via chat)"
    )
    itinerary_id: Optional[str] = Field(
        None,
        description="UUID itinerary yang sedang diedit (wajib jika current_itinerary diisi)"
    )


class UpdateItineraryRequest(BaseModel):
    title: Optional[str] = Field(None, description="Judul baru itinerary")
    itinerary_data: Dict = Field(..., description="Seluruh objek itinerary yang sudah dimodifikasi")
    total_budget_idr: Optional[int] = Field(None, description="Total budget baru dalam IDR")
    is_public: Optional[bool] = Field(None, description="Toggle visibilitas publik/privat")


class ToggleVisibilityRequest(BaseModel):
    is_public: bool = Field(..., description="True = bagikan ke publik, False = privat")


# =====================================================================
# AI TOOLS (dipanggil oleh Gemini secara otomatis)
# =====================================================================

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
        "avoid_zones_info": "Daftar bbox zona hindari Odalan untuk dimasukkan ke calculate_batch_routes"
    }


async def get_smart_recommendations(
    query: str,
    num_days: int = 1,
    limit_per_day: int = 4,
    category: str = "poi"
) -> list[dict]:
    """
    Cari tempat wisata menggunakan Semantic Search (RAG), lalu cluster per hari (DBSCAN),
    dan ranking berdasarkan TOPSIS multi-dimensi.
    
    Params:
        query: Kata kunci (misal: 'pantai sunset', 'budaya ubud', 'kuliner seminyak')
        num_days: Jumlah hari itinerary (menentukan jumlah cluster)
        limit_per_day: Jumlah tempat per hari (disarankan 3-5)
        category: 'poi', 'hotel', atau 'restaurant'
    """
    raw_pois = await supabase_service.search_pois_semantic(query=query, limit=40)
    return cluster_and_rank_pois(
        raw_pois,
        num_clusters=num_days,
        top_n_per_cluster=limit_per_day,
        category=category,
    )


async def search_specific_place(query: str, category: str = "attraction") -> list[dict]:
    """
    Cari tempat SPESIFIK berdasarkan nama (untuk mode edit: ganti/tambah tempat - TC-10).
    Gunakan ini ketika user menyebut nama tempat tertentu.
    
    Params:
        query: Nama tempat (misal: 'Bebek Tepi Sawah', 'Tanah Lot')
        category: 'attraction', 'hotel', atau 'restaurant'
    """
    return await supabase_service.search_specific_place(query, category)


async def get_nearby_food_and_lodging(
    lat: float, lng: float, category: str, radius_km: float = 3.0
) -> list[dict]:
    """
    Cari hotel atau restoran terdekat dari koordinat tertentu.
    
    Params:
        lat, lng: Koordinat pusat pencarian (biasanya pusat area kunjungan)
        category: 'hotel' atau 'restaurant'
        radius_km: Radius pencarian dalam km (default 3km)
    """
    return await supabase_service.search_amenities_nearby(
        category, lat, lng, radius_km * 1000, limit=5
    )


async def calculate_batch_routes(
    waypoints: list[dict],
    avoid_zones: list[str] = []
) -> dict:
    """
    WAJIB DIPANGGIL untuk menghitung rute satu hari perjalanan secara efisien.
    Panggil SATU KALI PER HARI dengan semua waypoints hari tersebut.
    
    Format waypoints: [{"lat": -8.5, "lng": 115.2, "name": "Tanah Lot"}, ...]
    Urutan: Hotel (start) → POI1 → POI2 → ... → POIn → Hotel (end)
    
    Returns:
        {
          "total_distance_km": float,
          "total_travel_time_mins": int,
          "segments": [{"distance_km", "travel_time_mins", "polyline": [{lat,lng}...], "status"}, ...],
          "full_day_polyline": [{lat,lng}, ...]  <- untuk render di peta
        }
    """
    return await tomtom_service.get_full_day_route(waypoints, avoid_zones or [])



# =====================================================================
# OPENAI TOOLS SCHEMA MAP
# =====================================================================

AVAILABLE_FUNCTIONS = {
    "get_bali_context": get_bali_context,
    "get_smart_recommendations": get_smart_recommendations,
    "search_specific_place": search_specific_place,
    "get_nearby_food_and_lodging": get_nearby_food_and_lodging,
    "calculate_batch_routes": calculate_batch_routes
}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_bali_context",
            "description": "Ambil info cuaca & daftar zona hindari Odalan.",
            "parameters": {
                "type": "object",
                "properties": {"date_start": {"type": "string"}, "date_end": {"type": "string"}, "district": {"type": "string"}},
                "required": ["date_start", "date_end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_smart_recommendations",
            "description": "Cari tempat wisata Semantic Search, cluster per hari & TOPSIS.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "num_days": {"type": "integer"}, "limit_per_day": {"type": "integer"}, "category": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_specific_place",
            "description": "Cari tempat SPESIFIK berdasarkan nama.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "category": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearby_food_and_lodging",
            "description": "Cari hotel atau restoran terdekat dari koordinat.",
            "parameters": {
                "type": "object",
                "properties": {"lat": {"type": "number"}, "lng": {"type": "number"}, "category": {"type": "string"}, "radius_km": {"type": "number"}},
                "required": ["lat", "lng", "category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_batch_routes",
            "description": "Kira jarak rute satu hari perjalanan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "waypoints": {"type": "array", "items": {"type": "object", "properties": {"lat": {"type": "number"}, "lng": {"type": "number"}, "name": {"type": "string"}}}},
                    "avoid_zones": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["waypoints"]
            }
        }
    }
]



# =====================================================================
# HEIDI SYSTEM PROMPT
# =====================================================================

def build_heidi_prompt(mode: str, is_editing: bool, current_itinerary: Optional[Dict]) -> str:
    schema_string = json.dumps(FinalAIResponse.model_json_schema(), indent=2)

    if is_editing:
        edit_context = json.dumps(current_itinerary, ensure_ascii=False, indent=2)
        mode_instruction = f"""
═══════════════════════════════════════════════
MODE: EDIT ITINERARY (via Chat)
═══════════════════════════════════════════════
Kamu sedang memodifikasi itinerary berikut:
{edit_context}

ATURAN EDIT:
1. HAPUS TEMPAT: Keluarkan item dari array `places` tanpa memanggil tool apapun.
2. TAMBAH/GANTI TEMPAT: WAJIB panggil `search_specific_place(query=<nama tempat>, category=<kategori>)` untuk mendapatkan data ASLI dari database. JANGAN mengarang koordinat atau ID.
3. Setelah modifikasi, hitung ulang rute harian dengan `calculate_batch_routes`.
4. Kembalikan SELURUH struktur itinerary yang sudah dimodifikasi dengan response_type="itinerary".
5. Hotel (base_hotel) TIDAK BOLEH berubah kecuali user secara eksplisit meminta ganti hotel.
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

- Jika SEMUA variabel sudah ada → lanjut buat itinerary (ikuti alur di bawah).

- Jika ADA YANG KURANG → response_type="clarifying". Isi `clarifying_questions`.
  `message_to_user` WAJIB Markdown. Contoh:
  "Hampir siap! 😊\n\n**Aku masih butuh beberapa info dulu:**\n- 📍 Area mana di Bali yang kamu tuju?\n- 📅 Tanggal berapa berangkat?"

- Jika user hanya menyapa → response_type="chat".
  `message_to_user` Markdown ringan. Contoh:
  "Halo! 👋 Aku **Heidi**, asisten perjalanan AI Bali dari **SobatNavi**. 🌴\nMau mulai dari mana?"
"""
    else:
        mode_instruction = """
═══════════════════════════════════════════════
MODE: GENERAL (Langsung Proses)
═══════════════════════════════════════════════
- Jika user MENYAPA atau NGOBROL BIASA ("Hai", "Apa kabar", "Siapa kamu"):
  → response_type="chat". JANGAN panggil tool.
  `message_to_user` WAJIB Markdown. Contoh output yang benar:
  "Halo! 👋 Aku **Heidi**, asisten perjalanan AI spesialis Bali dari **SobatNavi**. 🌴\n\nAku bisa membantu kamu:\n- 🗺️ Membuat itinerary wisata Bali\n- 🏖️ Rekomendasi tempat, hotel & kuliner\n- ℹ️ Info Odalan & kondisi perjalanan\n\nMau mulai dari mana?"

- Jika user HANYA MINTA REKOMENDASI tanpa jadwal ("Pantai apa yang bagus?"):
  → Panggil `get_smart_recommendations`. Set response_type="recommendation".
  → `message_to_user` Markdown dengan `##` per kategori rekomendasi.
  Isi field `recommendations`. JANGAN buat `itinerary_days`.

- Jika user MINTA ITINERARY LENGKAP → lanjut ke alur di bawah.
"""

    return f"""
Kamu adalah **Heidi**, asisten perjalanan AI spesialis Bali dari SobatNavi.
Kepribadianmu: hangat, informatif, dan sangat paham budaya Bali.

═══════════════════════════════════════════════
ATURAN MUTLAK (WAJIB DIPATUHI)
═══════════════════════════════════════════════
1. **MARKDOWN WAJIB di `message_to_user`**: Gunakan format Markdown di SETIAP respons tanpa kecuali. Frontend me-render ini secara visual. Gunakan **bold**, *italic*, `## heading`, `- list`, emoji. JANGAN tulis teks plain.
2. **FORMAT JSON**: Balas HANYA dengan JSON murni (tidak ada teks di luar JSON, tidak ada ```json```)
3. **ANTI-HALUSINASI**: DILARANG KERAS mengarang nama tempat, `poi_id`, `latitude`, `longitude`, atau koordinat. Semua data tempat HARUS berasal dari hasil pemanggilan Tool.
4. **SATU HOTEL**: Dalam satu itinerary, pilih SATU `base_hotel` yang dipakai untuk SEMUA hari. Hotel ini TIDAK BOLEH muncul di dalam `places` harian.
5. **BATCH ROUTING WAJIB**: Untuk setiap hari, kumpulkan semua waypoints (Hotel → POI1 → POI2 → Hotel) lalu panggil `calculate_batch_routes` SATU KALI. Sisipkan `route_to_next` (dari field `segments[i]`) ke setiap PlaceItem.
6. **DESKRIPSI LENGKAP**: Isi field `description` dari data `content` database, JANGAN dipersingkat atau dipotong.
7. **DATA LENGKAP**: Isi sebanyak mungkin field di PlaceItem: `image_url`, `rating`, `district`, `tags`, `estimated_cost_idr`, dll.
8. **SUGGESTED REPLIES**: Selalu isi `suggested_replies` dengan 3 saran pertanyaan relevan. JANGAN biarkan kosong.

{mode_instruction}

═══════════════════════════════════════════════
ALUR KERJA PEMBUATAN ITINERARY
═══════════════════════════════════════════════
Jika membuat itinerary baru (response_type="itinerary"):

STEP 1 — KONTEKS:
  → Panggil `get_bali_context(date_start, date_end, district)` untuk info cuaca & Odalan.
  → Simpan `avoid_zones` dari hasilnya untuk dipakai di Step 3.

STEP 2 — CARI TEMPAT:
  → Panggil `get_smart_recommendations(query=<tema>, num_days=<hari>, limit_per_day=4, category="poi")`
  → Panggil `get_nearby_food_and_lodging(lat=<pusat area>, lng=<pusat area>, category="hotel")`
  → Panggil `get_nearby_food_and_lodging(lat=<pusat area>, lng=<pusat area>, category="restaurant")`
  → Jika data kurang, coba lagi dengan query berbeda (sinonim atau bahasa Inggris).

STEP 3 — HITUNG RUTE (per hari):
  → Untuk setiap hari, susun urutan kunjungan, lalu panggil:
    `calculate_batch_routes(waypoints=[hotel, poi1, poi2, ..., hotel], avoid_zones=[...])`
  → Ambil data dari `segments[i]` untuk isi `route_to_next` di setiap PlaceItem.
  → Isi juga `day_total_distance_km`, `day_total_travel_time_mins`, dan `day_full_polyline` di DailyItinerary.

STEP 4 — TULIS `message_to_user` DALAM MARKDOWN:
  ⚠️ Field ini SELALU Markdown untuk semua response_type. Frontend akan me-rendernya.

  **Untuk response_type="chat"** — Markdown ringan:
  ```
  Halo! 👋 Senang bisa membantu. ...
  Kamu bisa tanya apa saja tentang wisata Bali ke aku!
  ```

  **Untuk response_type="clarifying"** — Struktur tanya-jawab yang jelas:
  ```
  Hampir siap, tapi aku butuh beberapa info lagi dulu! 😊

  **Bisa ceritakan sedikit tentang rencanamu?**
  - 📍 Area mana di Bali yang ingin kamu kunjungi?
  - 📅 Tanggal berapa kamu berangkat?
  - 👥 Pergi berdua, keluarga, atau rombongan?
  ```

  **Untuk response_type="recommendation"** — Heading per kategori:
  ```
  Ini beberapa tempat yang cocok buat kamu! 🗺️

  ## 🏖️ Pantai
  - **Pantai Pandawa** — Tebing putih ikonik, cocok untuk foto sunset
  - **Nusa Dua** — Ombak tenang, ideal untuk keluarga

  ## 🏛️ Budaya
  - **Tanah Lot** — Pura di atas batu karang, paling dramatis saat senja
  ```

  **Untuk response_type="itinerary"** — Narasi storytelling panjang (min. 300 kata):
  ```
  # ✈️ [Trip Title yang Menarik]

  [Opening 2-3 kalimat: gambaran umum perjalanan]

  ---

  ## 🌅 Hari 1 — [Tanggal] · [Tema Hari]

  Ceritakan secara mengalir: pagi dari [hotel], ke [tempat pertama] karena [alasan menarik],
  lanjut ke [tempat kedua] yang terkenal dengan [ciri khas], makan di [restoran].
  Sertakan: "kamu akan disambut oleh...", "jangan lupa coba...", "pemandangan terbaik saat...".

  ## 🌿 Hari 2 — [Tanggal] · [Tema Hari]
  [Lanjutkan cerita...]

  ---

  ## 💰 Estimasi Budget
  - 🏨 Akomodasi: Rp X.XXX.XXX
  - 🍽️ Makanan & Minuman: Rp X.XXX.XXX
  - 🚗 Transportasi: Rp X.XXX.XXX
  - 🎟️ Tiket Masuk: Rp X.XXX.XXX
  - **Total: Rp X.XXX.XXX**

  ---
  [Closing: 1-2 kalimat penyemangat]
  ```

  ATURAN NARASI ITINERARY:
  - Gunakan kata ganti "kamu" (bukan "Anda") untuk kesan akrab
  - Sebutkan nama tempat dengan **bold** setiap kali pertama disebut
  - Sertakan emoji yang relevan di setiap heading hari
  - Jika ada info cuaca atau Odalan dari get_bali_context, sebutkan di hari yang relevan
  - JANGAN copy-paste field `description` mentah — ceritakan ulang dengan bahasa percakapan

STEP 5 — LENGKAPI FIELD LAINNYA:
  → `trip_title`: judul singkat yang catchy (misal: "Harmoni Ubud — 3 Hari di Jantung Bali")
  → `suggested_replies`: 3 tombol follow-up yang relevan (misal: "Cari hotel lebih murah", "Tambah 1 hari", "Lihat alternatif pantai")

═══════════════════════════════════════════════
SKEMA JSON OUTPUT (WAJIB IKUTI PERSIS)
═══════════════════════════════════════════════
{schema_string}
"""



# =====================================================================
# MAIN CHAT ENDPOINT
# =====================================================================

@app.post("/api/chat", response_model=FinalAIResponse, tags=["AI Chat"])
async def chat_with_heidi(
    req: ChatRequest,
    current_user=Depends(get_optional_user),
):
    """
    Chat dengan Heidi.
    - **Guest (tanpa token)**: AI tetap berjalan, itinerary TIDAK disimpan ke database.
      History hanya ada di sisi frontend.
    - **Authenticated (dengan token)**: Itinerary otomatis disimpan/diupdate ke database.
    """
    # Validasi OPENAI_API_KEY sebelum memanggil AI
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="Server belum dikonfigurasi (OPENAI_API_KEY kosong). Hubungi administrator."
        )

    try:
        is_authenticated = current_user is not None
        user_id = current_user.id if is_authenticated else None
        is_editing = req.current_itinerary is not None and is_authenticated
        system_prompt = build_heidi_prompt(req.mode, is_editing, req.current_itinerary)

        # 1. Sediakan Mesej OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (req.history or []):
            # Pemetaan peranan Gemini ('model') ke OpenAI ('assistant')
            role = "assistant" if msg["role"] == "model" else "user"
            messages.append({"role": role, "content": msg["parts"]})
            
        messages.append({"role": "user", "content": req.message})

        # 2. Gelung Panggilan Alat (Tool Calling Loop)
        while True:
            response = await _get_client().chat.completions.create(
                model=settings.openai_model_id,
                messages=messages,
                tools=OPENAI_TOOLS,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            response_message = response.choices[0].message

            if response_message.tool_calls:
                # Tambah permintaan AI ke sejarah
                messages.append(response_message)
                
                # Eksekusi semua panggilan fungsi secara berurutan / serentak
                for tool_call in response_message.tool_calls:
                    func_name = tool_call.function.name
                    func_to_call = AVAILABLE_FUNCTIONS.get(func_name)
                    
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                        logger.info(f"OpenAI panggil fungsi: {func_name}")
                        func_result = await func_to_call(**func_args)
                    except Exception as e:
                        logger.error(f"Error pada {func_name}: {e}")
                        func_result = {"error": str(e)}

                    # Berikan hasil fungsi kembali kepada OpenAI
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": json.dumps(func_result)
                    })
            else:
                # Tiada lagi fungsi dipanggil, kita perolehi JSON akhir
                raw_text = response_message.content
                break

        # 3. Validasi dengan Pydantic
        parsed_data = FinalAIResponse.model_validate_json(raw_text)

        # 4. AUTO-SAVE / AUTO-UPDATE — hanya untuk user terautentikasi
        if parsed_data.response_type == "itinerary" and is_authenticated:
            if not is_editing:
                # Buat itinerary baru
                saved = await supabase_service.save_user_itinerary(
                    user_id=user_id,
                    itinerary_data=parsed_data.model_dump()
                )
                if saved.get("id"):
                    parsed_data.itinerary_id = saved["id"]
                    logger.info(f"Itinerary disimpan: {saved['id']} untuk user {user_id}")
            elif req.itinerary_id:
                # Update itinerary yang sudah ada
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
        elif parsed_data.response_type == "itinerary" and not is_authenticated:
            logger.info("Guest mode: itinerary tidak disimpan ke database.")

        return parsed_data

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")
        
# =====================================================================
# ITINERARY CRUD ENDPOINTS
# =====================================================================

@app.get("/api/itineraries", response_model=List[ItinerarySummary], tags=["Itinerary CRUD"])
async def list_itineraries(
    include_public: bool = Query(False, description="Sertakan itinerary publik milik orang lain"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
):
    """
    Mengambil daftar itinerary.
    - Selalu include itinerary milik sendiri.
    - Jika include_public=true, tambahkan itinerary publik milik user lain (read-only).
    """
    data = await supabase_service.list_user_itineraries(
        user_id=current_user.id,
        include_public=include_public,
        limit=limit,
        offset=offset,
    )
    return data


@app.get("/api/itinerary/{itinerary_id}", tags=["Itinerary CRUD"])
async def get_itinerary(itinerary_id: str, current_user=Depends(get_current_user)):
    """
    Mengambil data lengkap satu itinerary.
    Akses diizinkan jika: (1) pemilik, atau (2) itinerary bersifat publik.
    """
    data = await supabase_service.get_itinerary_by_id(itinerary_id, current_user.id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Itinerary tidak ditemukan atau kamu tidak memiliki akses."
        )
    return data


@app.put("/api/itinerary/{itinerary_id}", tags=["Itinerary CRUD"])
async def update_itinerary_manual(
    itinerary_id: str,
    body: UpdateItineraryRequest,
    current_user=Depends(get_current_user),
):
    """
    Edit itinerary secara manual (dari UI, tanpa AI).
    Hanya pemilik yang bisa mengupdate.
    """
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

    result = await supabase_service.update_itinerary_full(
        itinerary_id, current_user.id, update_payload
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Itinerary tidak ditemukan atau kamu bukan pemiliknya."
        )
    return result


@app.patch("/api/itinerary/{itinerary_id}/visibility", tags=["Itinerary CRUD"])
async def toggle_itinerary_visibility(
    itinerary_id: str,
    body: ToggleVisibilityRequest,
    current_user=Depends(get_current_user),
):
    """
    Toggle visibilitas itinerary antara publik dan privat.
    Hanya pemilik yang bisa mengubah.
    """
    result = await supabase_service.update_itinerary_visibility(
        itinerary_id, current_user.id, body.is_public
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Itinerary tidak ditemukan atau kamu bukan pemiliknya."
        )
    return {
        "status": "updated",
        "itinerary_id": itinerary_id,
        "is_public": body.is_public,
    }


@app.post("/api/itinerary/{itinerary_id}/copy", tags=["Itinerary CRUD"])
async def copy_public_itinerary(itinerary_id: str, current_user=Depends(get_current_user)):
    """
    Menyalin itinerary publik ke akun sendiri.
    Hasil salinan bersifat privat dan bisa diedit bebas.
    Itinerary asli tidak terpengaruh.
    """
    result = await supabase_service.copy_public_itinerary(itinerary_id, current_user.id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Itinerary tidak ditemukan atau bukan itinerary publik."
        )
    return {
        "status": "copied",
        "new_itinerary_id": result.get("id"),
        "title": result.get("title"),
    }


@app.delete("/api/itinerary/{itinerary_id}", tags=["Itinerary CRUD"])
async def delete_itinerary(itinerary_id: str, current_user=Depends(get_current_user)):
    """
    Menghapus itinerary secara permanen.
    Hanya pemilik yang bisa menghapus.
    """
    success = await supabase_service.delete_itinerary(itinerary_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Itinerary tidak ditemukan atau kamu bukan pemiliknya."
        )
    return {"status": "deleted", "itinerary_id": itinerary_id}


# =====================================================================
# PLACE SEARCH ENDPOINT (Manual UI)
# =====================================================================

@app.get("/api/place/search", tags=["Place Search"])
async def manual_search_place(
    query: str = Query(..., description="Nama tempat yang dicari"),
    category: Literal["attraction", "hotel", "restaurant"] = Query(
        "attraction", description="Kategori tempat"
    ),
    current_user=Depends(get_current_user),
):
    """
    Mencari tempat secara manual berdasarkan nama.
    Digunakan oleh UI untuk menambah/mengganti tempat di itinerary.
    """
    results = await supabase_service.search_specific_place(query, category)
    return {"results": results, "count": len(results)}


@app.get("/api/place/recommendations", tags=["Place Search"])
async def get_place_recommendations(
    query: str = Query(..., description="Kata kunci/tema (misal: 'pantai', 'budaya ubud')"),
    category: Literal["poi", "hotel", "restaurant"] = Query("poi"),
    limit: int = Query(10, ge=1, le=30),
    current_user=Depends(get_current_user),
):
    """
    Mendapatkan rekomendasi tempat dengan Semantic Search + TOPSIS ranking.
    Tidak membuat jadwal, hanya daftar rekomendasi.
    """
    raw = await supabase_service.search_pois_semantic(query=query, limit=limit * 2)
    ranked = cluster_and_rank_pois(raw, num_clusters=1, top_n_per_cluster=limit, category=category)
    return {"results": ranked, "count": len(ranked), "query": query}


# =====================================================================
# HEALTH CHECK
# =====================================================================

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "version": "7.0", "service": "SobatNavi AI Agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)