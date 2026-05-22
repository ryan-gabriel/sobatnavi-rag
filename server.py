# server.py
import asyncio
from typing import List, Optional
from fastmcp import FastMCP
from pydantic import Field

from app.services.supabase_service import supabase_service
from app.services.tomtom_service import tomtom_service
from app.services.live_intel_service import live_intel_service
from app.engine.odalan_checker import evaluate_odalan_status, extract_global_avoid_zones
from app.engine.recommender import generate_clustered_pool_delivery, rank_pois_by_topsis

mcp = FastMCP("SobatNavi", version="5.0.0")

# MCP TOOLS

@mcp.tool()
async def get_bali_context(
    date_start: str = Field(..., description="Tanggal mulai perjalanan (YYYY-MM-DD)"),
    date_end: str = Field(..., description="Tanggal selesai perjalanan (YYYY-MM-DD)"),
    district: str = Field("Bali", description="Kabupaten spesifik (misal: Ubud, Kuta, Seminyak)")
) -> dict:
    """
    Mengambil info cuaca real-time dan daftar upacara Odalan/zona hindari kemacetan
    untuk tanggal perjalanan yang diminta.
    """
    weather_query = f"cuaca dan acara lokal di {district}, Bali pada tanggal {date_start}"
    weather_and_events = await live_intel_service.search_tavily(weather_query)
    active_odalans = await supabase_service.get_all_active_odalans(date_start, date_end)
    avoid_zones = extract_global_avoid_zones(active_odalans)

    return {
        "live_intel": weather_and_events,
        "active_odalans_count": len(active_odalans),
        "avoid_zones": avoid_zones,
        "avoid_zones_note": "Gunakan avoid_zones ini sebagai parameter di calculate_batch_routes"
    }


@mcp.tool()
async def get_smart_recommendations(
    query: str = Field(..., description="Kata kunci tema wisata"),
    num_days: int = Field(1, description="Jumlah hari perjalanan"),
    category: str = Field("poi", description="'poi', 'hotel', atau 'restaurant'"),
    preference_mode: str = Field("standard", description="'standard', 'hidden_gem', 'luxury', 'budget'"),
    user_detected_location: str = Field(None, description="District/area Bali yang disebutkan user (opsional)")
) -> list[dict]:
    """
    Cari tempat wisata dengan Radius & Anchor-First clustering + TOPSIS ranking.
    Untuk category='poi': mengembalikan pool POI + restoran per hari yang geographically tight.
    Output berisi per-hari: {day, anchor, pois[], restaurants[]}.
    """
    if category == "hotel":
        raw = await supabase_service.search_amenities_semantic(query, "hotel", limit=40)
        return rank_pois_by_topsis(raw, category="hotel", preference_mode=preference_mode, top_n=10)
    elif category == "restaurant":
        raw = await supabase_service.search_amenities_semantic(query, "restaurant", limit=40)
        return rank_pois_by_topsis(raw, category="restaurant", preference_mode=preference_mode, top_n=10)
    else:
        return await generate_clustered_pool_delivery(
            supabase_service=supabase_service,
            query=query,
            num_days=num_days,
            user_detected_location=user_detected_location,
            preference_mode=preference_mode,
        )


@mcp.tool()
async def search_specific_place(
    query: str = Field(..., description="Nama tempat spesifik yang dicari"),
    category: str = Field("attraction", description="'attraction', 'hotel', atau 'restaurant'")
) -> list[dict]:
    """
    Mencari tempat SPESIFIK berdasarkan nama dari database.
    Gunakan ini saat user menyebut nama tempat tertentu untuk ditambah/diganti di itinerary.
    """
    return await supabase_service.search_specific_place(query, category)


@mcp.tool()
async def search_amenities_nearby(
    amenity_type: str = Field(..., description="'hotel' atau 'restaurant'"),
    lat: float = Field(..., description="Latitude pusat pencarian"),
    lng: float = Field(..., description="Longitude pusat pencarian"),
    radius_m: float = Field(5000, description="Radius pencarian dalam meter (default 5km)"),
    limit: int = Field(5, description="Jumlah hasil maksimal")
) -> list[dict]:
    """
    Mencari hotel atau restoran terdekat berdasarkan koordinat menggunakan PostGIS.
    """
    return await supabase_service.search_amenities_nearby(amenity_type, lat, lng, radius_m, limit)


@mcp.tool()
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


@mcp.tool()
async def calculate_batch_routes(
    waypoints: list[dict] = Field(
        ...,
        description=(
            "Daftar waypoints satu hari perjalanan. "
            "Format: [{\"lat\": -8.5, \"lng\": 115.2, \"name\": \"Tanah Lot\"}, ...]. "
            "Urutan: Hotel (start) → POI1 → POI2 → ... → Hotel (end)"
        )
    ),
    avoid_zones: list[str] = Field(
        default_factory=list,
        description="Daftar bbox zona Odalan dari get_bali_context (untuk dihindari saat routing)"
    )
) -> dict:
    """
    Menghitung rute satu hari perjalanan secara efisien (satu panggilan per hari).
    
    Returns:
        total_distance_km, total_travel_time_mins, segments (dengan polyline per segmen),
        dan full_day_polyline (gabungan semua titik untuk render jalur di peta).
    """
    return await tomtom_service.get_full_day_route(waypoints, avoid_zones)


@mcp.tool()
async def calculate_route_with_avoidance(
    origin_lat: float = Field(..., description="Latitude titik asal"),
    origin_lng: float = Field(..., description="Longitude titik asal"),
    dest_lat: float = Field(..., description="Latitude tujuan"),
    dest_lng: float = Field(..., description="Longitude tujuan"),
    avoid_zones: List[str] = Field(default_factory=list, description="Daftar bbox zona hindari")
) -> dict:
    """
    Kalkulasi rute point-to-point dengan menghindari zona Odalan.
    Untuk rute batch per hari, gunakan `calculate_batch_routes` yang lebih efisien.
    """
    origin_str = f"{origin_lat},{origin_lng}"
    dest_str = f"{dest_lat},{dest_lng}"
    try:
        result = await tomtom_service.calculate_route(origin_str, dest_str, avoid_zones)
        return result
    except Exception:
        return await tomtom_service.get_fallback_route(origin_lat, origin_lng, dest_lat, dest_lng)


# MCP PROMPTS

@mcp.prompt("heidi_persona")
def heidi_persona() -> str:
    """Instruksi Sistem untuk Heidi AI — Asisten Perjalanan Bali."""
    return """
    Kamu adalah Heidi, asisten perjalanan AI spesialis Bali dari SobatNavi.
    
    ATURAN MUTLAK:
    1. Balas HANYA dengan format JSON murni (tidak ada markdown ```json)
    2. DILARANG KERAS mengarang nama tempat, koordinat, atau ID — semua data dari Tool
    3. Pilih SATU base_hotel untuk semua hari (tidak boleh ganti hotel per hari)
    4. Panggil calculate_batch_routes SATU KALI PER HARI untuk menghitung rute
    
    ALUR KERJA ITINERARY:
    1. get_bali_context → dapatkan cuaca & avoid_zones
    2. get_smart_recommendations → dapatkan POI (category="poi")
    3. search_amenities_nearby → dapatkan hotel & restoran
    4. validate_itinerary_safety → cek konflik Odalan
    5. calculate_batch_routes (per hari) → dapatkan rute + polyline
    6. Susun respons JSON lengkap sesuai FinalAIResponse schema
    
    TOPSIS WEIGHTS (informasi internal):
    - POI: rating(30%), popularity(20%), price_value(15%), strategic_score(20%), visual_interest(15%)
    - Hotel: rating(35%), comfort(30%), amenity_density(20%), accessibility(15%)
    - Restaurant: rating(35%), menu_variety(25%), ambience(25%), payment_modern(15%)
    """