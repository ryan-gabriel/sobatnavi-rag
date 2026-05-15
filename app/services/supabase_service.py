# app/services/supabase_service.py
import asyncio
import json
import logging
from typing import Optional
from supabase import create_client, Client
from app.core.config import settings
from app.core.resilience import retry_with_backoff
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Lazy OpenAI client — hanya dibuat saat pertama kali dipakai
# Agar server bisa start meski OPENAI_API_KEY belum diset
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY belum dikonfigurasi. "
                "Isi OPENAI_API_KEY di file .env terlebih dahulu."
            )
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client

# Kolom yang dikembalikan untuk POI
POI_SELECT_COLS = (
    "place_id, name, district, content, latitude, longitude, "
    "rating, user_rating_count, image_url, metadata"
)

# Kolom untuk hotel/restoran
AMENITY_SELECT_COLS = (
    "id, place_id, name, district, latitude, longitude, "
    "rating, user_rating_count, image_url, metadata"
)

# Kolom user_itineraries (sesuai schema Supabase aktual — tanpa cover_image_url & district_tags)
ITINERARY_LIST_COLS = (
    "id, user_id, title, description, days_count, total_budget_idr, "
    "is_public, share_slug, created_at, updated_at"
)


class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(settings.supabase_url, settings.supabase_service_key)

    # =====================================================================
    # ODALAN
    # =====================================================================

    @retry_with_backoff(retries=3)
    async def get_all_active_odalans(self, date_start: str, date_end: str) -> list[dict]:
        """Mengambil semua upacara Odalan aktif dalam rentang tanggal."""
        result = await asyncio.to_thread(
            self.client.table("odalan_events").select(
                "id, poi_attraction_id, location_name, start_time, end_time, "
                "north_east_latitude, north_east_longitude, "
                "south_west_latitude, south_west_longitude"
            )
            .gte("end_time", date_start)
            .lte("start_time", date_end)
            .execute
        )
        return result.data or []

    # =====================================================================
    # POI SEARCH
    # =====================================================================

    @retry_with_backoff(retries=3)
    async def search_pois_semantic(
        self,
        query: str = "",
        limit: int = 30,
        filter_district: str = None,
        filter_category: str = None,
        filter_price_level: int = None,
        filter_min_rating: float = None,
    ) -> list[dict]:
        """
        Pencarian POI berdasarkan MAKNA menggunakan OpenAI Embedding + pgvector.
        Semua parameter filter bersifat opsional.
        """
        if not query:
            result = await asyncio.to_thread(
                self.client.table("poi_attractions")
                .select(POI_SELECT_COLS)
                .limit(limit)
                .execute
            )
            return result.data or []

        try:
            # 1. Buat embedding dari query
            oai = _get_openai_client()
            response = await oai.embeddings.create(
                input=query,
                model=settings.openai_embedding_model
            )
            query_embedding = response.data[0].embedding

            # 2. Panggil RPC match_poi_attractions dengan semua parameter yang dibutuhkan
            # (semua filter opsional, kirim None jika tidak diset)
            rpc_params = {
                "query_embedding": query_embedding,
                "match_count": limit,
                "filter_district": filter_district,       # NULL = tidak filter
                "filter_category": filter_category,       # NULL = tidak filter
                "filter_price_level": filter_price_level, # NULL = tidak filter
                "filter_min_rating": filter_min_rating,   # NULL = tidak filter
            }
            result = await asyncio.to_thread(
                self.client.rpc("match_poi_attractions", rpc_params).execute
            )
            data = result.data or []
            logger.info(f"Semantic search '{query}': {len(data)} hasil")
            return data

        except Exception as e:
            logger.warning(f"Vector search gagal ({e}), fallback ke keyword search.")
            return await self._keyword_search_pois(query, limit)

    async def search_pois(self, search: str = "", limit: int = 30) -> list[dict]:
        """Alias untuk backward compatibility."""
        return await self.search_pois_semantic(query=search, limit=limit)

    async def _keyword_search_pois(self, query: str, limit: int = 30) -> list[dict]:
        """Keyword search fallback."""
        safe_query = query.replace('"', "").strip()
        search_filter = (
            f'name.ilike."%{safe_query}%",'
            f'content.ilike."%{safe_query}%",'
            f'district.ilike."%{safe_query}%"'
        )
        result = await asyncio.to_thread(
            self.client.table("poi_attractions")
            .select(POI_SELECT_COLS)
            .or_(search_filter)
            .limit(limit)
            .execute
        )
        return result.data or []

    # =====================================================================
    # AMENITY SEARCH (Hotel & Restoran)
    # =====================================================================

    @retry_with_backoff(retries=3)
    async def search_amenities_nearby(
        self,
        amenity_type: str,
        lat: float,
        lng: float,
        radius_m: float = 5000,
        limit: int = 5,
        max_price: int = None,
    ) -> list[dict]:
        """Mencari hotel atau restoran terdekat menggunakan PostGIS RPC."""
        rpc_func = "search_hotels_nearby" if amenity_type == "hotel" else "search_restaurants_nearby"
        result = await asyncio.to_thread(
            self.client.rpc(rpc_func, {
                "lat": lat,
                "lng": lng,
                "radius_m": radius_m,
                "max_price": max_price,   # NULL = tidak filter harga
                "result_limit": limit
            }).execute
        )
        return result.data or []

    async def search_specific_place(self, query: str, category: str = "attraction") -> list[dict]:
        """Mencari tempat spesifik berdasarkan nama."""
        table_map = {
            "attraction": "poi_attractions",
            "hotel": "hotel_amenities",
            "restaurant": "culinary_amenities"
        }
        table_name = table_map.get(category, "poi_attractions")
        select_cols = POI_SELECT_COLS if category == "attraction" else AMENITY_SELECT_COLS

        def _fetch():
            return (
                self.client.table(table_name)
                .select(select_cols)
                .ilike("name", f"%{query}%")
                .limit(5)
                .execute()
            )

        result = await asyncio.to_thread(_fetch)
        return result.data or []

    # =====================================================================
    # ITINERARY CRUD
    # =====================================================================

    async def save_user_itinerary(self, user_id: str, itinerary_data: dict) -> dict:
        """
        Menyimpan itinerary baru ke database (INSERT).
        Hanya menggunakan kolom yang ada di schema aktual.
        """
        try:
            days = itinerary_data.get("itinerary_days", [])
            trip_title = itinerary_data.get("trip_title")
            if not trip_title:
                first_theme = days[0].get("theme", "Trip Bali") if days else "Trip Bali"
                trip_title = f"{first_theme} ({len(days)} hari)"

            data = {
                "user_id": user_id,
                "title": trip_title,
                "itinerary_data": itinerary_data,
                "total_budget_idr": itinerary_data.get("total_budget_idr", 0),
                "days_count": len(days),
                "is_public": False,
            }

            result = await asyncio.to_thread(
                self.client.table("user_itineraries").insert(data).execute
            )
            return result.data[0] if result.data else {}

        except Exception as e:
            logger.error(f"Gagal simpan itinerary: {e}")
            return {}

    async def get_itinerary_by_id(self, itinerary_id: str, user_id: str) -> Optional[dict]:
        """
        Mengambil satu itinerary.
        Akses diizinkan jika: (1) pemilik, atau (2) itinerary bersifat publik.
        """
        try:
            result = await asyncio.to_thread(
                self.client.table("user_itineraries")
                .select("*")
                .eq("id", itinerary_id)
                .maybe_single()
                .execute
            )
            row = result.data
            if not row:
                return None

            is_owner = row.get("user_id") == user_id
            is_public = row.get("is_public", False)

            if not is_owner and not is_public:
                return None

            row["is_owner"] = is_owner
            return row

        except Exception as e:
            logger.error(f"get_itinerary_by_id error: {e}")
            return None

    async def list_user_itineraries(
        self,
        user_id: str,
        include_public: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """
        Mengambil daftar itinerary.
        Setiap item di-annotate dengan `is_owner: bool`.
        """
        try:
            own_result = await asyncio.to_thread(
                self.client.table("user_itineraries")
                .select(ITINERARY_LIST_COLS)
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute
            )
            own_data = own_result.data or []
            for row in own_data:
                row["is_owner"] = True

            if not include_public:
                return own_data

            public_result = await asyncio.to_thread(
                self.client.table("user_itineraries")
                .select(ITINERARY_LIST_COLS)
                .eq("is_public", True)
                .neq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute
            )
            public_data = public_result.data or []
            for row in public_data:
                row["is_owner"] = False

            return own_data + public_data

        except Exception as e:
            logger.error(f"list_user_itineraries error: {e}")
            return []

    async def update_itinerary_full(
        self, itinerary_id: str, user_id: str, update_data: dict
    ) -> Optional[dict]:
        """Update itinerary. Hanya bisa dilakukan oleh pemilik."""
        try:
            result = await asyncio.to_thread(
                self.client.table("user_itineraries")
                .update(update_data)
                .eq("id", itinerary_id)
                .eq("user_id", user_id)  # Security check
                .execute
            )
            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"update_itinerary_full error: {e}")
            return None

    async def update_itinerary_visibility(
        self, itinerary_id: str, user_id: str, is_public: bool
    ) -> Optional[dict]:
        """Toggle visibility itinerary."""
        return await self.update_itinerary_full(
            itinerary_id, user_id, {"is_public": is_public}
        )

    async def copy_public_itinerary(
        self, itinerary_id: str, requester_user_id: str
    ) -> Optional[dict]:
        """
        Menyalin itinerary publik ke akun user lain.
        User yang menyalin menjadi pemilik baru salinan.
        """
        try:
            result = await asyncio.to_thread(
                self.client.table("user_itineraries")
                .select("*")
                .eq("id", itinerary_id)
                .eq("is_public", True)
                .maybe_single()
                .execute
            )
            original = result.data
            if not original:
                return None

            new_data = {
                "user_id": requester_user_id,
                "title": f"Salinan: {original.get('title', 'Trip Bali')}",
                "itinerary_data": original.get("itinerary_data", {}),
                "total_budget_idr": original.get("total_budget_idr", 0),
                "days_count": original.get("days_count", 0),
                "is_public": False,  # Salinan selalu privat
            }
            copy_result = await asyncio.to_thread(
                self.client.table("user_itineraries").insert(new_data).execute
            )
            return copy_result.data[0] if copy_result.data else None

        except Exception as e:
            logger.error(f"copy_public_itinerary error: {e}")
            return None

    async def delete_itinerary(self, itinerary_id: str, user_id: str) -> bool:
        """Menghapus itinerary. Hanya pemilik yang bisa menghapus."""
        try:
            result = await asyncio.to_thread(
                self.client.table("user_itineraries")
                .delete()
                .eq("id", itinerary_id)
                .eq("user_id", user_id)
                .execute
            )
            return bool(result.data)
        except Exception as e:
            logger.error(f"delete_itinerary error: {e}")
            return False


supabase_service = SupabaseService()