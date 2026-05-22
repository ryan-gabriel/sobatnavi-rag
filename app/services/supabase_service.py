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
    "place_id, name, district, latitude, longitude, "
    "rating, user_rating_count, image_url, price_level, metadata"
)

# Kolom user_itineraries (sesuai schema Supabase aktual)
ITINERARY_LIST_COLS = (
    "id, user_id, title, description, days_count, total_budget_idr, "
    "is_public, share_slug, created_at, updated_at"
)


class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(settings.supabase_url, settings.supabase_service_key)
        self._cached_db_districts = None
        self._normalization_cache = {}

    # =========================================================================
    # ODALAN
    # =========================================================================

    @retry_with_backoff(retries=3)
    async def get_all_active_odalans(self, date_start: str, date_end: str) -> list[dict]:
        """Mengambil semua upacara Odalan aktif dalam rentang tanggal."""
        def _fetch():
            return (
                self.client.table("odalan_events")
                .select(
                    "id, poi_attraction_id, location_name, start_time, end_time, "
                    "north_east_latitude, north_east_longitude, "
                    "south_west_latitude, south_west_longitude"
                )
                .lt("start_time", date_end)
                .gt("end_time", date_start)
                .execute()
            )
        result = await asyncio.to_thread(_fetch)
        return result.data or []

    # =========================================================================
    # POI SEARCH
    # =========================================================================

    async def get_db_districts(self) -> list[str]:
        """Mendapatkan daftar unik district dari database secara dinamis dengan caching."""
        if self._cached_db_districts is None:
            try:
                def _fetch_districts():
                    return self.client.table("poi_attractions").select("district").execute()
                res = await asyncio.to_thread(_fetch_districts)
                self._cached_db_districts = sorted(list(set(
                    r["district"] for r in res.data if r.get("district")
                )))
            except Exception as e:
                logger.warning(f"Gagal memuat daftar district dari database secara dinamis: {e}.")
                self._cached_db_districts = []
        return self._cached_db_districts

    async def _normalize_district_names(self, location: str) -> list[str]:
        """
        Memetakan nama sub-distrik/daerah wisata di Bali ke nama-nama distrik
        yang ada di database secara dinamis menggunakan AI (gpt-4.1-nano)
        tanpa hardcoded string mapping dan dengan caching.
        """
        if not location:
            return []

        # 1. Cek cache normalisasi
        loc_clean = location.strip().lower()
        if loc_clean in self._normalization_cache:
            logger.info(f"Normalization cache hit untuk '{location}': {self._normalization_cache[loc_clean]}")
            return self._normalization_cache[loc_clean]

        db_districts = await self.get_db_districts()
        if not db_districts:
            return [location]

        # 2. Gunakan gpt-4.1-nano untuk memetakan input 'location' ke db_districts secara cerdas & bertoleransi typo
        try:
            oai = _get_openai_client()
            prompt = (
                "You are an expert travel assistant for Bali.\n"
                "Given a user-provided location name (which might have typos, spelling mistakes, or refer to a sub-district, village, beach, landmark, or regency),\n"
                "map it to all relevant database district/location names from this allowed list:\n"
                f"{db_districts}\n\n"
                "Rules:\n"
                "1. Connect the input to any regencies it belongs to, and any matching beaches, landmarks, or streets in the allowed list.\n"
                "2. Gracefully handle typos or phonetic spelling variations.\n"
                "3. Return a JSON object with exactly one key: 'normalized' containing a list of the matching database names.\n"
                "Return ONLY the raw JSON object. No markdown, no backticks, no explanations."
            )
            
            response = await oai.chat.completions.create(
                model=settings.openai_model_id,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Location: {location}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=150
            )
            
            data = json.loads(response.choices[0].message.content)
            normalized = data.get("normalized", [])
            valid_normalized = [d for d in normalized if d in db_districts]
            if valid_normalized:
                self._normalization_cache[loc_clean] = valid_normalized
                return valid_normalized

        except Exception as e:
            logger.warning(f"Normalisasi district menggunakan AI gagal ({e}). Menggunakan fallback substring matching.")

        # Fallback substring matching
        matches = [d for d in db_districts if loc_clean in d.lower()]
        if matches:
            return matches

        return [location]

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
            def _fetch():
                return (
                    self.client.table("poi_attractions")
                    .select(POI_SELECT_COLS)
                    .limit(limit)
                    .execute()
                )
            result = await asyncio.to_thread(_fetch)
            return result.data or []

        try:
            oai = _get_openai_client()
            response = await oai.embeddings.create(
                input=query,
                model=settings.openai_embedding_model,
                dimensions=768
            )
            query_embedding = response.data[0].embedding

            rpc_params = {
                "query_embedding": query_embedding,
                "match_count": max(limit * 4, 100) if filter_district else limit,
                "filter_district": None,  # Matikan filter district di RPC agar bisa difilter di Python
                "filter_category": filter_category,
                "filter_price_level": filter_price_level,
                "filter_min_rating": filter_min_rating,
            }

            def _fetch():
                return self.client.rpc("match_poi_attractions", rpc_params).execute()

            result = await asyncio.to_thread(_fetch)
            data = result.data or []
            
            if filter_district:
                normalized_districts = await self._normalize_district_names(filter_district)
                filtered_data = [
                    poi for poi in data
                    if poi.get("district") in normalized_districts
                ]
                logger.info(
                    f"District filter '{filter_district}' mapped to {normalized_districts}. "
                    f"Filtered {len(data)} results down to {len(filtered_data)}."
                )
                if len(filtered_data) >= 3:
                    data = filtered_data[:limit]
                else:
                    logger.warning(
                        f"Python district filtering for '{filter_district}' yielded only {len(filtered_data)} results. "
                        f"Falling back to unfiltered global semantic results to prevent empty pool."
                    )
                    data = data[:limit]
            else:
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

        def _fetch():
            return (
                self.client.table("poi_attractions")
                .select(POI_SELECT_COLS)
                .or_(search_filter)
                .limit(limit)
                .execute()
            )

        result = await asyncio.to_thread(_fetch)
        return result.data or []

    async def search_pois_nearby(
        self, lat: float, lng: float, radius_m: float = 3000, limit: int = 10
    ) -> list[dict]:
        """Mencari POI terdekat menggunakan PostGIS via RPC (fallback ke semantic jika gagal)."""
        try:
            def _fetch():
                return self.client.rpc("search_pois_nearby", {
                    "lat": lat,
                    "lng": lng,
                    "radius_m": radius_m,
                    "result_limit": limit,
                }).execute()

            result = await asyncio.to_thread(_fetch)
            return result.data or []
        except Exception as e:
            logger.warning(f"search_pois_nearby RPC gagal ({e}), fallback semantic.")
            return await self.search_pois_semantic(query="wisata bali", limit=limit)

    # =========================================================================
    # AMENITY SEARCH (Hotel & Restoran)
    # =========================================================================

    @retry_with_backoff(retries=3)
    async def search_amenities_semantic(
        self,
        query: str,
        amenity_type: str,  # "hotel" atau "restaurant"
        limit: int = 40,
        filter_price_level: int = None,
    ) -> list[dict]:
        """
        Pencarian hotel/restoran berdasarkan MAKNA menggunakan OpenAI Embedding + pgvector.
        Memanggil RPC match_hotel_amenities atau match_culinary_amenities.
        """
        rpc_func = "match_hotel_amenities" if amenity_type == "hotel" else "match_culinary_amenities"

        try:
            oai = _get_openai_client()
            response = await oai.embeddings.create(
                input=query,
                model=settings.openai_embedding_model,
                dimensions=768
            )
            query_embedding = response.data[0].embedding

            rpc_params = {
                "query_embedding": query_embedding,
                "match_count": limit,
                "filter_price_level": filter_price_level,
            }

            def _fetch():
                return self.client.rpc(rpc_func, rpc_params).execute()

            result = await asyncio.to_thread(_fetch)
            data = result.data or []
            logger.info(f"Amenity semantic search '{query}' [{amenity_type}]: {len(data)} hasil")
            return data

        except Exception as e:
            logger.warning(f"Amenity vector search gagal ({e}), fallback ke keyword search.")
            table = "hotel_amenities" if amenity_type == "hotel" else "culinary_amenities"
            safe_query = query.replace('"', "").strip()

            def _fetch():
                return (
                    self.client.table(table)
                    .select(AMENITY_SELECT_COLS)
                    .ilike("name", f"%{safe_query}%")
                    .limit(limit)
                    .execute()
                )

            result = await asyncio.to_thread(_fetch)
            return result.data or []

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

        def _fetch():
            return self.client.rpc(rpc_func, {
                "lat": lat,
                "lng": lng,
                "radius_m": radius_m,
                "max_price": max_price,
                "result_limit": limit,
            }).execute()

        result = await asyncio.to_thread(_fetch)
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

    async def search_specific_place_nearby(
        self,
        query: str,
        lat: float,
        lng: float,
        radius_m: float = 15000,
    ) -> list[dict]:
        """
        Mencari tempat spesifik berdasarkan nama DAN membatasi hasil
        hanya dalam radius geografis tertentu dari koordinat referensi.
        Menggunakan RPC search_specific_place_nearby di Supabase.
        Fallback ke search_specific_place tanpa filter geografi jika RPC gagal.
        """
        try:
            def _fetch():
                return self.client.rpc("search_specific_place_nearby", {
                    "search_query": query,
                    "lat": lat,
                    "lng": lng,
                    "radius_m": radius_m,
                }).execute()

            result = await asyncio.to_thread(_fetch)
            data = result.data or []
            logger.info(
                f"search_specific_place_nearby '{query}' @ ({lat:.4f},{lng:.4f}) "
                f"radius={radius_m}m → {len(data)} hasil"
            )
            return data
        except Exception as e:
            logger.warning(
                f"search_specific_place_nearby RPC gagal ({e}), "
                "fallback ke search_specific_place tanpa filter geografi."
            )
            return await self.search_specific_place(query)


    @retry_with_backoff(retries=3)
    async def search_inspiration_narrations(self, query: str, limit: int = 3) -> list[dict]:
        """
        Pencarian narasi inspiratif berdasarkan makna/vibe menggunakan OpenAI Embedding + pgvector.
        Memanggil RPC match_inspiration_narrations.
        """
        try:
            oai = _get_openai_client()
            response = await oai.embeddings.create(
                input=query,
                model=settings.openai_embedding_model,
                dimensions=768
            )
            query_embedding = response.data[0].embedding

            def _fetch():
                return self.client.rpc("match_inspiration_narrations", {
                    "query_embedding": query_embedding,
                    "match_count": limit,
                }).execute()

            result = await asyncio.to_thread(_fetch)
            data = result.data or []
            logger.info(f"Inspiration semantic search '{query}': {len(data)} hasil")
            return data

        except Exception as e:
            logger.warning(f"Inspiration vector search gagal ({e}), fallback ke keyword search.")
            safe_query = query.replace('"', "").strip()

            def _fetch():
                return (
                    self.client.table("inspiration_narrations")
                    .select("place_id, content, metadata")
                    .ilike("content", f"%{safe_query}%")
                    .limit(limit)
                    .execute()
                )

            result = await asyncio.to_thread(_fetch)
            return result.data or []

    # =========================================================================
    # ITINERARY CRUD
    # =========================================================================

    async def get_latest_itinerary_by_session(
        self,
        chat_session_id: str,
        user_id: str,
    ) -> Optional[dict]:
        """
        Mengambil itinerary terakhir untuk chat_session_id tertentu milik user_id.
        Mengembalikan dict itinerary_data, atau None jika tidak ditemukan.
        """
        try:
            def _fetch():
                return (
                    self.client.table("user_itineraries")
                    .select("itinerary_data")
                    .eq("chat_session_id", chat_session_id)
                    .eq("user_id", user_id)
                    .order("created_at", descending=True)
                    .limit(1)
                    .maybe_single()
                    .execute()
                )

            result = await asyncio.to_thread(_fetch)
            row = result.data
            if row and row.get("itinerary_data"):
                return row["itinerary_data"]
            return None
        except Exception as e:
            logger.warning(f"get_latest_itinerary_by_session gagal (session={chat_session_id}): {e}")
            return None

    async def save_user_itinerary(
        self,
        user_id: str,
        itinerary_data: dict,
        chat_session_id: str = None,
    ) -> dict:
        """Menyimpan itinerary baru ke database (INSERT)."""
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
            if chat_session_id:
                data["chat_session_id"] = chat_session_id

            def _fetch():
                return self.client.table("user_itineraries").insert(data).execute()

            result = await asyncio.to_thread(_fetch)
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
            def _fetch():
                return (
                    self.client.table("user_itineraries")
                    .select("*")
                    .eq("id", itinerary_id)
                    .maybe_single()
                    .execute()
                )

            result = await asyncio.to_thread(_fetch)
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
            def _fetch_own():
                return (
                    self.client.table("user_itineraries")
                    .select(ITINERARY_LIST_COLS)
                    .eq("user_id", user_id)
                    .order("created_at", desc=True)
                    .range(offset, offset + limit - 1)
                    .execute()
                )

            own_result = await asyncio.to_thread(_fetch_own)
            own_data = own_result.data or []
            for row in own_data:
                row["is_owner"] = True

            if not include_public:
                return own_data

            def _fetch_public():
                return (
                    self.client.table("user_itineraries")
                    .select(ITINERARY_LIST_COLS)
                    .eq("is_public", True)
                    .neq("user_id", user_id)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )

            public_result = await asyncio.to_thread(_fetch_public)
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
            def _fetch():
                return (
                    self.client.table("user_itineraries")
                    .update(update_data)
                    .eq("id", itinerary_id)
                    .eq("user_id", user_id)
                    .execute()
                )

            result = await asyncio.to_thread(_fetch)
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
            def _fetch_original():
                return (
                    self.client.table("user_itineraries")
                    .select("*")
                    .eq("id", itinerary_id)
                    .eq("is_public", True)
                    .maybe_single()
                    .execute()
                )

            result = await asyncio.to_thread(_fetch_original)
            original = result.data
            if not original:
                return None

            new_data = {
                "user_id": requester_user_id,
                "title": f"Salinan: {original.get('title', 'Trip Bali')}",
                "itinerary_data": original.get("itinerary_data", {}),
                "total_budget_idr": original.get("total_budget_idr", 0),
                "days_count": original.get("days_count", 0),
                "is_public": False,
            }

            def _fetch_insert():
                return self.client.table("user_itineraries").insert(new_data).execute()

            copy_result = await asyncio.to_thread(_fetch_insert)
            return copy_result.data[0] if copy_result.data else None

        except Exception as e:
            logger.error(f"copy_public_itinerary error: {e}")
            return None

    async def delete_itinerary(self, itinerary_id: str, user_id: str) -> bool:
        """Menghapus itinerary. Hanya pemilik yang bisa menghapus."""
        try:
            def _fetch():
                return (
                    self.client.table("user_itineraries")
                    .delete()
                    .eq("id", itinerary_id)
                    .eq("user_id", user_id)
                    .execute()
                )

            result = await asyncio.to_thread(_fetch)
            return bool(result.data)
        except Exception as e:
            logger.error(f"delete_itinerary error: {e}")
            return False

    # =========================================================================
    # CHAT SESSION & HISTORY
    # =========================================================================

    async def get_or_create_chat_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Mendapatkan ID sesi chat yang ada, atau membuat sesi baru jika belum ada."""
        if session_id:
            try:
                def _fetch():
                    return (
                        self.client.table("chat_sessions")
                        .select("id")
                        .eq("id", session_id)
                        .eq("user_id", user_id)
                        .maybe_single()
                        .execute()
                    )

                result = await asyncio.to_thread(_fetch)
                if result.data:
                    return session_id
            except Exception as e:
                logger.warning(f"Gagal verifikasi session {session_id}: {e}")

        new_session = {
            "user_id": user_id,
            "title": "Percakapan Baru",
        }

        def _fetch():
            return self.client.table("chat_sessions").insert(new_session).execute()

        result = await asyncio.to_thread(_fetch)
        return result.data[0]["id"]

    async def get_chat_history(self, session_id: str) -> list[dict]:
        """Mengambil riwayat obrolan masa lalu dari database."""
        def _fetch():
            return (
                self.client.table("chat_messages")
                .select("role, content")
                .eq("session_id", session_id)
                .order("created_at", desc=False)
                .execute()
            )

        result = await asyncio.to_thread(_fetch)
        return result.data or []

    async def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        itinerary_data: dict = None,
    ):
        """Menyimpan satu balon pesan (dari user atau dari AI) ke database."""
        data = {
            "session_id": session_id,
            "role": role,
            "content": content,
        }
        if itinerary_data:
            data["itinerary"] = itinerary_data

        def _fetch():
            return self.client.table("chat_messages").insert(data).execute()

        await asyncio.to_thread(_fetch)


supabase_service = SupabaseService()