# app/schemas/response_schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any


# =====================================================================
# SUB-MODELS
# =====================================================================

class GeoPoint(BaseModel):
    """Titik koordinat tunggal untuk rendering peta."""
    lat: float = Field(description="Latitude")
    lng: float = Field(description="Longitude")


class RouteSegment(BaseModel):
    """
    Data rute dari satu tempat ke tempat berikutnya.
    Berisi polyline koordinat untuk di-render di peta frontend.
    """
    distance_km: Optional[float] = Field(None, description="Jarak dalam km")
    travel_time_mins: Optional[int] = Field(None, description="Waktu tempuh dalam menit")
    traffic_delay_mins: Optional[int] = Field(0, description="Estimasi delay akibat macet (menit)")
    polyline: Optional[List[GeoPoint]] = Field(
        None,
        description="Daftar koordinat {lat, lng} untuk render rute di peta (Leaflet/Google Maps/MapLibre)"
    )
    status: Optional[str] = Field("OK", description="'OK' atau 'DEGRADED (Estimasi Haversine)'")


class OpeningHours(BaseModel):
    """Jam operasional tempat."""
    is_open_now: Optional[bool] = Field(None, description="Apakah buka sekarang")
    weekday_text: Optional[List[str]] = Field(None, description="Jam buka per hari dalam teks")


class PlaceItem(BaseModel):
    """
    Representasi satu tempat dalam itinerary atau rekomendasi.
    Berisi data lengkap yang dibutuhkan frontend untuk render card, pin peta, dan rute.
    """
    # === Identitas ===
    poi_id: Optional[str] = Field(None, description="ID tempat dari database (place_id/UUID)")
    name: str = Field(description="Nama tempat")
    category: Literal["attraction", "restaurant", "hotel"] = Field(
        description="Kategori: 'attraction', 'restaurant', atau 'hotel'"
    )

    # === Lokasi ===
    latitude: Optional[float] = Field(None, description="Koordinat latitude")
    longitude: Optional[float] = Field(None, description="Koordinat longitude")
    district: Optional[str] = Field(None, description="Kabupaten/kecamatan (misal: Ubud, Kuta)")
    address: Optional[str] = Field(None, description="Alamat lengkap")
    google_maps_url: Optional[str] = Field(None, description="Link Google Maps")

    # === Konten & Kualitas ===
    description: str = Field(description="Deskripsi tempat (dari database, TIDAK dikarang)")
    rating: Optional[float] = Field(None, description="Rating (0.0 - 5.0)")
    review_count: Optional[int] = Field(None, description="Jumlah ulasan pengguna")
    price_level: Optional[str] = Field(
        None,
        description="Level harga: PRICE_LEVEL_FREE / INEXPENSIVE / MODERATE / EXPENSIVE / VERY_EXPENSIVE"
    )
    image_url: Optional[str] = Field(None, description="URL foto utama tempat")
    image_urls: Optional[List[str]] = Field(None, description="Daftar URL foto (untuk gallery)")
    tags: Optional[List[str]] = Field(None, description="Tag/label tempat (misal: ['pantai', 'sunset'])")

    # === Jadwal Kunjungan ===
    visit_time: Optional[str] = Field(None, description="Estimasi jam berkunjung (misal: '09:00')")
    duration_mins: Optional[int] = Field(None, description="Estimasi durasi kunjungan dalam menit")
    opening_hours: Optional[OpeningHours] = Field(None, description="Jam operasional")

    # === Biaya ===
    estimated_cost_idr: Optional[int] = Field(0, description="Estimasi biaya dalam IDR")
    ticket_price_info: Optional[str] = Field(None, description="Info harga tiket jika ada")

    # === Peringatan & Status ===
    odalan_warning: Optional[str] = Field(
        None,
        description="Peringatan jika ada Odalan (upacara adat) di tanggal kunjungan"
    )
    is_recommended: Optional[bool] = Field(True, description="Apakah ini rekomendasi utama")
    topsis_score: Optional[float] = Field(None, description="Skor TOPSIS internal (0.0 - 1.0)")

    # === Navigasi ke Tempat Berikutnya ===
    route_to_next: Optional[RouteSegment] = Field(
        None,
        description="Data rute ke tempat berikutnya termasuk polyline untuk render di peta"
    )


class DailyItinerary(BaseModel):
    """Jadwal satu hari perjalanan."""
    day_number: int = Field(description="Nomor hari (mulai dari 1)")
    day_date: Optional[str] = Field(None, description="Tanggal hari ini (YYYY-MM-DD)")
    theme: str = Field(description="Tema hari ini (misal: 'Eksplorasi Pantai Selatan')")
    places: List[PlaceItem] = Field(
        description="Urutan tempat kunjungan (HANYA wisata & restoran, BUKAN hotel)"
    )
    # Summary rute hari ini untuk overview peta
    day_total_distance_km: Optional[float] = Field(
        None, description="Total jarak tempuh hari ini (km)"
    )
    day_total_travel_time_mins: Optional[int] = Field(
        None, description="Total waktu di kendaraan hari ini (menit)"
    )
    day_full_polyline: Optional[List[GeoPoint]] = Field(
        None,
        description="Gabungan semua koordinat rute hari ini (untuk render jalur lengkap di peta)"
    )


class ItinerarySummary(BaseModel):
    """
    Summary ringkas itinerary untuk tampilan LIST.
    Digunakan di endpoint GET /api/itineraries (tidak berisi data hari lengkap).
    Field disesuaikan dengan schema tabel user_itineraries di Supabase.
    """
    id: str = Field(description="UUID itinerary")
    user_id: Optional[str] = Field(None, description="UUID pemilik itinerary")
    title: str = Field(description="Judul trip")
    description: Optional[str] = Field(None, description="Deskripsi singkat trip")
    days_count: int = Field(description="Jumlah hari")
    total_budget_idr: Optional[int] = Field(None, description="Total estimasi budget dalam IDR")
    is_public: bool = Field(description="Apakah itinerary dibagikan ke publik")
    is_owner: bool = Field(description="Apakah user ini pemilik itinerary")
    share_slug: Optional[str] = Field(None, description="Slug unik untuk sharing (misal: /trip/ubud-3-hari)")
    created_at: Optional[str] = Field(None, description="Waktu dibuat (ISO 8601)")
    updated_at: Optional[str] = Field(None, description="Waktu terakhir diubah (ISO 8601)")


# =====================================================================
# MAIN RESPONSE MODEL
# =====================================================================

class FinalAIResponse(BaseModel):
    """
    Model respons utama dari AI Heidi.
    Frontend harus memeriksa `response_type` untuk menentukan tampilan yang sesuai.
    """
    # === Tipe Respons (WAJIB) ===
    response_type: Literal["chat", "recommendation", "clarifying", "itinerary"] = Field(
        description=(
            "Tipe respons AI:\n"
            "- 'chat': Obrolan biasa, tidak ada data tempat\n"
            "- 'recommendation': Daftar rekomendasi tanpa jadwal harian\n"
            "- 'clarifying': AI butuh info tambahan dari user\n"
            "- 'itinerary': Jadwal perjalanan lengkap per hari"
        )
    )
    message_to_user: str = Field(
        description=(
            "Pesan dari Heidi untuk ditampilkan ke user. "
            "**SEMUA response_type WAJIB menggunakan format Markdown** agar frontend bisa me-render dengan baik. "
            "Panduan format per tipe:\n\n"
            "- **'chat'**: Markdown ringan. Boleh pakai **bold**, *italic*, bullet list, dan emoji. "
            "Contoh: menyambut user, menjawab pertanyaan singkat, atau konfirmasi aksi.\n\n"
            "- **'clarifying'**: Markdown dengan struktur jelas. Gunakan **bold** untuk pertanyaan utama "
            "dan bullet list untuk pilihan/opsi yang bisa dijawab user.\n\n"
            "- **'recommendation'**: Gunakan heading `##` per kategori rekomendasi, **bold** nama tempat, "
            "bullet list untuk keunggulan tiap tempat, dan emoji kategori (🏖️ 🏛️ 🍜 dll).\n\n"
            "- **'itinerary'**: Narasi storytelling panjang (min. 300 kata). Struktur: "
            "`# Trip Title` → `## 🌅 Hari N — Tanggal · Tema` (per hari, cerita mengalir) "
            "→ `## 💰 Estimasi Budget` (breakdown) → closing 1-2 kalimat. "
            "Gunakan **bold** untuk nama tempat, emoji di setiap heading hari, gaya akrab ('kamu')."
        )
    )


    # === Data Itinerary (hanya untuk response_type='itinerary') ===
    itinerary_id: Optional[str] = Field(
        None,
        description="UUID itinerary setelah disimpan ke database (auto-filled oleh backend)"
    )
    trip_title: Optional[str] = Field(
        None,
        description="Judul perjalanan yang digenerate AI (misal: 'Petualangan Ubud 3 Hari')"
    )
    base_hotel: Optional[PlaceItem] = Field(
        None,
        description=(
            "Hotel utama selama perjalanan (SATU hotel untuk semua hari - TC-07). "
            "Hotel INI TIDAK BOLEH muncul lagi di dalam places harian."
        )
    )
    itinerary_days: Optional[List[DailyItinerary]] = Field(
        None,
        description="Jadwal harian. Hanya ada jika response_type='itinerary'."
    )
    total_budget_idr: Optional[int] = Field(
        None,
        description="Total estimasi budget semua hari dalam IDR"
    )
    budget_breakdown: Optional[dict] = Field(
        None,
        description=(
            "Rincian budget per kategori, misal: "
            "{'accommodation': 1500000, 'food': 800000, 'transport': 300000, 'tickets': 200000}"
        )
    )

    # === Data Rekomendasi (hanya untuk response_type='recommendation') ===
    recommendations: Optional[List[PlaceItem]] = Field(
        None,
        description="Daftar rekomendasi tempat. Hanya ada jika response_type='recommendation'."
    )

    # === UX Helpers ===
    suggested_replies: List[str] = Field(
        default_factory=list,
        description="Saran balasan cepat untuk ditampilkan sebagai tombol di chat UI"
    )
    clarifying_questions: Optional[List[str]] = Field(
        None,
        description=(
            "Daftar pertanyaan spesifik yang perlu dijawab user. "
            "Hanya ada jika response_type='clarifying'."
        )
    )