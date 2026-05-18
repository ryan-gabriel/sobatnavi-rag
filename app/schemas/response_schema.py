# app/schemas/response_schema.py
"""
FinalAIResponse — skema output tunggal untuk seluruh sistem Heidi.

PENTING UNTUK DEVELOPER:
  Schema ini di-inject ke dalam system prompt OpenAI via:
      json.dumps(FinalAIResponse.model_json_schema(), indent=2)
  Artinya field `description=` pada setiap Field() akan DIBACA LANGSUNG oleh LLM.
  Tulis description yang instruktif, bukan sekadar komentar kode.

PENTING UNTUK LLM (Heidi):
  Baca setiap `description` di bawah ini dengan seksama — itulah aturan pengisian field.
  Jangan pernah menggunakan nilai di luar yang diizinkan.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY COERCION
# ══════════════════════════════════════════════════════════════════════════════
# LLM kadang menggunakan nilai semantik yang tepat tapi bukan nilai kanonik.
# Validator ini memetakan SEMUA varian ke salah satu dari tiga nilai resmi:
# "attraction" | "restaurant" | "hotel"

_CATEGORY_ALIAS: dict[str, str] = {
    # ── nilai kanonik (identity) ──────────────────────────────────────────
    "attraction":     "attraction",
    "restaurant":     "restaurant",
    "hotel":          "hotel",

    # ── → attraction ──────────────────────────────────────────────────────
    "temple":         "attraction",
    "pura":           "attraction",
    "beach":          "attraction",
    "pantai":         "attraction",
    "zoo":            "attraction",
    "kebun binatang": "attraction",
    "museum":         "attraction",
    "park":           "attraction",
    "taman":          "attraction",
    "waterfall":      "attraction",
    "air terjun":     "attraction",
    "market":         "attraction",
    "pasar":          "attraction",
    "landmark":       "attraction",
    "nature":         "attraction",
    "alam":           "attraction",
    "cultural":       "attraction",
    "budaya":         "attraction",
    "spa":            "attraction",
    "activity":       "attraction",
    "tour":           "attraction",
    "wisata":         "attraction",
    "rice terrace":   "attraction",
    "sawah":          "attraction",
    "cliff":          "attraction",
    "tebing":         "attraction",
    "snorkeling":     "attraction",
    "diving":         "attraction",
    "gallery":        "attraction",
    "galeri":         "attraction",
    "art":            "attraction",
    "seni":           "attraction",
    "cooking class":  "attraction",
    "workshop":       "attraction",
    "viewpoint":      "attraction",
    "sunset point":   "attraction",
    "hot spring":     "attraction",
    "waterpark":      "attraction",
    "adventure":      "attraction",

    # ── → restaurant ─────────────────────────────────────────────────────
    "cafe":           "restaurant",
    "kafe":           "restaurant",
    "coffee":         "restaurant",
    "kopi":           "restaurant",
    "food":           "restaurant",
    "makanan":        "restaurant",
    "culinary":       "restaurant",
    "kuliner":        "restaurant",
    "bar":            "restaurant",
    "dining":         "restaurant",
    "eatery":         "restaurant",
    "warung":         "restaurant",
    "rumah makan":    "restaurant",
    "seafood":        "restaurant",
    "bakery":         "restaurant",
    "bakeri":         "restaurant",
    "bistro":         "restaurant",
    "diner":          "restaurant",
    "food court":     "restaurant",
    "warungs":        "restaurant",

    # ── → hotel ──────────────────────────────────────────────────────────
    "lodging":        "hotel",
    "penginapan":     "hotel",
    "accommodation":  "hotel",
    "akomodasi":      "hotel",
    "resort":         "hotel",
    "villa":          "hotel",
    "hostel":         "hotel",
    "guesthouse":     "hotel",
    "guest house":    "hotel",
    "inn":            "hotel",
    "homestay":       "hotel",
    "bungalow":       "hotel",
    "glamping":       "hotel",
    "airbnb":         "hotel",
}

_VALID_CATEGORIES = {"attraction", "restaurant", "hotel"}


def _coerce_category(value: Any) -> str:
    """
    Normalise kategori ke salah satu dari: 'attraction', 'restaurant', 'hotel'.
    Dipanggil SEBELUM validasi Pydantic (mode='before').

    Urutan:
      1. Lookup langsung di alias table (case-insensitive, strip whitespace).
      2. Cek apakah salah satu nilai kanonik merupakan substring dari value.
      3. Fallback ke 'attraction' + log warning agar developer tahu alias baru
         apa yang perlu ditambahkan.
    """
    if not isinstance(value, str):
        logger.warning(f"category bukan string: {value!r} → fallback 'attraction'")
        return "attraction"

    normalised = value.strip().lower()

    # 1. Direct / alias lookup
    if normalised in _CATEGORY_ALIAS:
        mapped = _CATEGORY_ALIAS[normalised]
        if mapped != normalised:
            logger.debug(f"category coerced: '{value}' → '{mapped}'")
        return mapped

    # 2. Substring match against canonical values
    for canonical in _VALID_CATEGORIES:
        if canonical in normalised:
            logger.debug(f"category substring match: '{value}' → '{canonical}'")
            return canonical

    # 3. Last resort — don't crash, but log so developer can add alias
    logger.warning(
        f"Unknown category '{value}' coerced to 'attraction'. "
        "Tambahkan alias baru di _CATEGORY_ALIAS jika ini sering terjadi."
    )
    return "attraction"


# ══════════════════════════════════════════════════════════════════════════════
# SUB-MODELS
# ══════════════════════════════════════════════════════════════════════════════

class RouteSegment(BaseModel):
    """
    Data rute satu segmen perjalanan antar dua titik (hasil dari calculate_batch_routes).
    Ambil dari field `segments[i]` hasil pemanggilan tool calculate_batch_routes.
    Sisipkan ke field `route_to_next` milik PlaceItem yang bersangkutan.
    """

    distance_km: Optional[float] = Field(
        default=None,
        description=(
            "Jarak segmen dalam kilometer. "
            "Ambil dari segments[i].distance_km hasil calculate_batch_routes."
        ),
    )
    travel_time_mins: Optional[int] = Field(
        default=None,
        description=(
            "Estimasi waktu perjalanan dalam menit (sudah memperhitungkan kondisi lalu lintas). "
            "Ambil dari segments[i].travel_time_mins hasil calculate_batch_routes."
        ),
    )
    traffic_delay_mins: Optional[int] = Field(
        default=None,
        description=(
            "Tambahan waktu akibat kemacetan dalam menit. "
            "Ambil dari segments[i].traffic_delay_mins. Bisa 0 jika tidak ada kemacetan."
        ),
    )
    polyline: Optional[List[Dict]] = Field(
        default=None,
        description=(
            "Daftar titik koordinat rute dalam format [{\"lat\": float, \"lng\": float}, ...]. "
            "Digunakan oleh frontend untuk menggambar jalur di peta. "
            "Ambil dari segments[i].polyline hasil calculate_batch_routes."
        ),
    )
    status: Optional[str] = Field(
        default=None,
        description=(
            "'OK' jika data langsung dari TomTom API. "
            "'DEGRADED (Estimasi Haversine)' jika TomTom API gagal dan sistem pakai fallback jarak lurus."
        ),
    )


class PlaceItem(BaseModel):
    """
    Satu tempat/destinasi dalam jadwal harian.
    SEMUA data HARUS berasal dari hasil tool call — DILARANG mengarang.

    ATURAN WAJIB:
    - `category` HANYA boleh: "attraction", "restaurant", atau "hotel".
      DILARANG menggunakan: "temple", "zoo", "beach", "cafe", "museum",
      "lodging", "cultural", "spa", atau kata lain apapun.
    - `poi_id` dan `place_id` HARUS diambil dari database, JANGAN ditulis manual.
    - `latitude` dan `longitude` HARUS dari database, JANGAN dikira-kira.
    - `description` WAJIB diisi dari field `content` di database.
    - Hotel base TIDAK BOLEH masuk ke dalam `places` — gunakan field `base_hotel`.
    """

    place_id: Optional[str] = Field(
        default=None,
        description=(
            "Google Place ID tempat ini (format string: 'ChIJ...'). "
            "Ambil dari field `place_id` hasil tool. JANGAN dikarang atau ditebak."
        ),
    )
    poi_id: Optional[int] = Field(
        default=None,
        description=(
            "ID integer dari tabel poi_attractions di Supabase. "
            "Ambil dari field `id` atau `poi_id` hasil get_smart_recommendations. "
            "Dibutuhkan oleh frontend dan untuk validate_itinerary_safety. "
            "JANGAN dikarang."
        ),
    )
    name: str = Field(
        description=(
            "Nama resmi tempat persis seperti yang ada di database. "
            "JANGAN disingkat, JANGAN diterjemahkan, JANGAN dimodifikasi."
        ),
    )
    category: str = Field(
        description=(
            "Kategori tempat. "
            "NILAI YANG DIIZINKAN — HANYA TIGA INI SAJA:\n"
            "  \"attraction\" → untuk SEMUA tempat wisata tanpa terkecuali: "
            "pura/temple, pantai/beach, museum, kebun binatang/zoo, taman/park, "
            "air terjun/waterfall, sawah/rice terrace, tebing/cliff, galeri seni, "
            "spa, cooking class, snorkeling, diving, viewpoint, hot spring, dll.\n"
            "  \"restaurant\" → untuk SEMUA tempat makan & minum: restoran, warung, "
            "cafe/kafe, coffee shop, bar, bistro, food court, seafood, bakery, dll.\n"
            "  \"hotel\" → untuk SEMUA akomodasi: hotel, villa, resort, hostel, "
            "guesthouse, homestay, bungalow, glamping, dll.\n"
            "DILARANG KERAS menggunakan nilai lain seperti: 'temple', 'zoo', 'beach', "
            "'cafe', 'museum', 'lodging', 'cultural', 'spa', 'nature', dll."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "Deskripsi lengkap tempat ini. "
            "WAJIB diisi dari field `content` di hasil tool — jangan dipersingkat atau dipotong. "
            "Jika field `content` kosong dari database, gunakan informasi lain dari metadata "
            "seperti nama, district, rating, dan tags untuk membuat narasi minimal 2-3 kalimat. "
            "Contoh jika data kosong: "
            "'Pura Tirta Empul adalah salah satu pura Hindu paling sakral di Bali, terletak di "
            "Tampaksiring, Gianyar. Terkenal dengan mata air suci yang digunakan untuk ritual "
            "pembersihan (melukat). Wisatawan bisa menyaksikan upacara dan masuk ke kolam dengan "
            "menyewa sarung di pintu masuk.'"
        ),
    )
    latitude: Optional[float] = Field(
        default=None,
        description=(
            "Koordinat latitude (lintang) dalam format desimal. "
            "Contoh: -8.6152 untuk Kuta, -8.5069 untuk Tanah Lot. "
            "HARUS dari database — DILARANG dikira-kira, diestimasi, atau dikarang."
        ),
    )
    longitude: Optional[float] = Field(
        default=None,
        description=(
            "Koordinat longitude (bujur) dalam format desimal. "
            "Contoh: 115.1310 untuk Kuta, 115.0863 untuk Tanah Lot. "
            "HARUS dari database — DILARANG dikira-kira, diestimasi, atau dikarang."
        ),
    )
    district: Optional[str] = Field(
        default=None,
        description=(
            "Kabupaten/kecamatan lokasi tempat di Bali. "
            "Nilai umum: 'Badung', 'Gianyar', 'Tabanan', 'Buleleng', 'Karangasem', "
            "'Klungkung', 'Bangli', 'Jembrana', 'Denpasar'. "
            "Ambil dari field `district` hasil tool."
        ),
    )
    image_url: Optional[str] = Field(
        default=None,
        description=(
            "URL foto utama tempat ini (jpeg/png/webp yang bisa ditampilkan langsung). "
            "Ambil dari field `image_url` hasil tool. "
            "Jika tidak ada di database, set null — JANGAN membuat URL palsu atau placeholder."
        ),
    )
    rating: Optional[float] = Field(
        default=None,
        description=(
            "Rating tempat skala 1.0–5.0 (Google Maps rating). "
            "Ambil dari field `rating` hasil tool. Contoh: 4.7"
        ),
    )
    user_rating_count: Optional[int] = Field(
        default=None,
        description=(
            "Jumlah total ulasan pengguna. "
            "Ambil dari field `user_rating_count` hasil tool. Contoh: 12500"
        ),
    )
    estimated_cost_idr: Optional[int] = Field(
        default=None,
        description=(
            "Estimasi biaya kunjungan dalam Rupiah (IDR) per orang. "
            "Untuk attraction: harga tiket masuk. "
            "Untuk restaurant: estimasi pengeluaran makan per orang. "
            "Contoh: 50000 (tiket Rp 50.000), 150000 (makan Rp 150.000/orang). "
            "Estimasi dari price_level metadata jika tidak ada data eksak."
        ),
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "Label deskriptif untuk filtering dan pencarian di frontend. "
            "Contoh: [\"sunset\", \"fotografi\", \"budaya\", \"keluarga\", \"murah\", \"vegetarian\"]. "
            "Ambil dari metadata database atau inferensikan dari konten tempat."
        ),
    )
    visit_duration_mins: Optional[int] = Field(
        default=None,
        description=(
            "Estimasi durasi kunjungan ideal dalam menit. "
            "Panduan: pura kecil 30-45 mnt, pura besar/Besakih 90-120 mnt, "
            "pantai 60-90 mnt, museum 60-90 mnt, kebun binatang 120-180 mnt, "
            "restoran 45-60 mnt, spa 90-120 mnt. "
            "Sesuaikan dengan ukuran dan jenis tempat."
        ),
    )
    visit_time: Optional[str] = Field(
        default=None,
        description=(
            "Waktu kunjungan yang disarankan dalam format HH:MM (24 jam). "
            "Pertimbangkan: urutan tempat dalam hari, jam operasional, "
            "dan waktu terbaik (mis. sunrise/sunset). "
            "Contoh: '08:00' (pagi, sebelum ramai), '12:30' (makan siang), '17:00' (sunset)."
        ),
    )
    opening_hours_note: Optional[str] = Field(
        default=None,
        description=(
            "Catatan jam operasional jika relevan untuk perencanaan. "
            "Contoh: 'Buka setiap hari 07:00–19:00, paling sepi sebelum jam 09:00'. "
            "Set null jika tidak ada info spesifik."
        ),
    )
    tips: Optional[str] = Field(
        default=None,
        description=(
            "Tips kunjungan yang praktis, spesifik, dan berguna. "
            "Contoh: 'Bawa sarung, wajib dipakai saat masuk area pura (bisa sewa Rp 10.000 di pintu)', "
            "'Datang sebelum jam 08:00 untuk hindari antrian dan foto lebih bagus', "
            "'Parkir motor lebih mudah dan murah daripada mobil'. "
            "Tulis dalam bahasa Indonesia akrab. Set null jika tidak ada tips relevan."
        ),
    )
    route_to_next: Optional[RouteSegment] = Field(
        default=None,
        description=(
            "Data rute dari tempat INI ke tempat BERIKUTNYA dalam urutan jadwal hari itu. "
            "WAJIB diisi dari hasil calculate_batch_routes — ambil dari segments[i] "
            "di mana i adalah indeks urutan tempat ini dalam array waypoints. "
            "Contoh: PlaceItem pertama (POI1) → route_to_next = segments[0] (rute hotel→POI1... "
            "atau lebih tepatnya POI1→POI2). "
            "PlaceItem TERAKHIR dalam satu hari (biasanya restoran malam/POI terakhir) → "
            "route_to_next = segmen rute kembali ke hotel. "
            "JANGAN biarkan null jika calculate_batch_routes sudah berhasil dipanggil."
        ),
    )

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v: Any) -> str:
        return _coerce_category(v)


class BaseHotel(BaseModel):
    """
    Hotel utama yang menjadi pusat/home base seluruh itinerary.

    ATURAN WAJIB:
    - SATU hotel untuk SEMUA hari — JANGAN ganti hotel di tengah itinerary.
    - Hotel ini TIDAK BOLEH muncul di dalam `places` di DailyItinerary manapun.
    - Semua data HARUS dari hasil tool get_nearby_food_and_lodging(category='hotel').
    - Field `description` WAJIB diisi — sistem akan auto-generate jika LLM lupa,
      tapi LLM tetap HARUS mengisi dari data database.
    """

    place_id: Optional[str] = Field(
        default=None,
        description=(
            "Google Place ID hotel (format: 'ChIJ...'). "
            "Ambil dari field `place_id` hasil tool. JANGAN dikarang."
        ),
    )
    name: str = Field(
        description=(
            "Nama resmi hotel persis seperti di database. "
            "JANGAN disingkat, diterjemahkan, atau dimodifikasi."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "WAJIB DIISI. Deskripsi hotel yang informatif dan menarik. "
            "Prioritas: ambil dari field `content` di database. "
            "Jika tidak ada data dari database, BUAT SENDIRI minimal 3 kalimat yang mencakup: "
            "(1) posisi/lokasi strategis hotel, "
            "(2) keunggulan utama dan nuansa atmosfer, "
            "(3) fasilitas andalan dan cocok untuk siapa. "
            "Contoh deskripsi yang dibuat sendiri: "
            "'Komaneka at Bisma adalah boutique resort mewah yang menggantung di tebing Ubud "
            "dengan pemandangan hutan dan Sungai Wos yang dramatis. Setiap villa dilengkapi "
            "kolam renang pribadi dan teras menghadap lembah hijau yang menenangkan jiwa. "
            "Ideal untuk pasangan atau solo traveler yang mencari ketenangan di tengah "
            "pusat seni dan budaya Ubud, hanya 5 menit berjalan kaki ke Monkey Forest Road.' "
            "JANGAN biarkan field ini null atau berisi string kosong."
        ),
    )
    latitude: Optional[float] = Field(
        default=None,
        description=(
            "Koordinat latitude hotel dalam format desimal. "
            "HARUS dari database — DILARANG dikira-kira."
        ),
    )
    longitude: Optional[float] = Field(
        default=None,
        description=(
            "Koordinat longitude hotel dalam format desimal. "
            "HARUS dari database — DILARANG dikira-kira."
        ),
    )
    district: Optional[str] = Field(
        default=None,
        description=(
            "Kabupaten/area lokasi hotel di Bali. "
            "Contoh: 'Ubud', 'Seminyak', 'Nusa Dua', 'Kuta', 'Canggu', 'Sanur'. "
            "Ambil dari field `district` hasil tool."
        ),
    )
    image_url: Optional[str] = Field(
        default=None,
        description=(
            "URL foto utama hotel. Ambil dari `image_url` hasil tool. "
            "Set null jika tidak ada — JANGAN buat URL palsu."
        ),
    )
    rating: Optional[float] = Field(
        default=None,
        description="Rating hotel skala 1.0–5.0. Contoh: 4.6",
    )
    user_rating_count: Optional[int] = Field(
        default=None,
        description="Jumlah ulasan pengguna hotel. Contoh: 3200",
    )
    price_per_night_idr: Optional[int] = Field(
        default=None,
        description=(
            "Estimasi harga kamar per malam dalam Rupiah (IDR). "
            "Ambil dari metadata atau estimasi dari price_level dan bintang hotel. "
            "Panduan kasar: budget <300rb, midrange 300rb-1jt, luxury >1jt/malam. "
            "Contoh: 850000 untuk Rp 850.000/malam."
        ),
    )
    amenities: Optional[List[str]] = Field(
        default=None,
        description=(
            "Daftar fasilitas utama hotel dalam Bahasa Indonesia. "
            "Contoh: [\"kolam renang\", \"wifi gratis\", \"sarapan termasuk\", "
            "\"spa\", \"parkir gratis\", \"AC\", \"restoran\", \"antar-jemput bandara\"]. "
            "Ambil dari field amenities di metadata database."
        ),
    )
    check_in_time: Optional[str] = Field(
        default="14:00",
        description=(
            "Jam check-in standar hotel dalam format HH:MM. "
            "Default: '14:00'. Ambil dari metadata jika tersedia."
        ),
    )
    check_out_time: Optional[str] = Field(
        default="12:00",
        description=(
            "Jam check-out standar hotel dalam format HH:MM. "
            "Default: '12:00'. Ambil dari metadata jika tersedia."
        ),
    )

    @model_validator(mode="after")
    def ensure_description(self) -> "BaseHotel":
        """
        Safety net: auto-generate description minimal jika LLM tidak mengisinya.
        Ini adalah fallback — LLM tetap HARUS mengisi dari database.
        """
        if not self.description:
            parts = [f"{self.name}"]
            if self.district:
                parts.append(f"berlokasi di {self.district}, Bali")
            if self.rating:
                parts.append(f"dengan rating {self.rating}/5")
            if self.amenities:
                top = ", ".join(self.amenities[:3])
                parts.append(f"dilengkapi fasilitas {top}")
            if self.price_per_night_idr:
                formatted = f"Rp {self.price_per_night_idr:,}".replace(",", ".")
                parts.append(f"harga mulai {formatted}/malam")
            self.description = ". ".join(parts) + "."
            logger.debug(
                f"BaseHotel.description auto-generated untuk '{self.name}': {self.description}"
            )
        return self


class BudgetBreakdown(BaseModel):
    """
    Rincian estimasi biaya perjalanan per kategori dalam Rupiah (IDR).
    Semua nilai integer (tidak ada desimal). Jumlah semua komponen = total_budget_idr.
    """

    accommodation_idr: Optional[int] = Field(
        default=None,
        description=(
            "Total biaya akomodasi seluruh trip dalam IDR. "
            "Rumus: base_hotel.price_per_night_idr × jumlah_malam. "
            "Contoh: 850000 × 3 malam = 2550000."
        ),
    )
    food_idr: Optional[int] = Field(
        default=None,
        description=(
            "Total estimasi biaya makan & minum seluruh trip dalam IDR. "
            "Hitung dari estimated_cost_idr setiap PlaceItem berkategori 'restaurant' "
            "× jumlah_orang, ditambah estimasi jajan/kopi di luar jadwal (~50rb/hari/orang)."
        ),
    )
    transport_idr: Optional[int] = Field(
        default=None,
        description=(
            "Total estimasi biaya transportasi seluruh trip dalam IDR "
            "(sewa motor/mobil, BBM, ojek/taksi, parkir). "
            "Panduan: sewa motor Rp 80.000/hari, sewa mobil+driver Rp 500.000/hari. "
            "Estimasi juga dari total_distance_km harian jika ada."
        ),
    )
    entrance_fee_idr: Optional[int] = Field(
        default=None,
        description=(
            "Total estimasi tiket masuk semua attraction seluruh trip dalam IDR. "
            "Jumlahkan estimated_cost_idr dari semua PlaceItem berkategori 'attraction' "
            "× jumlah_orang."
        ),
    )
    miscellaneous_idr: Optional[int] = Field(
        default=None,
        description=(
            "Biaya lain-lain dalam IDR: oleh-oleh, tips pemandu lokal, "
            "sewa sarung di pura, donasi, dll. "
            "Estimasi 10–15% dari total komponen lainnya."
        ),
    )


class DailyItinerary(BaseModel):
    """
    Jadwal satu hari perjalanan lengkap beserta semua tempat dan data rute.

    ATURAN WAJIB:
    - `places` HARUS berisi minimal 2 attraction DAN minimal 1 restaurant per hari.
    - Urutan `places` harus logis secara geografis untuk meminimalkan backtracking.
    - `route_to_next` di setiap PlaceItem WAJIB diisi dari hasil calculate_batch_routes.
    - Hotel base (base_hotel) TIDAK MASUK ke dalam `places`.
    - Setiap hari WAJIB memanggil calculate_batch_routes satu kali untuk mengisi
      day_total_distance_km, day_total_travel_time_mins, dan day_full_polyline.
    """

    day: int = Field(
        description=(
            "Nomor hari perjalanan, mulai dari 1. "
            "Hari pertama = 1, hari kedua = 2, dst."
        ),
    )
    date: Optional[str] = Field(
        default=None,
        description=(
            "Tanggal kalender hari ini dalam format YYYY-MM-DD. "
            "Hitung dari tanggal keberangkatan + (day - 1). "
            "Contoh: jika berangkat 2026-05-20 dan ini hari ke-2, maka '2026-05-21'."
        ),
    )
    theme: Optional[str] = Field(
        default=None,
        description=(
            "Tema/judul singkat hari ini yang deskriptif dan menarik. "
            "Contoh: 'Spiritual & Budaya Ubud', 'Sunset di Tebing Barat', "
            "'Kuliner & Belanja Seminyak', 'Alam Hijau Munduk & Bedugul'. "
            "Maksimal 5 kata. Pilih berdasarkan mayoritas jenis tempat hari itu."
        ),
    )
    places: List[PlaceItem] = Field(
        default_factory=list,
        description=(
            "Daftar tempat yang dikunjungi hari ini, berurutan sesuai jadwal. "
            "ATURAN URUTAN: Susun searah/berdekatan secara geografis untuk minimasi backtracking. "
            "Umum: POI pagi (attraction) → attraction siang → restaurant makan siang → "
            "attraction sore → restaurant makan malam. "
            "WAJIB: minimal 2 attraction + minimal 1 restaurant per hari. "
            "WAJIB: setiap PlaceItem punya route_to_next terisi dari calculate_batch_routes. "
            "DILARANG: memasukkan hotel base ke dalam list ini."
        ),
    )
    day_total_distance_km: Optional[float] = Field(
        default=None,
        description=(
            "Total jarak berkendara hari ini dalam km. "
            "Ambil dari field `total_distance_km` hasil calculate_batch_routes untuk hari ini."
        ),
    )
    day_total_travel_time_mins: Optional[int] = Field(
        default=None,
        description=(
            "Total waktu berkendara hari ini dalam menit (tidak termasuk waktu di destinasi). "
            "Ambil dari field `total_travel_time_mins` hasil calculate_batch_routes untuk hari ini."
        ),
    )
    day_full_polyline: Optional[List[Dict]] = Field(
        default=None,
        description=(
            "Gabungan semua titik koordinat rute hari ini untuk render jalur di peta. "
            "Format: [{\"lat\": float, \"lng\": float}, ...]. "
            "Ambil dari field `full_day_polyline` hasil calculate_batch_routes untuk hari ini."
        ),
    )
    odalan_warning: Optional[str] = Field(
        default=None,
        description=(
            "Peringatan jika ada konflik upacara Odalan di area kunjungan hari ini. "
            "Isi berdasarkan hasil get_bali_context atau validate_itinerary_safety. "
            "Contoh: 'Pura Besakih mengadakan upacara Odalan hari ini — "
            "kemungkinan tertutup untuk wisatawan atau sangat padat. "
            "Disarankan kunjungi alternatif seperti Pura Kehen di Bangli.' "
            "Set null jika tidak ada konflik."
        ),
    )
    weather_note: Optional[str] = Field(
        default=None,
        description=(
            "Catatan cuaca hari ini dari hasil get_bali_context (jika relevan). "
            "Contoh: 'Prakiraan hujan sore hari ab jam 15:00 — bawa jas hujan atau "
            "jadwalkan aktivitas outdoor sebelum jam 14:00.' "
            "Set null jika tidak ada info cuaca spesifik untuk hari ini."
        ),
    )


class RecommendationItem(BaseModel):
    """
    Satu item rekomendasi tempat, dipakai saat response_type='recommendation'.
    Semua data HARUS dari hasil tool get_smart_recommendations.
    """

    place_id: Optional[str] = Field(
        default=None,
        description="Google Place ID. Ambil dari hasil tool. JANGAN dikarang.",
    )
    poi_id: Optional[int] = Field(
        default=None,
        description="ID integer dari database. Ambil dari field `id` hasil tool.",
    )
    name: str = Field(
        description="Nama resmi tempat persis seperti di database.",
    )
    category: str = Field(
        description=(
            "Kategori tempat. "
            "HANYA tiga nilai yang diizinkan: "
            "\"attraction\" (semua tempat wisata), "
            "\"restaurant\" (semua tempat makan/minum), "
            "\"hotel\" (semua akomodasi). "
            "JANGAN gunakan nilai lain."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "Deskripsi singkat tapi menarik dari field `content` database. "
            "Minimal 1-2 kalimat yang membuat user tertarik mengunjungi."
        ),
    )
    district: Optional[str] = Field(
        default=None,
        description="Kabupaten/area lokasi di Bali.",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL foto tempat. Set null jika tidak ada.",
    )
    rating: Optional[float] = Field(
        default=None,
        description="Rating 1.0–5.0.",
    )
    latitude: Optional[float] = Field(
        default=None,
        description="Koordinat latitude dari database.",
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Koordinat longitude dari database.",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Label deskriptif. Contoh: [\"sunset\", \"fotografi\", \"keluarga\"].",
    )
    estimated_cost_idr: Optional[int] = Field(
        default=None,
        description="Estimasi biaya kunjungan per orang dalam IDR.",
    )
    topsis_score: Optional[float] = Field(
        default=None,
        description=(
            "Skor TOPSIS (0.0–1.0) dari engine rekomendasi — semakin tinggi semakin direkomendasikan. "
            "Ambil dari field `topsis_score` hasil tool jika tersedia."
        ),
    )

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v: Any) -> str:
        return _coerce_category(v)


# ══════════════════════════════════════════════════════════════════════════════
# ROOT RESPONSE — FinalAIResponse
# ══════════════════════════════════════════════════════════════════════════════

class FinalAIResponse(BaseModel):
    """
    ═══════════════════════════════════════════════════════════════════════════
    OUTPUT UTAMA HEIDI — Selalu kembalikan struktur JSON ini tanpa terkecuali.
    Tidak boleh ada teks di luar JSON. Tidak boleh ada markdown ```json```.

    PILIH response_type yang sesuai dengan konteks permintaan user:
    ┌──────────────────┬────────────────────────────────────────────────────┐
    │ response_type    │ Kapan digunakan                                    │
    ├──────────────────┼────────────────────────────────────────────────────┤
    │ "chat"           │ Sapaan, obrolan biasa, pertanyaan umum Bali        │
    │ "clarifying"     │ Data kurang (HANYA di mode deep_research)          │
    │ "recommendation" │ User minta rekomendasi tempat tanpa jadwal penuh   │
    │ "itinerary"      │ User minta itinerary/jadwal perjalanan lengkap     │
    └──────────────────┴────────────────────────────────────────────────────┘

    FIELD WAJIB PER response_type:
    • "chat":           message_to_user, suggested_replies
    • "clarifying":     message_to_user, clarifying_questions, suggested_replies
    • "recommendation": message_to_user, recommendations, suggested_replies
    • "itinerary":      message_to_user, trip_title, base_hotel, itinerary_days,
                        total_budget_idr, budget_breakdown, suggested_replies

    INGAT: message_to_user SELALU Markdown — TIDAK BOLEH teks plain.
    ═══════════════════════════════════════════════════════════════════════════
    """

    response_type: Literal["chat", "clarifying", "recommendation", "itinerary"] = Field(
        description=(
            "Tipe respons — pilih SATU nilai yang paling sesuai:\n"
            "  'chat'           → obrolan biasa, sapaan, pertanyaan umum.\n"
            "  'clarifying'     → butuh info lebih dari user (HANYA di mode deep_research).\n"
            "  'recommendation' → daftar rekomendasi tempat tanpa itinerary penuh.\n"
            "  'itinerary'      → jadwal perjalanan lengkap dengan hotel, rute, dan budget."
        ),
    )

    message_to_user: str = Field(
        description=(
            "WAJIB MARKDOWN — Pesan utama yang ditampilkan di chat UI. "
            "Frontend me-render Markdown ini secara visual. "
            "JANGAN tulis teks plain — gunakan: **bold**, *italic*, # heading, ## subheading, "
            "- bullet list, * list, emoji.\n\n"
            "FORMAT YANG DIHARAPKAN PER TIPE:\n\n"
            "── 'chat' ──\n"
            "Sapaan hangat + perkenalan Heidi + list kemampuan + ajakan mulai.\n"
            "Contoh:\n"
            "Halo! 👋 Aku **Heidi**, asisten perjalanan AI spesialis Bali dari **SobatNavi**. 🌴\n\n"
            "Aku siap bantu kamu:\n"
            "- 🗺️ Membuat itinerary wisata Bali yang personal\n"
            "- 🏖️ Rekomendasi tempat wisata, hotel & kuliner terbaik\n"
            "- ℹ️ Info Odalan & kondisi perjalanan real-time\n\n"
            "Mau ke mana di Bali? 😊\n\n"
            "── 'clarifying' ──\n"
            "Daftar pertanyaan dengan bullet + emoji per poin.\n"
            "Contoh:\n"
            "Hampir siap! 😊 Aku butuh beberapa info dulu:\n\n"
            "- 📍 Area mana di Bali yang ingin kamu kunjungi?\n"
            "- 📅 Tanggal berapa kamu berangkat?\n"
            "- 👥 Pergi berdua, keluarga, atau rombongan?\n\n"
            "── 'recommendation' ──\n"
            "Heading ## per kategori, nama tempat **bold**, deskripsi 1 kalimat per tempat.\n"
            "Contoh:\n"
            "Ini rekomendasiku! 🗺️\n\n"
            "## 🏖️ Pantai Terbaik\n"
            "- **Pantai Pandawa** — Tersembunyi di balik tebing kapur, pasir putih bersih, cocok untuk foto.\n"
            "- **Nusa Dua** — Ombak tenang, cocok untuk keluarga dengan anak kecil.\n\n"
            "## 🛕 Wisata Budaya\n"
            "- **Pura Tanah Lot** — Pura ikonik di atas batu karang, paling dramatis saat matahari terbenam.\n\n"
            "── 'itinerary' ──\n"
            "Narasi storytelling panjang MIN. 300 KATA dengan struktur:\n"
            "# ✈️ [Judul Trip yang Catchy]\n\n"
            "[Opening 2-3 kalimat gambaran umum perjalanan]\n\n"
            "---\n\n"
            "## 🌅 Hari 1 — [Tanggal] · [Tema Hari]\n"
            "[Narasi mengalir: dari hotel → tempat pertama (kenapa menarik) → "
            "tempat kedua (ciri khas) → makan siang di restoran → "
            "tempat sore → makan malam. Gunakan 'kamu' bukan 'Anda'.]\n\n"
            "## 🌿 Hari 2 — [Tanggal] · [Tema Hari]\n"
            "[Lanjutkan cerita...]\n\n"
            "---\n\n"
            "## 💰 Estimasi Budget\n"
            "- 🏨 Akomodasi: Rp X.XXX.XXX\n"
            "- 🍽️ Makan & Minum: Rp X.XXX.XXX\n"
            "- 🚗 Transportasi: Rp X.XXX.XXX\n"
            "- 🎟️ Tiket Masuk: Rp X.XXX.XXX\n"
            "- 🛍️ Lain-lain: Rp X.XXX.XXX\n"
            "- **Total: Rp X.XXX.XXX**\n\n"
            "---\n"
            "[Closing 1-2 kalimat penyemangat + ajakan tanya lagi]"
        ),
    )

    suggested_replies: List[str] = Field(
        default_factory=list,
        description=(
            "WAJIB berisi TEPAT 3 saran pertanyaan/aksi lanjutan yang relevan dan berguna. "
            "JANGAN biarkan kosong. JANGAN kurang dari 3.\n"
            "Contoh untuk 'itinerary': "
            "[\"Cari hotel lebih murah di area ini\", \"Tambah 1 hari ke itinerary\", \"Lihat alternatif pantai\"].\n"
            "Contoh untuk 'recommendation': "
            "[\"Buatkan itinerary dari rekomendasi ini\", \"Rekomendasikan restoran di area Ubud\", "
            "\"Tempat yang cocok untuk keluarga dengan anak?\"].\n"
            "Contoh untuk 'chat': "
            "[\"Buatkan itinerary 3 hari di Ubud\", \"Rekomendasikan pantai terbaik di Bali\", "
            "\"Apa itu Odalan dan bagaimana pengaruhnya ke wisata?\"]."
        ),
    )

    # ── Itinerary-only fields ─────────────────────────────────────────────

    itinerary_id: Optional[str] = Field(
        default=None,
        description=(
            "UUID itinerary tersimpan di database. "
            "JANGAN diisi oleh Heidi — diisi otomatis oleh sistem backend setelah save. "
            "Set null saat generate itinerary baru."
        ),
    )

    trip_title: Optional[str] = Field(
        default=None,
        description=(
            "Judul singkat itinerary yang catchy, deskriptif, dan memorable. "
            "WAJIB diisi saat response_type='itinerary'. "
            "Format yang disarankan: '[Tema/Nuansa] — [Durasi] di [Area]'. "
            "Contoh: 'Harmoni Ubud — 3 Hari di Jantung Bali', "
            "'Surga Tersembunyi Nusa Penida — Weekend Escape 2 Hari', "
            "'Kuliner & Seni Seminyak — 4 Hari Penuh Rasa'."
        ),
    )

    base_hotel: Optional[BaseHotel] = Field(
        default=None,
        description=(
            "Hotel utama/home base untuk SELURUH durasi perjalanan. "
            "WAJIB diisi saat response_type='itinerary'. "
            "Pilih dari hasil get_nearby_food_and_lodging(category='hotel') — "
            "pilih yang rating terbaik dan lokasi paling strategis. "
            "SATU hotel untuk SEMUA hari — JANGAN ganti hotel antar hari. "
            "Hotel ini TIDAK BOLEH muncul lagi di dalam itinerary_days[N].places."
        ),
    )

    itinerary_days: Optional[List[DailyItinerary]] = Field(
        default=None,
        description=(
            "Daftar jadwal harian lengkap. WAJIB diisi saat response_type='itinerary'. "
            "Panjang array = jumlah hari perjalanan (day=1 s.d. N). "
            "Setiap DailyItinerary WAJIB punya: "
            "minimal 2 attraction, minimal 1 restaurant, "
            "dan semua route_to_next terisi dari calculate_batch_routes. "
            "Urutan array: [hari_ke_1, hari_ke_2, ..., hari_ke_N]."
        ),
    )

    total_budget_idr: Optional[int] = Field(
        default=None,
        description=(
            "Total estimasi biaya seluruh perjalanan dalam Rupiah (IDR), integer. "
            "WAJIB diisi saat response_type='itinerary'. "
            "Harus sama dengan jumlah semua komponen di budget_breakdown. "
            "Contoh: 5750000 untuk total Rp 5.750.000."
        ),
    )

    budget_breakdown: Optional[BudgetBreakdown] = Field(
        default=None,
        description=(
            "Rincian biaya per kategori. WAJIB diisi saat response_type='itinerary'. "
            "Pastikan: accommodation + food + transport + entrance_fee + miscellaneous = total_budget_idr."
        ),
    )

    # ── Recommendation-only fields ────────────────────────────────────────

    recommendations: Optional[List[RecommendationItem]] = Field(
        default=None,
        description=(
            "Daftar tempat rekomendasi. WAJIB diisi saat response_type='recommendation'. "
            "Isi dari hasil get_smart_recommendations, diurutkan dari topsis_score tertinggi. "
            "Minimal 5 item, maksimal 10 item. "
            "JANGAN buat itinerary_days saat mengisi ini."
        ),
    )

    # ── Clarifying-only fields ────────────────────────────────────────────

    clarifying_questions: Optional[List[str]] = Field(
        default=None,
        description=(
            "Daftar pertanyaan klarifikasi. WAJIB diisi saat response_type='clarifying'. "
            "Maksimal 4 pertanyaan, fokus pada info krusial yang belum ada. "
            "Urutkan dari yang paling penting. "
            "Contoh: ['Di area mana di Bali yang ingin kamu kunjungi?', "
            "'Berapa hari rencanamu di Bali?', "
            "'Pergi berdua, bersama keluarga, atau rombongan?', "
            "'Kisaran budget per hari (hemat/menengah/mewah)?']."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ITINERARY LIST — untuk endpoint GET /api/itineraries
# ══════════════════════════════════════════════════════════════════════════════

class ItinerarySummary(BaseModel):
    """Summary ringkas itinerary untuk tampilan list di frontend."""

    id: str
    user_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    days_count: Optional[int] = None
    total_budget_idr: Optional[int] = None
    is_public: bool = False
    share_slug: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_owner: bool = True