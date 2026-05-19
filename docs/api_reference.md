# SobatNavi AI Agent — API Reference v8.0

> Base URL: `http://localhost:8000` (dev) | `https://api.sobatnavi.id` (prod)
> Authentication: **Bearer Token** (Supabase JWT) pada semua endpoint kecuali `GET /health`

---

## Autentikasi

Semua endpoint (kecuali `GET /health`) wajib menyertakan header:

```
Authorization: Bearer <supabase_jwt_token>
```

| HTTP Status | Kondisi |
|-------------|---------|
| `401` | Token tidak ada, tidak valid, atau expired |
| `403` | Token valid tapi tidak punya akses ke resource |

> **Guest mode** tersedia di `POST /api/chat` — request tanpa token diterima, tetapi history tidak disimpan ke database.

---

## Ringkasan Fitur Baru v8.0

| Fitur | Deskripsi |
|-------|-----------|
| **POI Budget Calculator** | Backend menghitung jumlah atraksi per hari secara otomatis berdasarkan `pace` dan durasi perjalanan, sebelum AI dipanggil |
| **Meal Injection** | Backend menyisipkan restoran (sarapan/makan siang/makan malam) ke setiap hari setelah AI selesai — AI tidak lagi bertanggung jawab atas slot makan |
| **Hotel Guarantee** | Jika AI tidak menghasilkan `base_hotel` yang valid, backend otomatis mencari hotel dari database |
| **Edge Case Handling** | Trip setengah hari, trip 5+ hari, user minta jumlah POI eksplisit, restoran tidak ditemukan, database kosong |

---

## 1. AI Chat — `POST /api/chat`

Endpoint utama untuk berinteraksi dengan Heidi. Mendukung pembuatan itinerary baru, edit itinerary via chat, rekomendasi tempat, dan percakapan biasa.

### Request Body

```json
{
  "message": "Buatkan itinerary Bali 3 hari santai mulai 2026-07-01",
  "mode": "general",
  "session_id": null,
  "history": [
    { "role": "user", "parts": "Hai Heidi" },
    { "role": "model", "parts": "{...json response sebelumnya...}" }
  ],
  "current_itinerary": null,
  "itinerary_id": null
}
```

| Field | Tipe | Wajib | Deskripsi |
|-------|------|-------|-----------|
| `message` | `string` | Yes | Pesan user |
| `mode` | `"general"` \| `"deep_research"` | No | Default `"general"`. `deep_research` akan meminta klarifikasi sebelum membuat itinerary |
| `session_id` | `string (UUID)` | No | ID sesi chat untuk melanjutkan percakapan sebelumnya (hanya efektif jika terautentikasi) |
| `history` | `array` | No | Riwayat percakapan untuk guest mode. Diabaikan jika terautentikasi (diambil dari DB) |
| `current_itinerary` | `object` | No | Data itinerary aktif untuk mode edit via chat. Wajib diisi bersamaan dengan `itinerary_id` |
| `itinerary_id` | `string (UUID)` | No | UUID itinerary yang sedang diedit |

#### Deteksi Otomatis dari `message`

Backend v8.0 mem-parsing `message` sebelum memanggil AI untuk mendeteksi parameter berikut secara otomatis:

| Kata kunci terdeteksi | Efek |
|-----------------------|------|
| `"santai"`, `"rileks"`, `"slow"` | `pace = santai` → 2–3 atraksi/hari, 1 sesi makan |
| `"padat"`, `"penuh"`, `"banyak tempat"` | `pace = padat` → 4–5 atraksi/hari, 2 sesi makan |
| `"setengah hari"`, `"half day"`, `"beberapa jam"` | Trip setengah hari → maks 2 atraksi + 1 makan siang |
| `"3 tempat"`, `"5 wisata"`, `"kunjungi 4"` | Override jumlah atraksi secara eksplisit (clamp: 1–8) |

### `response_type` Values

| Nilai | Field yang terisi |
|-------|------------------|
| `"chat"` | `message_to_user`, `suggested_replies` |
| `"recommendation"` | `recommendations`, `message_to_user`, `suggested_replies` |
| `"clarifying"` | `message_to_user`, `clarifying_questions`, `suggested_replies` |
| `"itinerary"` | `itinerary_id`, `trip_title`, `base_hotel`, `itinerary_days`, `total_budget_idr`, `budget_breakdown`, `message_to_user`, `suggested_replies` |

### Response Body — `FinalAIResponse`

```json
{
  "response_type": "itinerary",
  "message_to_user": "# ✈️ Santai di Bali — 3 Hari\n\n...",
  "itinerary_id": "uuid-auto-filled",
  "trip_title": "Santai di Bali — 3 Hari",
  "base_hotel": {
    "place_id": "hotel-place-id",
    "name": "Kuta Beach Hotel",
    "category": "hotel",
    "latitude": -8.7176,
    "longitude": 115.1686,
    "district": "Badung",
    "rating": 4.3,
    "image_url": "https://...",
    "description": "Hotel tepi pantai dengan fasilitas lengkap.",
    "price_level": 2,
    "estimated_cost_per_night_idr": 500000
  },
  "itinerary_days": [
    {
      "day": 1,
      "day_date": "2026-07-01",
      "theme": "Pura & Pantai Ikonik",
      "day_total_distance_km": 45.2,
      "day_total_travel_time_mins": 95,
      "day_full_polyline": [
        {"lat": -8.71, "lng": 115.16},
        {"lat": -8.62, "lng": 115.08}
      ],
      "places": [
        {
          "place_id": "ChIJxxx",
          "name": "Tanah Lot",
          "category": "attraction",
          "latitude": -8.6215,
          "longitude": 115.0866,
          "district": "Tabanan",
          "description": "Pura ikonik di atas batu karang...",
          "rating": 4.7,
          "image_url": "https://...",
          "tags": ["budaya", "sunset", "pura"],
          "visit_time": "09:00",
          "visit_duration_mins": 90,
          "estimated_cost_idr": 60000,
          "tips": "Datang sebelum jam 10 pagi untuk menghindari keramaian.",
          "topsis_score": 0.8731,
          "route_to_next": {
            "distance_km": 12.5,
            "travel_time_mins": 28,
            "traffic_delay_mins": 5,
            "polyline": [
              {"lat": -8.6215, "lng": 115.0866},
              {"lat": -8.6350, "lng": 115.1100}
            ],
            "status": "OK"
          }
        },
        {
          "place_id": "resto-place-id",
          "name": "Warung Makan Lokal Tabanan",
          "category": "restaurant",
          "latitude": -8.6300,
          "longitude": 115.0900,
          "district": "Tabanan",
          "description": "Restoran dengan masakan Bali autentik.",
          "rating": 4.2,
          "image_url": null,
          "tags": ["makan siang", "kuliner", "restoran"],
          "visit_time": "12:30",
          "visit_duration_mins": 60,
          "estimated_cost_idr": 75000,
          "tips": "Nikmati makan siang di sini. Cocok untuk pengunjung yang ingin cita rasa lokal.",
          "topsis_score": null,
          "route_to_next": { "...": "..." }
        }
      ]
    }
  ],
  "total_budget_idr": 3500000,
  "budget_breakdown": {
    "accommodation": 1500000,
    "food": 800000,
    "transport": 500000,
    "tickets": 700000
  },
  "recommendations": null,
  "suggested_replies": ["Bisa tambah 1 hari?", "Cari hotel lebih murah", "Ganti ke tema budaya"],
  "clarifying_questions": null,
  "session_id": "uuid-sesi-aktif"
}
```

#### Catatan Penting tentang `places` di v8.0

- **Restoran disisipkan backend** — array `places` di setiap hari sudah mengandung restoran secara otomatis. Frontend tidak perlu memperlakukan restoran secara berbeda dari atraksi.
- **Posisi restoran dalam array:**
  - `sarapan` → index 0 (paling awal)
  - `makan_siang` → tengah array (setelah ~50% atraksi)
  - `makan_malam` → index terakhir
- **`base_hotel` dijamin selalu ada** — jika AI gagal mengisinya, backend otomatis mengisi dari database.

#### Perubahan field dari v7 → v8

| Field lama (v7) | Field baru (v8) | Catatan |
|-----------------|-----------------|---------|
| `base_hotel.poi_id` | `base_hotel.place_id` | Konsistensi dengan tabel DB |
| `base_hotel.estimated_cost_idr` | `base_hotel.estimated_cost_per_night_idr` | Lebih eksplisit |
| `places[].poi_id` | `places[].place_id` | Konsistensi dengan tabel DB |
| `places[].duration_mins` | `places[].visit_duration_mins` | Konsistensi penamaan |
| `places[].price_level` | dihapus | Diambil dari metadata internal, tidak dikirim ke frontend |
| `places[].review_count` | dihapus | Digabung ke dalam `topsis_score` |
| `itinerary_days[].day_number` | `itinerary_days[].day` | Sesuai schema Pydantic aktual |
| — | `places[].tips` | **Baru** — wajib diisi AI, tidak boleh null |
| — | `session_id` | **Baru** — dikembalikan jika user terautentikasi |

**Error Responses:**

| Status | Kondisi |
|--------|---------|
| `401` | Token tidak valid |
| `500` | AI Processing Error atau database error |
| `503` | Server belum dikonfigurasi (OPENAI_API_KEY kosong) |
| `504` | AI timeout — tool calling loop melebihi 20 iterasi |

---

## 2. List Itineraries — `GET /api/itineraries`

Hanya dapat diakses oleh user terautentikasi.

### Query Parameters

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `include_public` | `false` | Sertakan itinerary publik milik user lain |
| `limit` | `20` | Maks hasil (1–100) |
| `offset` | `0` | Offset paginasi |

### Response — Array of `ItinerarySummary`

```json
[
  {
    "id": "uuid",
    "title": "Santai di Bali — 3 Hari",
    "days_count": 3,
    "total_budget_idr": 3500000,
    "is_public": false,
    "is_owner": true,
    "cover_image_url": "https://...",
    "district_tags": ["Tabanan", "Badung"],
    "created_at": "2026-07-01T10:00:00Z",
    "updated_at": "2026-07-01T10:00:00Z"
  }
]
```

> Item dengan `is_owner: false` = itinerary publik orang lain, hanya bisa di-copy (tidak bisa di-edit atau dihapus).

---

## 3. Get Itinerary — `GET /api/itinerary/{itinerary_id}`

Akses diizinkan jika: (1) user adalah pemilik, atau (2) itinerary bersifat publik.

| Status | Kondisi |
|--------|---------|
| `200` | Data lengkap itinerary (`FinalAIResponse` object) |
| `404` | Tidak ditemukan atau tidak punya akses |

---

## 4. Update Itinerary — `PUT /api/itinerary/{itinerary_id}`

Edit manual dari UI. Hanya pemilik yang bisa mengupdate.

### Request Body

```json
{
  "title": "Judul Baru (opsional)",
  "itinerary_data": { "...FinalAIResponse object..." },
  "total_budget_idr": 4000000,
  "is_public": false
}
```

| Field | Tipe | Wajib | Deskripsi |
|-------|------|-------|-----------|
| `itinerary_data` | `object` | Yes | Seluruh objek itinerary yang sudah dimodifikasi |
| `title` | `string` | No | Judul baru |
| `total_budget_idr` | `integer` | No | Total budget baru dalam IDR |
| `is_public` | `boolean` | No | Toggle visibilitas |

| Status | Kondisi |
|--------|---------|
| `200` | Data ter-update |
| `404` | Tidak ditemukan atau bukan pemilik |
| `422` | Validasi request body gagal |

---

## 5. Toggle Visibility — `PATCH /api/itinerary/{itinerary_id}/visibility`

### Request Body

```json
{ "is_public": true }
```

### Response

```json
{ "status": "updated", "itinerary_id": "uuid", "is_public": true }
```

---

## 6. Copy Public Itinerary — `POST /api/itinerary/{itinerary_id}/copy`

Salin itinerary publik milik orang lain ke akun sendiri. Salinan bersifat privat dan bebas diedit. Itinerary asli tidak terpengaruh.

### Response

```json
{
  "status": "copied",
  "new_itinerary_id": "uuid-baru",
  "title": "Salinan: Santai di Bali — 3 Hari"
}
```

| Status | Kondisi |
|--------|---------|
| `200` | Berhasil disalin |
| `404` | Itinerary tidak ada atau bukan publik |

---

## 7. Delete Itinerary — `DELETE /api/itinerary/{itinerary_id}`

Hapus permanen. Hanya pemilik yang bisa menghapus.

### Response

```json
{ "status": "deleted", "itinerary_id": "uuid" }
```

| Status | Kondisi |
|--------|---------|
| `200` | Berhasil dihapus |
| `404` | Tidak ditemukan atau bukan pemilik |

---

## 8. Place Search — `GET /api/place/search`

Pencarian tempat berdasarkan nama (keyword match). Memerlukan autentikasi.

### Query Parameters

| Parameter | Wajib | Deskripsi |
|-----------|-------|-----------|
| `query` | Yes | Nama tempat yang dicari |
| `category` | No | `"attraction"` \| `"hotel"` \| `"restaurant"` (default: `"attraction"`) |

### Response

```json
{ "results": [ "...PlaceItem[]..." ], "count": 3 }
```

---

## 9. Place Recommendations — `GET /api/place/recommendations`

Semantic Search + TOPSIS ranking tanpa membuat jadwal. Memerlukan autentikasi.

### Query Parameters

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `query` | required | Tema/kata kunci (misal: `"pantai sunset"`, `"budaya ubud"`) |
| `category` | `"poi"` | `"poi"` \| `"hotel"` \| `"restaurant"` |
| `limit` | `10` | Jumlah hasil (1–30) |

### Response

```json
{ "results": [ "...PlaceItem[]..." ], "count": 10, "query": "pantai sunset" }
```

---

## 10. Health Check — `GET /health`

Tidak memerlukan autentikasi.

### Response

```json
{
  "status": "healthy",
  "version": "8.0",
  "service": "SobatNavi AI Agent",
  "features": [
    "poi_budget_calculator",
    "meal_injection",
    "hotel_guarantee",
    "odalan_safety_check",
    "tomtom_routing"
  ]
}
```

---

## Error Code Summary

| HTTP Status | Kondisi |
|-------------|---------|
| `200` | Sukses |
| `401` | Token tidak ada / tidak valid / expired |
| `403` | Tidak punya akses resource |
| `404` | Resource tidak ada atau tidak bisa diakses |
| `422` | Validasi request body gagal |
| `500` | AI processing error atau database error |
| `503` | Server belum dikonfigurasi (OPENAI_API_KEY kosong) |
| `504` | AI timeout — loop melebihi batas maksimal |

---

## Ownership & Sharing Rules

```
User A membuat Itinerary X (is_public=false)
  → Hanya User A: GET / PUT / DELETE / toggle visibility

User A publish Itinerary X (is_public=true)
  → User A + User B: bisa GET
  → User B: bisa POST /copy  → Itinerary Y milik User B (privat, bebas edit)
  → User B: TIDAK BISA PUT/DELETE Itinerary X
  → Hanya User A: bisa toggle kembali ke privat
```

---

## POI Budget Reference (v8.0)

Backend otomatis menghitung anggaran POI sebelum AI dipanggil. Hasilnya dikirim ke AI sebagai konteks wajib.

| Pace | Atraksi/hari | Sesi makan/hari | Total places/hari |
|------|-------------|-----------------|-------------------|
| `santai` | 2–3 | 1 (makan siang) | 3–4 |
| `normal` (default) | 3–4 | 2 (siang + malam) | 5–6 |
| `padat` | 4–5 | 2 (siang + malam) | 6–7 |
| Setengah hari | 2 | 1 (makan siang) | 3 |
| Custom (user sebut angka) | sesuai permintaan (1–8) | 1 jika < 4 atraksi, 2 jika ≥ 4 | sesuai + sesi makan |

> Trip 5+ hari: jumlah atraksi ideal dikurangi 1 secara otomatis untuk menghindari kelelahan.

---

## Meal Slot Reference (v8.0)

Restoran disisipkan backend setelah AI selesai. Frontend menerimanya sebagai `PlaceItem` biasa dengan `category: "restaurant"`.

| Slot | `visit_time` | `visit_duration_mins` | `estimated_cost_idr` |
|------|--------------|-----------------------|----------------------|
| Sarapan | `08:00` | 45 | 50.000 |
| Makan Siang | `12:30` | 60 | 75.000 |
| Makan Malam | `19:00` | 75 | 100.000 |

**Sumber data restoran (prioritas):**
1. `search_amenities_nearby` — radius 5km dari sentroid atraksi hari itu
2. Semantic search fallback — query `"restoran {district} Bali"` jika nearby tidak cukup
3. Jika benar-benar tidak ada → slot makan dilewati (warning di server log)

---

## TOPSIS Internal Weights

| Kategori | Fitur (Bobot) |
|----------|--------------|
| **POI** | rating(30%), popularity(20%), price_value(15%), strategic_score(20%), visual_interest(15%) |
| **Hotel** | rating(35%), comfort_index(30%), amenity_density(20%), accessibility_index(15%) |
| **Restoran** | rating(35%), menu_variety_index(25%), ambience_score(25%), payment_modern_index(15%) |

> Fitur dibaca dari `metadata.topsis_features` di database. Jika tidak ada, sistem melakukan fallback otomatis dari raw metadata.

> Backend selalu mem-fetch **3× lebih banyak** dari jumlah POI yang dibutuhkan ke database (min 30, max 80) sebelum TOPSIS + DBSCAN memilih yang terbaik. Ini memastikan kualitas seleksi tetap baik meski kuota akhir kecil.