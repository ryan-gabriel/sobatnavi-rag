# SobatNavi AI Agent — API Reference v7.0

> Base URL: `http://localhost:8000` (dev) | `https://api.sobatnavi.id` (prod)  
> Authentication: **Bearer Token** (Supabase JWT) pada semua endpoint kecuali `/health`

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

---

## 1. AI Chat — `POST /api/chat`

### Request Body

```json
{
  "message": "Buatkan itinerary Bali 3 hari mulai 2026-07-01",
  "mode": "general",
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
| `mode` | `"general"/"deep_research"` | No | Default `"general"` |
| `history` | `array` | No | Riwayat percakapan |
| `current_itinerary` | `object` | No | Data itinerary aktif untuk mode edit |
| `itinerary_id` | `string (UUID)` | No | Wajib bersama `current_itinerary` |

### `response_type` Values

| Nilai | Field yang terisi |
|-------|------------------|
| `"chat"` | `message_to_user`, `suggested_replies` |
| `"recommendation"` | `recommendations`, `message_to_user` |
| `"clarifying"` | `message_to_user`, `clarifying_questions` |
| `"itinerary"` | `itinerary_id`, `trip_title`, `base_hotel`, `itinerary_days`, `total_budget_idr` |

### Response Body — `FinalAIResponse`

```json
{
  "response_type": "itinerary",
  "message_to_user": "Hei! Ini jadwal Bali 3 hari kamu",
  "itinerary_id": "uuid-auto-filled",
  "trip_title": "Petualangan Pantai Bali 3 Hari",
  "base_hotel": {
    "poi_id": "hotel-uuid",
    "name": "Kuta Beach Hotel",
    "category": "hotel",
    "latitude": -8.7176, "longitude": 115.1686,
    "district": "Badung",
    "rating": 4.3,
    "image_url": "https://...",
    "estimated_cost_idr": 500000
  },
  "itinerary_days": [
    {
      "day_number": 1,
      "day_date": "2026-07-01",
      "theme": "Pantai & Pura Ikonik",
      "day_total_distance_km": 45.2,
      "day_total_travel_time_mins": 95,
      "day_full_polyline": [{"lat": -8.71, "lng": 115.16}, {"lat": -8.62, "lng": 115.08}],
      "places": [
        {
          "poi_id": "ChIJ...",
          "name": "Tanah Lot",
          "category": "attraction",
          "latitude": -8.6215, "longitude": 115.0866,
          "district": "Tabanan",
          "description": "Pura ikonik di atas batu karang...",
          "rating": 4.7, "review_count": 87420,
          "price_level": "PRICE_LEVEL_INEXPENSIVE",
          "image_url": "https://...",
          "tags": ["budaya", "sunset", "pura"],
          "visit_time": "17:00",
          "duration_mins": 90,
          "estimated_cost_idr": 60000,
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
  "suggested_replies": ["Bisa tambah 1 hari?", "Cari hotel lebih murah"],
  "clarifying_questions": null
}
```

**Error Responses:**

| Status | Kondisi |
|--------|---------|
| `401` | Token tidak valid |
| `500` | AI Processing Error |

---

## 2. List Itineraries — `GET /api/itineraries`

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
    "title": "Petualangan Bali 3 Hari",
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

> Item dengan `is_owner: false` = itinerary publik orang lain, hanya bisa di-copy.

---

## 3. Get Itinerary — `GET /api/itinerary/{itinerary_id}`

Akses diizinkan jika pemilik ATAU itinerary bersifat publik.

| Status | Kondisi |
|--------|---------|
| `200` | Data lengkap itinerary |
| `404` | Tidak ditemukan atau tidak punya akses |

---

## 4. Update Itinerary — `PUT /api/itinerary/{itinerary_id}`

Hanya pemilik. Edit manual dari UI.

### Request Body

```json
{
  "title": "Judul Baru (opsional)",
  "itinerary_data": { ...FinalAIResponse object... },
  "total_budget_idr": 4000000,
  "is_public": false
}
```

| Status | Kondisi |
|--------|---------|
| `200` | Data ter-update |
| `404` | Tidak ditemukan atau bukan pemilik |
| `422` | Validasi request gagal |

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

Salin itinerary publik milik orang lain. Salinan bersifat privat dan bebas diedit.

### Response
```json
{
  "status": "copied",
  "new_itinerary_id": "uuid-baru",
  "title": "Salinan: Petualangan Bali 3 Hari"
}
```

| Status | Kondisi |
|--------|---------|
| `200` | Berhasil disalin |
| `404` | Itinerary tidak ada atau bukan publik |

---

## 7. Delete Itinerary — `DELETE /api/itinerary/{itinerary_id}`

Hanya pemilik. Hapus permanen.

```json
{ "status": "deleted", "itinerary_id": "uuid" }
```

---

## 8. Place Search — `GET /api/place/search`

### Query Parameters

| Parameter | Wajib | Deskripsi |
|-----------|-------|-----------|
| `query` | Yes | Nama tempat |
| `category` | No | `"attraction"` \| `"hotel"` \| `"restaurant"` (default: `"attraction"`) |

### Response
```json
{ "results": [ ...PlaceItem[]... ], "count": 3 }
```

---

## 9. Place Recommendations — `GET /api/place/recommendations`

Semantic Search + TOPSIS ranking tanpa membuat jadwal.

### Query Parameters

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `query` | required | Tema/kata kunci |
| `category` | `"poi"` | `"poi"` \| `"hotel"` \| `"restaurant"` |
| `limit` | `10` | 1–30 |

```json
{ "results": [ ...PlaceItem[]... ], "count": 10, "query": "pantai sunset" }
```

---

## 10. Health Check — `GET /health`

```json
{ "status": "healthy", "version": "7.0", "service": "SobatNavi AI Agent" }
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

## TOPSIS Internal Weights

| Kategori | Fitur (Bobot) |
|----------|--------------|
| **POI** | rating(30%), popularity(20%), price_value(15%), strategic_score(20%), visual_interest(15%) |
| **Hotel** | rating(35%), comfort_index(30%), amenity_density(20%), accessibility_index(15%) |
| **Restoran** | rating(35%), menu_variety_index(25%), ambience_score(25%), payment_modern_index(15%) |

> Fitur dibaca dari `metadata.topsis_features` di database. Jika tidak ada, sistem melakukan fallback otomatis dari raw metadata (panjang array amenities, dsb).
