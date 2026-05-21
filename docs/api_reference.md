# SobatNavi AI Agent — API Reference v8.1

> **Base URL (dev):** `http://localhost:8000`
> **Base URL (prod):** `https://api.sobatnavi.id`
> **Authentication:** Bearer Token (Supabase JWT) on all endpoints except `GET /health`

---

## Authentication

Include the following header on every authenticated request:

```
Authorization: Bearer <supabase_jwt_token>
```

| HTTP Status | Condition |
|-------------|-----------|
| `401` | Token missing, invalid, or expired |
| `403` | Token valid but no access to resource |

> **Guest mode** is available on `POST /api/chat` — requests without a token are accepted, but history is not saved to the database and `session_id` will not be returned.

---

## Rate Limits

| Endpoint Group | Limit |
|----------------|-------|
| `POST /api/chat` | **10 requests / 60 s** |
| `POST/PUT/PATCH/DELETE` itinerary | **30 requests / 60 s** |
| `GET` itinerary list / detail | **60 requests / 60 s** |
| `GET /api/place/*` | **30 requests / 60 s** |
| `GET /health` | **120 requests / 60 s** |
| Global fallback (all other routes) | **200 requests / 60 s** |

Rate-limit key: authenticated `user_id` → `X-Forwarded-For` IP → direct remote IP.

When exceeded:

```json
{ "error": "Rate limit exceeded: 10 per 1 minute" }
```
HTTP `429 Too Many Requests`

---

## Endpoint Index

| # | Method | Path | Auth | Description |
|---|--------|------|------|-------------|
| 1 | `POST` | `/api/chat` | Optional | Main AI chat — itinerary, recommendation, clarifying, chat |
| 2 | `GET` | `/api/itineraries` | Required | List user's itineraries |
| 3 | `GET` | `/api/itinerary/{id}` | Required | Get one itinerary by ID |
| 4 | `PUT` | `/api/itinerary/{id}` | Required | Manual update itinerary |
| 5 | `PATCH` | `/api/itinerary/{id}/visibility` | Required | Toggle public/private |
| 6 | `POST` | `/api/itinerary/{id}/copy` | Required | Copy a public itinerary |
| 7 | `DELETE` | `/api/itinerary/{id}` | Required | Delete itinerary |
| 8 | `GET` | `/api/place/search` | Required | Keyword search for a place |
| 9 | `GET` | `/api/place/recommendations` | Required | Semantic search + TOPSIS ranking |
| 10 | `GET` | `/health` | None | Health check |

---

## 1. AI Chat — `POST /api/chat`

Main endpoint for interacting with Heidi. Supports itinerary creation, editing via chat, place recommendations, and general conversation.

### Request Body

```json
{
  "message": "Buatkan itinerary Bali 3 hari santai mulai 2026-07-01",
  "mode": "general",
  "budget_preference": "moderate",
  "session_id": null,
  "history": [
    { "role": "user", "parts": "Hai Heidi" },
    { "role": "model", "parts": "{...json AI response...}" }
  ],
  "current_itinerary": null,
  "itinerary_id": null
}
```

### Request Fields

| Field | Type | Required | Possible Values | Default | Description |
|-------|------|----------|-----------------|---------|-------------|
| `message` | `string` | **Yes** | Any string, max 4000 chars | — | User message to Heidi |
| `mode` | `string` | No | `"general"` \| `"deep_research"` | `"general"` | `general` = immediately processes request. `deep_research` = asks for all 5 variables (location, date, duration, budget, companion, pace) before building itinerary |
| `budget_preference` | `string` | No | `"budget"` \| `"moderate"` \| `"luxury"` | `"moderate"` | Controls TOPSIS weighting, hotel query, and restaurant search. See mapping below |
| `session_id` | `string (UUID)` | No | Any UUID | `null` | Chat session ID to continue a previous conversation. Only effective when authenticated |
| `history` | `array` | No | Array of `{role, parts}` or `{role, content}` objects. Max 50 items | `[]` | Conversation history for guest mode. Ignored when authenticated (loaded from DB) |
| `current_itinerary` | `object` | No | Full `FinalAIResponse` object | `null` | Active itinerary data when editing via chat. Fill together with `itinerary_id` |
| `itinerary_id` | `string (UUID)` | No | Any valid UUID | `null` | ID of the itinerary being edited. Required if `current_itinerary` is set |

#### `budget_preference` Value Mapping

| Request value | Internal `preference_mode` | Effect |
|---------------|---------------------------|--------|
| `"budget"` | `"budget"` | TOPSIS boosts `price_value` weight to 35%. Semantic hotel search uses `"budget guesthouse hostel bali murah"`. Restaurant fallback searches for `"warung makan murah … harga terjangkau"` |
| `"moderate"` | `"standard"` | Default balanced TOPSIS weights. Neutral hotel/restaurant search |
| `"luxury"` | `"luxury"` | TOPSIS flips `price_value` impact to cost (higher price = better). Hotel search uses `"luxury resort hotel bali"`. Restaurant fallback searches for `"restoran fine dining mewah … premium"` |

#### Auto-detection from `message` Content

Backend parses `message` before calling AI to set pace and POI budget:

| Keyword(s) detected | Effect |
|---------------------|--------|
| `"santai"`, `"rileks"`, `"slow"` | `pace = santai` → 2–3 attractions/day, 1 meal |
| `"padat"`, `"penuh"`, `"banyak tempat"`, `"maksimal"` | `pace = padat` → 4–5 attractions/day, 2 meals |
| `"setengah hari"`, `"half day"`, `"beberapa jam"`, `"sore aja"`, `"pagi aja"` | Half-day trip → max 2 attractions + 1 lunch |
| `"3 tempat"`, `"5 wisata"`, `"kunjungi 4"` | Explicit override of attraction count (clamped 1–8) |
| `"3 hari"`, `"2 malam"` | Sets `num_days_hint` for POI budget calculation |

---

### Response Body — `FinalAIResponse`

The response structure is always the same shape. Fields are populated depending on `response_type`.

#### `response_type` Values

| Value | Populated fields |
|-------|-----------------|
| `"chat"` | `message_to_user`, `suggested_replies` |
| `"recommendation"` | `message_to_user`, `recommendations`, `suggested_replies` |
| `"clarifying"` | `message_to_user`, `clarifying_questions`, `suggested_replies` |
| `"itinerary"` | `message_to_user`, `trip_title`, `base_hotel`, `itinerary_days`, `total_budget_idr`, `budget_breakdown`, `suggested_replies`, `itinerary_id` (if authenticated), `session_id` (if authenticated) |

---

#### Full `"itinerary"` Response Example

```json
{
  "response_type": "itinerary",
  "message_to_user": "# ✈️ Harmoni Ubud — 3 Hari di Jantung Bali\n\n...(Markdown storytelling min. 300 kata)...",
  "suggested_replies": [
    "Tambah 1 hari ke Nusa Penida",
    "Cari hotel lebih murah",
    "Ganti ke tema kuliner"
  ],
  "session_id": "a1b2c3d4-uuid",
  "itinerary_id": "f4e5d6c7-uuid",
  "trip_title": "Harmoni Ubud — 3 Hari di Jantung Bali",
  "base_hotel": {
    "place_id": "ChIJxxxHotelId",
    "name": "Komaneka at Bisma",
    "description": "Boutique resort mewah yang menggantung di tebing Ubud dengan pemandangan hutan dan Sungai Wos yang dramatis.",
    "latitude": -8.5069,
    "longitude": 115.2625,
    "district": "Ubud",
    "image_url": "https://example.com/hotel.jpg",
    "rating": 4.8,
    "user_rating_count": 3200,
    "price_per_night_idr": 1500000,
    "amenities": ["kolam renang", "wifi gratis", "sarapan termasuk", "spa", "restoran"],
    "check_in_time": "14:00",
    "check_out_time": "12:00"
  },
  "itinerary_days": [
    {
      "day": 1,
      "date": "2026-07-01",
      "theme": "Spiritual & Budaya Ubud",
      "odalan_warning": null,
      "weather_note": "Prakiraan cerah sepanjang hari — ideal untuk outdoor.",
      "day_total_distance_km": 38.4,
      "day_total_travel_time_mins": 82,
      "day_full_polyline": [
        {"lat": -8.5069, "lng": 115.2625},
        {"lat": -8.4178, "lng": 115.3317}
      ],
      "route_from_hotel": {
        "distance_km": 5.2,
        "travel_time_mins": 14,
        "traffic_delay_mins": 2,
        "polyline": [{"lat": -8.5069, "lng": 115.2625}, {"lat": -8.4178, "lng": 115.3317}],
        "status": "OK"
      },
      "places": [
        {
          "place_id": "ChIJxxxAttractionId",
          "poi_id": "42",
          "name": "Pura Tirta Empul",
          "category": "attraction",
          "description": "Pura Hindu paling sakral di Bali dengan mata air suci untuk ritual melukat...",
          "latitude": -8.4178,
          "longitude": 115.3317,
          "district": "Gianyar",
          "image_url": "https://example.com/tirta-empul.jpg",
          "rating": 4.7,
          "user_rating_count": 18500,
          "estimated_cost_idr": 50000,
          "tags": ["pura", "spiritual", "melukat", "budaya"],
          "visit_duration_mins": 90,
          "visit_time": "08:00",
          "opening_hours_note": "Buka setiap hari 07:00–18:00",
          "tips": "Bawa/sewa kain sarung (Rp 10.000) dan siapkan pakaian ganti jika ingin ikut melukat.",
          "topsis_score": 0.9124,
          "route_to_next": {
            "distance_km": 12.5,
            "travel_time_mins": 28,
            "traffic_delay_mins": 5,
            "polyline": [{"lat": -8.4178, "lng": 115.3317}, {"lat": -8.5069, "lng": 115.2625}],
            "status": "OK"
          }
        },
        {
          "place_id": "ChIJxxxRestaurantId",
          "poi_id": null,
          "name": "Warung Tepi Sawah",
          "category": "restaurant",
          "description": "Restoran dengan pemandangan sawah langsung, menu Bali dan Western tersedia.",
          "latitude": -8.5100,
          "longitude": 115.2620,
          "district": "Ubud",
          "image_url": null,
          "rating": 4.3,
          "user_rating_count": 980,
          "estimated_cost_idr": 75000,
          "tags": ["makan siang", "kuliner", "restoran"],
          "visit_duration_mins": 60,
          "visit_time": "12:30",
          "opening_hours_note": null,
          "tips": "Nikmati makan siang di sini. Cocok untuk pengunjung yang ingin cita rasa lokal.",
          "topsis_score": null,
          "route_to_next": {
            "distance_km": 3.1,
            "travel_time_mins": 9,
            "traffic_delay_mins": 0,
            "polyline": [{"lat": -8.5100, "lng": 115.2620}, {"lat": -8.5200, "lng": 115.2500}],
            "status": "OK"
          }
        }
      ]
    }
  ],
  "total_budget_idr": 5750000,
  "budget_breakdown": {
    "accommodation_idr": 4500000,
    "food_idr": 600000,
    "transport_idr": 300000,
    "entrance_fee_idr": 200000,
    "miscellaneous_idr": 150000
  },
  "recommendations": null,
  "clarifying_questions": null
}
```

---

#### `"chat"` Response Example

```json
{
  "response_type": "chat",
  "message_to_user": "Halo! 👋 Aku **Heidi**, asisten perjalanan AI spesialis Bali dari **SobatNavi**. 🌴\n\nAku siap bantu kamu:\n- 🗺️ Membuat itinerary wisata Bali yang personal\n- 🏖️ Rekomendasi tempat terbaik\n- ℹ️ Info Odalan & kondisi jalan real-time\n\nMau ke mana di Bali? 😊",
  "suggested_replies": [
    "Buatkan itinerary 3 hari di Ubud",
    "Rekomendasikan pantai terbaik di Bali",
    "Apa itu Odalan dan bagaimana pengaruhnya?"
  ],
  "session_id": null,
  "itinerary_id": null,
  "trip_title": null,
  "base_hotel": null,
  "itinerary_days": null,
  "total_budget_idr": null,
  "budget_breakdown": null,
  "recommendations": null,
  "clarifying_questions": null
}
```

---

#### `"recommendation"` Response Example

```json
{
  "response_type": "recommendation",
  "message_to_user": "Ini rekomendasiku! 🗺️\n\n## 🏖️ Pantai Terbaik\n- **Pantai Pandawa** — Tersembunyi di balik tebing kapur, pasir putih bersih.\n- **Nusa Dua** — Ombak tenang, cocok untuk keluarga.\n\n## 🛕 Wisata Budaya\n- **Pura Tanah Lot** — Paling dramatis saat matahari terbenam.",
  "suggested_replies": [
    "Buatkan itinerary dari rekomendasi ini",
    "Rekomendasikan restoran di area Kuta",
    "Tempat yang cocok untuk anak-anak?"
  ],
  "session_id": null,
  "itinerary_id": null,
  "trip_title": null,
  "base_hotel": null,
  "itinerary_days": null,
  "total_budget_idr": null,
  "budget_breakdown": null,
  "recommendations": [
    {
      "place_id": "ChIJxxxPantaiPandawa",
      "poi_id": "17",
      "name": "Pantai Pandawa",
      "category": "attraction",
      "description": "Pantai tersembunyi di balik tebing kapur yang ikonik, pasir putih bersih dan air jernih.",
      "district": "Badung",
      "image_url": "https://example.com/pandawa.jpg",
      "rating": 4.6,
      "latitude": -8.8478,
      "longitude": 115.2105,
      "tags": ["pantai", "sunset", "foto", "keluarga"],
      "estimated_cost_idr": 15000,
      "topsis_score": 0.8732
    }
  ],
  "clarifying_questions": null
}
```

---

#### `"clarifying"` Response Example (deep_research mode)

```json
{
  "response_type": "clarifying",
  "message_to_user": "Hampir siap! 😊 Aku butuh beberapa info dulu sebelum membuat itinerary terbaik untukmu:\n\n- 📍 Area mana di Bali yang ingin kamu kunjungi?\n- 📅 Tanggal berapa kamu berangkat?\n- 👥 Pergi sendiri, berdua, atau bersama keluarga/rombongan?\n- 💰 Kisaran budget per hari (hemat / menengah / mewah)?",
  "suggested_replies": [
    "Ubud, 3 hari, berdua, budget menengah",
    "Kuta & Seminyak, mulai besok",
    "Nusa Penida, 2 hari, rombongan"
  ],
  "session_id": null,
  "itinerary_id": null,
  "trip_title": null,
  "base_hotel": null,
  "itinerary_days": null,
  "total_budget_idr": null,
  "budget_breakdown": null,
  "recommendations": null,
  "clarifying_questions": [
    "Di area mana di Bali yang ingin kamu kunjungi?",
    "Berapa hari rencanamu di Bali?",
    "Pergi berdua, bersama keluarga, atau rombongan?",
    "Kisaran budget per hari (hemat/menengah/mewah)?"
  ]
}
```

---

### Complete `FinalAIResponse` Field Reference

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `response_type` | `"chat"` \| `"clarifying"` \| `"recommendation"` \| `"itinerary"` | No | Response type — determines which other fields are populated |
| `message_to_user` | `string` (Markdown) | No | Always populated. Rendered as Markdown in frontend UI |
| `suggested_replies` | `string[]` (exactly 3 items) | No | Always populated. 3 suggested follow-up actions |
| `session_id` | `string (UUID)` \| `null` | Yes | Returned only if authenticated. Use for next request |
| `itinerary_id` | `string (UUID)` \| `null` | Yes | Populated after itinerary auto-save (authenticated only) |
| `trip_title` | `string` \| `null` | Yes | Only for `itinerary` type |
| `base_hotel` | `BaseHotel` \| `null` | Yes | Only for `itinerary` type. Always guaranteed by backend |
| `itinerary_days` | `DailyItinerary[]` \| `null` | Yes | Only for `itinerary` type |
| `total_budget_idr` | `integer` \| `null` | Yes | Total trip cost in IDR. Only for `itinerary` type |
| `budget_breakdown` | `BudgetBreakdown` \| `null` | Yes | Per-category cost breakdown. Only for `itinerary` type |
| `recommendations` | `RecommendationItem[]` \| `null` | Yes | Only for `recommendation` type. 5–10 items |
| `clarifying_questions` | `string[]` \| `null` | Yes | Only for `clarifying` type. Max 4 questions |

---

### Sub-schema: `BaseHotel`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `place_id` | `string` | Yes | Google Place ID (format: `"ChIJ..."`) |
| `name` | `string` | No | Hotel name as in database |
| `description` | `string` | No | Always filled — backend auto-generates if AI omits it |
| `latitude` | `float` | Yes | Decimal latitude (e.g. `-8.5069`) |
| `longitude` | `float` | Yes | Decimal longitude (e.g. `115.2625`) |
| `district` | `string` | Yes | Bali district (e.g. `"Ubud"`, `"Seminyak"`, `"Kuta"`, `"Nusa Dua"`, `"Canggu"`, `"Sanur"`) |
| `image_url` | `string` \| `null` | Yes | Photo URL. `null` if not in database |
| `rating` | `float` | Yes | Google Maps rating 1.0–5.0 |
| `user_rating_count` | `integer` | Yes | Number of Google reviews |
| `price_per_night_idr` | `integer` | Yes | Estimated room price per night in IDR |
| `amenities` | `string[]` | Yes | e.g. `["kolam renang", "wifi gratis", "sarapan termasuk", "spa"]` |
| `check_in_time` | `string` | Yes | HH:MM format. Default `"14:00"` |
| `check_out_time` | `string` | Yes | HH:MM format. Default `"12:00"` |

---

### Sub-schema: `PlaceItem` (inside `itinerary_days[].places`)

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `place_id` | `string` | Yes | Google Place ID (format: `"ChIJ..."`) |
| `poi_id` | `string` \| `null` | Yes | Integer ID from Supabase `poi_attractions` table. `null` for restaurants |
| `name` | `string` | No | Place name exactly as in database |
| `category` | `"attraction"` \| `"restaurant"` \| `"hotel"` | No | Category. Backend coerces common variants automatically |
| `description` | `string` \| `null` | Yes | Full description from database `content` field |
| `latitude` | `float` \| `null` | Yes | Decimal latitude from database |
| `longitude` | `float` \| `null` | Yes | Decimal longitude from database |
| `district` | `string` \| `null` | Yes | Bali district (e.g. `"Badung"`, `"Gianyar"`, `"Tabanan"`) |
| `image_url` | `string` \| `null` | Yes | Photo URL. `null` if not in database |
| `rating` | `float` \| `null` | Yes | Rating 1.0–5.0 |
| `user_rating_count` | `integer` \| `null` | Yes | Number of reviews |
| `estimated_cost_idr` | `integer` \| `null` | Yes | Cost in IDR per person. Backend default: `25000` (attraction), `75000` (restaurant) |
| `tags` | `string[]` \| `null` | Yes | e.g. `["sunset", "fotografi", "budaya", "keluarga"]` |
| `visit_duration_mins` | `integer` \| `null` | Yes | Estimated visit duration in minutes. Backend default: `75` (attraction), `60` (restaurant) |
| `visit_time` | `string` \| `null` | Yes | HH:MM 24-hour format (e.g. `"08:00"`, `"12:30"`, `"17:00"`) |
| `opening_hours_note` | `string` \| `null` | Yes | Optional note about operating hours |
| `tips` | `string` \| `null` | Yes | One practical visit tip in Bahasa Indonesia |
| `topsis_score` | `float` \| `null` | Yes | TOPSIS ranking score 0.0–1.0. `null` for backend-injected restaurants |
| `route_to_next` | `RouteSegment` \| `null` | Yes | Route from this place to the next. **Always injected by backend, never by AI** |

---

### Sub-schema: `RouteSegment`

| Field | Type | Nullable | Possible Values | Description |
|-------|------|----------|-----------------|-------------|
| `distance_km` | `float` | Yes | Any positive float | Segment distance in km |
| `travel_time_mins` | `integer` | Yes | Any positive int | Estimated travel time in minutes |
| `traffic_delay_mins` | `integer` | Yes | 0 or positive int | Extra time due to traffic |
| `polyline` | `array` | Yes | `[{"lat": float, "lng": float}, ...]` | Route coordinates for map rendering |
| `status` | `string` | Yes | `"OK"` \| `"DEGRADED (Estimasi Haversine)"` | `"OK"` = TomTom API. `"DEGRADED"` = TomTom failed, Haversine fallback used |

---

### Sub-schema: `DailyItinerary`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `day` | `integer` | No | Day number starting from 1 |
| `date` | `string` \| `null` | Yes | Calendar date in `YYYY-MM-DD` format |
| `theme` | `string` \| `null` | Yes | Day theme (e.g. `"Spiritual & Budaya Ubud"`, `"Sunset di Tebing Barat"`) |
| `odalan_warning` | `string` \| `null` | Yes | Warning if Odalan ceremony conflicts with any planned place. `null` if no conflict |
| `weather_note` | `string` \| `null` | Yes | Weather note from live intel for this day. `null` if no specific info |
| `places` | `PlaceItem[]` | No | Ordered list of places. Backend injects restaurants automatically |
| `day_total_distance_km` | `float` \| `null` | Yes | Total driving distance for the day. **Injected by backend** |
| `day_total_travel_time_mins` | `integer` \| `null` | Yes | Total drive time for the day. **Injected by backend** |
| `day_full_polyline` | `array` \| `null` | Yes | Full day route polyline `[{"lat": float, "lng": float}, ...]`. **Injected by backend** |
| `route_from_hotel` | `RouteSegment` \| `null` | Yes | Route from hotel to first place. **Injected by backend** |

---

### Sub-schema: `BudgetBreakdown`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `accommodation_idr` | `integer` \| `null` | Yes | Hotel cost × number of nights |
| `food_idr` | `integer` \| `null` | Yes | Total food & drink cost for the trip |
| `transport_idr` | `integer` \| `null` | Yes | Transport (rental vehicle, fuel, ojek, parking) |
| `entrance_fee_idr` | `integer` \| `null` | Yes | Sum of all attraction `estimated_cost_idr` × number of people |
| `miscellaneous_idr` | `integer` \| `null` | Yes | Souvenirs, tips, sarong rental, donations (~10–15% of others) |

> `accommodation_idr + food_idr + transport_idr + entrance_fee_idr + miscellaneous_idr = total_budget_idr`

---

### Sub-schema: `RecommendationItem`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `place_id` | `string` \| `null` | Yes | Google Place ID |
| `poi_id` | `string` \| `null` | Yes | Supabase database integer ID |
| `name` | `string` | No | Official place name |
| `category` | `"attraction"` \| `"restaurant"` \| `"hotel"` | No | Place category |
| `description` | `string` \| `null` | Yes | Short 1–2 sentence description |
| `district` | `string` \| `null` | Yes | Bali district |
| `image_url` | `string` \| `null` | Yes | Photo URL |
| `rating` | `float` \| `null` | Yes | Rating 1.0–5.0 |
| `latitude` | `float` \| `null` | Yes | Decimal latitude |
| `longitude` | `float` \| `null` | Yes | Decimal longitude |
| `tags` | `string[]` \| `null` | Yes | Descriptive labels |
| `estimated_cost_idr` | `integer` \| `null` | Yes | Estimated cost per person in IDR |
| `topsis_score` | `float` \| `null` | Yes | TOPSIS score 0.0–1.0. Higher = better match |

---

### Error Responses for `POST /api/chat`

| HTTP Status | Condition |
|-------------|-----------|
| `401` | Token invalid or missing |
| `422` | Request body validation failed (e.g. `message` too long, invalid `mode`) |
| `429` | Rate limit exceeded (10/minute) |
| `500` | AI processing error or database error. Message: `"Terjadi kesalahan internal."` |
| `503` | Server not configured — `OPENAI_API_KEY` is empty |
| `504` | AI timeout — tool calling loop exceeded 120 seconds or 20 iterations |

---

## 2. List Itineraries — `GET /api/itineraries`

Requires authentication. Returns itineraries owned by the user (and optionally public ones from others).

### Query Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `include_public` | `boolean` | `false` | — | Include public itineraries from other users |
| `limit` | `integer` | `20` | 1–100 | Max results per request |
| `offset` | `integer` | `0` | ≥ 0 | Pagination offset |

### Response — `ItinerarySummary[]`

```json
[
  {
    "id": "f4e5d6c7-uuid",
    "user_id": "a1b2c3d4-user-uuid",
    "title": "Harmoni Ubud — 3 Hari di Jantung Bali",
    "description": null,
    "days_count": 3,
    "total_budget_idr": 5750000,
    "is_public": false,
    "share_slug": null,
    "created_at": "2026-07-01T10:00:00Z",
    "updated_at": "2026-07-01T10:00:00Z",
    "is_owner": true
  },
  {
    "id": "b3c4d5e6-uuid",
    "user_id": "z9y8x7-other-user",
    "title": "Weekend Kuta — 2 Hari Hedon",
    "description": null,
    "days_count": 2,
    "total_budget_idr": 2000000,
    "is_public": true,
    "share_slug": "weekend-kuta-2-hari",
    "created_at": "2026-06-15T08:00:00Z",
    "updated_at": "2026-06-15T08:00:00Z",
    "is_owner": false
  }
]
```

### `ItinerarySummary` Field Reference

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | `string (UUID)` | No | Itinerary unique ID |
| `user_id` | `string (UUID)` | No | Owner user ID |
| `title` | `string` \| `null` | Yes | Itinerary title |
| `description` | `string` \| `null` | Yes | Short description (rarely populated) |
| `days_count` | `integer` \| `null` | Yes | Number of days |
| `total_budget_idr` | `integer` \| `null` | Yes | Total estimated cost in IDR |
| `is_public` | `boolean` | No | `true` = visible to all; `false` = private |
| `share_slug` | `string` \| `null` | Yes | URL-friendly slug for public sharing |
| `created_at` | `string (ISO 8601)` \| `null` | Yes | Creation timestamp |
| `updated_at` | `string (ISO 8601)` \| `null` | Yes | Last update timestamp |
| `is_owner` | `boolean` | No | `true` = requesting user owns this; `false` = public item from another user |

> Items with `is_owner: false` = someone else's public itinerary. Can only be **copied**, not edited or deleted.

### Error Responses

| Status | Condition |
|--------|-----------|
| `401` | Unauthenticated |
| `429` | Rate limit exceeded |
| `500` | Database error |

---

## 3. Get Itinerary — `GET /api/itinerary/{itinerary_id}`

Fetch a single itinerary by UUID. Access is allowed if: (1) the requesting user is the owner, OR (2) the itinerary is public.

### Path Parameter

| Parameter | Type | Description |
|-----------|------|-------------|
| `itinerary_id` | `string (UUID)` | The itinerary UUID |

### Response

Returns the full raw row from `user_itineraries`, including `itinerary_data` (the full `FinalAIResponse` object) plus `is_owner: bool`.

```json
{
  "id": "f4e5d6c7-uuid",
  "user_id": "a1b2c3d4-user-uuid",
  "title": "Harmoni Ubud — 3 Hari di Jantung Bali",
  "itinerary_data": { "...full FinalAIResponse object..." },
  "days_count": 3,
  "total_budget_idr": 5750000,
  "is_public": false,
  "is_owner": true,
  "created_at": "2026-07-01T10:00:00Z",
  "updated_at": "2026-07-01T10:00:00Z"
}
```

### Error Responses

| Status | Condition |
|--------|-----------|
| `400` | `itinerary_id` is not a valid UUID format |
| `401` | Unauthenticated |
| `404` | Not found, or private and not owned by requesting user |
| `429` | Rate limit exceeded |
| `500` | Database error |

---

## 4. Update Itinerary — `PUT /api/itinerary/{itinerary_id}`

Manual update from UI. Only the owner can update.

### Path Parameter

| Parameter | Type | Description |
|-----------|------|-------------|
| `itinerary_id` | `string (UUID)` | The itinerary UUID |

### Request Body

```json
{
  "title": "Judul Baru (opsional)",
  "itinerary_data": {
    "response_type": "itinerary",
    "...full FinalAIResponse object..."
  },
  "total_budget_idr": 4000000,
  "is_public": false
}
```

### Request Fields

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `itinerary_data` | `object` | **Yes** | Full `FinalAIResponse` | The entire modified itinerary object |
| `title` | `string` \| `null` | No | Max 200 chars | New itinerary title. Omit to keep existing |
| `total_budget_idr` | `integer` \| `null` | No | ≥ 0 | Updated total budget in IDR |
| `is_public` | `boolean` \| `null` | No | — | Visibility toggle |

### Response

Returns the updated row from `user_itineraries` (same shape as `GET /api/itinerary/{id}`).

### Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Invalid UUID format |
| `401` | Unauthenticated |
| `404` | Not found or not the owner |
| `422` | Request body validation failed |
| `429` | Rate limit exceeded |
| `500` | Database error |

---

## 5. Toggle Visibility — `PATCH /api/itinerary/{itinerary_id}/visibility`

Make an itinerary public or private. Only the owner can toggle.

### Request Body

```json
{ "is_public": true }
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `is_public` | `boolean` | **Yes** | `true` = public; `false` = private |

### Response

```json
{
  "status": "updated",
  "itinerary_id": "f4e5d6c7-uuid",
  "is_public": true
}
```

### Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Invalid UUID format |
| `401` | Unauthenticated |
| `404` | Not found or not the owner |
| `422` | Body validation failed (e.g. `is_public` is missing) |
| `429` | Rate limit exceeded |
| `500` | Database error |

---

## 6. Copy Public Itinerary — `POST /api/itinerary/{itinerary_id}/copy`

Copy another user's public itinerary to your own account. The copy is private and fully editable. The original is unaffected.

### Response

```json
{
  "status": "copied",
  "new_itinerary_id": "b8c9d0e1-new-uuid",
  "title": "Salinan: Harmoni Ubud — 3 Hari di Jantung Bali"
}
```

### Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Invalid UUID format |
| `401` | Unauthenticated |
| `404` | Itinerary not found or not public |
| `429` | Rate limit exceeded |
| `500` | Database error |

---

## 7. Delete Itinerary — `DELETE /api/itinerary/{itinerary_id}`

Permanently delete an itinerary. Only the owner can delete.

### Response

```json
{
  "status": "deleted",
  "itinerary_id": "f4e5d6c7-uuid"
}
```

### Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Invalid UUID format |
| `401` | Unauthenticated |
| `404` | Not found or not the owner |
| `429` | Rate limit exceeded |
| `500` | Database error |

---

## 8. Place Search — `GET /api/place/search`

Keyword (name) search for a specific place. Returns exact or partial name matches.

### Query Parameters

| Parameter | Type | Required | Possible Values | Default | Description |
|-----------|------|----------|-----------------|---------|-------------|
| `query` | `string` | **Yes** | Any string 1–200 chars | — | Place name to search for |
| `category` | `string` | No | `"attraction"` \| `"hotel"` \| `"restaurant"` | `"attraction"` | Which table to search |

### Response

```json
{
  "results": [
    {
      "place_id": "ChIJxxxTanahLot",
      "name": "Tanah Lot",
      "district": "Tabanan",
      "latitude": -8.6215,
      "longitude": 115.0866,
      "rating": 4.7,
      "user_rating_count": 52000,
      "image_url": "https://example.com/tanah-lot.jpg",
      "content": "Pura ikonik di atas batu karang di tepi laut...",
      "metadata": { "...raw metadata object..." }
    }
  ],
  "count": 1
}
```

### Error Responses

| Status | Condition |
|--------|-----------|
| `401` | Unauthenticated |
| `422` | `query` missing, too short, or too long; invalid `category` value |
| `429` | Rate limit exceeded |
| `500` | Database error |

---

## 9. Place Recommendations — `GET /api/place/recommendations`

Semantic Search (RAG + pgvector) + TOPSIS ranking without building a full itinerary.

### Query Parameters

| Parameter | Type | Required | Possible Values | Default | Description |
|-----------|------|----------|-----------------|---------|-------------|
| `query` | `string` | **Yes** | Any string 1–200 chars | — | Theme or keyword (e.g. `"pantai sunset"`, `"budaya ubud"`, `"wisata alam air terjun"`) |
| `category` | `string` | No | `"poi"` \| `"hotel"` \| `"restaurant"` | `"poi"` | Which type of place to search (`"poi"` = tourist attractions) |
| `limit` | `integer` | No | 1–30 | `10` | Number of results |

### Response

```json
{
  "results": [
    {
      "place_id": "ChIJxxxPantaiPandawa",
      "name": "Pantai Pandawa",
      "district": "Badung",
      "latitude": -8.8478,
      "longitude": 115.2105,
      "rating": 4.6,
      "user_rating_count": 28000,
      "image_url": "https://example.com/pandawa.jpg",
      "content": "Pantai tersembunyi di balik tebing kapur...",
      "topsis_score": 0.8732,
      "topsis_category": "poi",
      "metadata": { "...raw metadata object..." }
    }
  ],
  "count": 10,
  "query": "pantai sunset"
}
```

### Error Responses

| Status | Condition |
|--------|-----------|
| `401` | Unauthenticated |
| `422` | `query` missing, too short, or too long; invalid `category` value; `limit` out of range |
| `429` | Rate limit exceeded |
| `500` | Database error |

---

## 10. Health Check — `GET /health`

No authentication required.

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
    "tomtom_routing",
    "rate_limiting",
    "security_headers"
  ]
}
```

| Field | Possible Values | Description |
|-------|-----------------|-------------|
| `status` | `"healthy"` | Always `"healthy"` if server is running |
| `version` | `"8.0"` | API version |
| `service` | `"SobatNavi AI Agent"` | Service identifier |
| `features` | Array of strings | List of active backend features |

---

## Global Error Code Summary

| HTTP Status | Condition |
|-------------|-----------|
| `200` | Success |
| `400` | Bad request (e.g. invalid UUID format) |
| `401` | Missing or invalid authentication token |
| `403` | Token valid but no access to resource |
| `404` | Resource not found or inaccessible |
| `422` | Request body or query parameter validation failed |
| `429` | Rate limit exceeded |
| `500` | Internal server error (AI processing or database error) |
| `503` | Server not configured — `OPENAI_API_KEY` is empty |
| `504` | AI timeout — exceeded 120 seconds or 20 tool-call iterations |

---

## Ownership & Sharing Rules

```
User A creates Itinerary X (is_public=false)
  → Only User A: GET / PUT / DELETE / PATCH visibility

User A publishes Itinerary X (is_public=true)
  → User A + User B: can GET
  → User B: can POST /copy → creates Itinerary Y (private, fully editable by B)
  → User B: CANNOT PUT/DELETE Itinerary X
  → Only User A: can toggle back to private

Guest (no token):
  → POST /api/chat: allowed (history from request, not saved to DB)
  → All other /api/* endpoints: 401 Unauthorized
```

---

## Backend Processing Pipeline (POST /api/chat)

```
1. Validate request & auth token (optional)
2. Parse message → detect pace, num_days, is_half_day
3. Map budget_preference → preference_mode ("budget"|"standard"|"luxury")
4. Calculate POI budget (attractions/day, meals/day)
5. Build Heidi system prompt (includes POI budget context + budget preference)
6. Load chat history (from DB if authenticated, from request.history if guest)
7. AI Tool-Calling Loop (max 20 iterations, 120s timeout):
   └── get_bali_context() → weather + Odalan zones
   └── get_smart_recommendations() → Semantic Search + DBSCAN + TOPSIS
   └── get_nearby_places() → nearby hotel/restaurant
   └── search_specific_place() → exact name lookup
   └── validate_itinerary_safety() → Odalan conflict check
   └── get_inspiration_narration() → poetic place narratives
8. Parse & validate JSON → FinalAIResponse (Pydantic)
9. Sanitize output (fix categories, fill missing defaults)
10. POST-PROCESSING (itinerary only):
    ├── guarantee_base_hotel() → ensure hotel always exists
    ├── inject_meals_to_itinerary() → insert restaurants per day
    └── Routing injection → TomTom batch routes + polylines
11. Auto-save to DB (authenticated only)
12. Return FinalAIResponse
```

---

## POI Budget Calculator Reference

Backend auto-calculates POI budget before AI is called. The result is sent to AI as mandatory context.

| Pace | Attractions/day | Meals/day | Slots | Total places/day |
|------|-----------------|-----------|-------|-----------------|
| `santai` | 2–3 | 1 (lunch) | `makan_siang` | 3–4 |
| `normal` (default) | 3–4 | 2 (lunch + dinner) | `makan_siang`, `makan_malam` | 5–6 |
| `padat` | 4–5 | 2 (lunch + dinner) | `makan_siang`, `makan_malam` | 6–7 |
| Half-day | 2 | 1 (lunch) | `makan_siang` | 3 |
| Custom (explicit count) | 1–8 | 1 if <4 attrs, else 2 | varies | count + meals |

> **5+ day trips**: ideal attraction count is reduced by 1 automatically to prevent fatigue.

---

## Meal Slot Reference

Restaurants are injected by the backend after AI responds. Frontend receives them as regular `PlaceItem` with `category: "restaurant"`.

| Slot | `visit_time` | `visit_duration_mins` | `estimated_cost_idr` | Position in `places[]` |
|------|--------------|-----------------------|----------------------|------------------------|
| `sarapan` | `"08:00"` | `45` | `50000` | Index 0 (start of day) |
| `makan_siang` | `"12:30"` | `60` | `75000` | Middle (~50% of attractions) |
| `makan_malam` | `"19:00"` | `75` | `100000` | Last position |

**Restaurant search order (priority):**
1. `search_amenities_nearby` — 5km radius from centroid of day's attractions
2. Semantic search fallback:
   - `budget`: `"warung makan murah {district} Bali harga terjangkau"`
   - `luxury`: `"restoran fine dining mewah {district} Bali premium"`
   - `moderate` (default): `"restoran {district} Bali"`
3. If nothing found → meal slot is skipped (server log warning)

---

## TOPSIS Internal Weights & Budget Mode Effects

| Category | Feature | Base Weight | `budget` mode | `luxury` mode |
|----------|---------|-------------|---------------|---------------|
| **POI** | `rating` | 30% | 30% | 30% |
| **POI** | `popularity` | 20% | 20% | 20% |
| **POI** | `price_value` | 15% | **35%** (boost) | 15% (impact flipped to cost) |
| **POI** | `strategic_score` | 20% | 20% | 20% |
| **POI** | `visual_interest_index` | 15% | 15% | 15% |
| **Hotel** | `rating` | 35% | — | — |
| **Hotel** | `comfort_index` | 30% | — | — |
| **Hotel** | `amenity_density` | 20% | — | — |
| **Hotel** | `accessibility_index` | 15% | — | — |
| **Restaurant** | `rating` | 35% | — | — |
| **Restaurant** | `menu_variety_index` | 25% | — | — |
| **Restaurant** | `ambience_score` | 25% | — | — |
| **Restaurant** | `payment_modern_index` | 15% | — | — |

> Features read from `metadata.topsis_features` in database. If missing, system falls back to raw metadata fields automatically.

> Backend always fetches **3× more** than needed from the database (min 30, max 80 items) before TOPSIS + DBSCAN selects the best. This ensures quality even when the final quota is small.