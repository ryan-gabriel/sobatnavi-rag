# app/engine/recommender.py
# ─────────────────────────────────────────────────────────────────────────────
# v9.0 — Radius & Anchor-First Clustering
#
# PERUBAHAN UTAMA dari v8 (DBSCAN):
#   1. DBSCAN sepenuhnya dihapus — tidak lagi bergantung pada sklearn
#   2. Clustering diganti dengan metode "Radius & Anchor-First":
#      - Hari 1 anchor = POI peringkat tertinggi dari semantic search global
#      - Hari 2..N anchor = kandidat peringkat rendah dalam radius 20km
#      - Setiap hari diisi via search_pois_nearby & search_restaurants_nearby
#   3. Anti-duplication filter menggunakan Python set()
#   4. TOPSIS scoring tetap dipertahankan untuk ranking internal
# ─────────────────────────────────────────────────────────────────────────────

from app.services.tomtom_service import _haversine_km
import numpy as np
import logging
import math
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ANCHOR & RADIUS CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Radius maksimum dari Day 1 Central Anchor untuk memilih anchor hari berikutnya
MAX_ANCHOR_DISTANCE_KM = 20.0

# Radius pencarian POI di sekitar anchor harian (meter)
POI_NEARBY_RADIUS_M = 12000   # 12 km (range: 8-15 km)

# Radius pencarian restoran di sekitar anchor harian (meter)
RESTAURANT_NEARBY_RADIUS_M = 6000   # 6 km (range: 5-8 km)

# Jumlah maksimal POI yang diambil dari global semantic search
GLOBAL_SEARCH_LIMIT = 25

# Jumlah POI per hari dari nearby search
NEARBY_POI_LIMIT = 10

# Jumlah restoran per hari dari nearby search
NEARBY_RESTAURANT_LIMIT = 6


# ─────────────────────────────────────────────────────────────────────────────
# TOPSIS CONFIG (TIDAK BERUBAH dari v8)
# ─────────────────────────────────────────────────────────────────────────────

TOPSIS_CONFIG = {
    "poi": {
        "features": ["rating", "popularity", "price_value", "strategic_score", "visual_interest_index"],
        "weights": [0.30, 0.20, 0.15, 0.20, 0.15],
        "impacts": [1, 1, 1, 1, 1],
    },
    "hotel": {
        "features": ["rating", "comfort_index", "amenity_density", "accessibility_index"],
        "weights": [0.35, 0.30, 0.20, 0.15],
        "impacts": [1, 1, 1, 1],
    },
    "restaurant": {
        "features": ["rating", "menu_variety_index", "ambience_score", "payment_modern_index"],
        "weights": [0.35, 0.25, 0.25, 0.15],
        "impacts": [1, 1, 1, 1],
    },
}

# Nilai default jika sebuah fitur tidak ditemukan di metadata
DEFAULT_FEATURE_VALUES = {
    "rating": 0.0,
    "popularity": 0.0,
    "price_value": 0.0,
    "strategic_score": 0.0,
    "visual_interest_index": 0.0,
    "comfort_index": 0.0,
    "amenity_density": 0.0,
    "accessibility_index": 0.0,
    "menu_variety_index": 0.0,
    "ambience_score": 0.0,
    "payment_modern_index": 0.0,
}


def _normalize_to_01(value: float, min_val: float = 0.0, max_val: float = 5.0) -> float:
    """Normalkan nilai ke rentang [0, 1]. Menangani edge case."""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def extract_topsis_features(place: dict, category: str = "poi") -> dict:
    """
    Mengekstrak fitur TOPSIS dari sebuah tempat.
    Urutan prioritas:
      1. metadata.topsis_features (data terstruktur)
      2. Estimasi dari raw metadata (fallback kalkulasi)
      3. Default values (fallback akhir, tidak crash)
    """
    config = TOPSIS_CONFIG.get(category, TOPSIS_CONFIG["poi"])
    feature_keys = config["features"]

    # Coba ambil metadata
    metadata = place.get("metadata") or {}
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    topsis_features = metadata.get("topsis_features") or {}

    extracted = {}

    for key in feature_keys:
        # Prioritas 1: Dari topsis_features
        if key in topsis_features:
            extracted[key] = float(topsis_features[key])
            continue

        # Prioritas 2: Fallback dari field utama
        if key == "rating":
            raw_rating = place.get("rating") or metadata.get("rating") or 0.0
            extracted[key] = _normalize_to_01(float(raw_rating), 0.0, 5.0)
            continue

        if key == "popularity":
            user_count = place.get("user_rating_count") or metadata.get("user_ratings_total") or 0
            # Normalisasi popularitas: 1000+ review → mendekati 1.0
            extracted[key] = _normalize_to_01(float(user_count), 0.0, 1000.0)
            continue

        if key == "amenity_density":
            # Estimasi dari panjang array amenities
            amenities = metadata.get("amenities") or []
            extracted[key] = _normalize_to_01(float(len(amenities)), 0.0, 20.0)
            continue

        if key == "menu_variety_index":
            # Estimasi dari panjang array serves (menu items)
            serves = metadata.get("serves") or []
            extracted[key] = _normalize_to_01(float(len(serves)), 0.0, 15.0)
            continue

        if key == "payment_modern_index":
            # Estimasi dari ketersediaan payment options
            payment = metadata.get("payment_options") or {}
            modern_methods = sum([
                payment.get("acceptsCreditCards", False),
                payment.get("acceptsDebitCards", False),
                payment.get("acceptsCashOnly", False) is False,
            ])
            extracted[key] = _normalize_to_01(float(modern_methods), 0.0, 3.0)
            continue

        if key == "price_value":
            # Estimasi dari price_level (lower is better value)
            price_level = metadata.get("price_level") or metadata.get("priceLevel") or "PRICE_LEVEL_MODERATE"
            price_map = {
                "PRICE_LEVEL_FREE": 1.0,
                "PRICE_LEVEL_INEXPENSIVE": 0.85,
                "PRICE_LEVEL_MODERATE": 0.65,
                "PRICE_LEVEL_EXPENSIVE": 0.4,
                "PRICE_LEVEL_VERY_EXPENSIVE": 0.2,
            }
            extracted[key] = price_map.get(price_level, DEFAULT_FEATURE_VALUES.get(key, 0.5))
            continue

        if key == "accessibility_index":
            # Estimasi dari wheelchair_accessible
            accessible = metadata.get("wheelchairAccessibleEntrance") or metadata.get("accessibilityOptions", {}).get("wheelchairAccessibleEntrance", False)
            extracted[key] = 0.8 if accessible else 0.4
            continue

        # Prioritas 3: Default value (tidak crash)
        logger.debug(f"Feature '{key}' tidak ditemukan untuk place '{place.get('name', 'unknown')}', pakai default.")
        extracted[key] = DEFAULT_FEATURE_VALUES.get(key, 0.5)

    return extracted


def topsis_score(data_matrix: np.ndarray, weights: np.ndarray, impacts: np.ndarray) -> np.ndarray:
    """
    Menghitung skor TOPSIS multi-dimensi.

    Args:
        data_matrix: Matrix (n_places x n_features) nilai yang sudah dalam rentang [0, 1]
        weights: Array bobot per fitur (harus sum = 1)
        impacts: Array +1 (benefit) atau -1 (cost) per fitur

    Returns:
        Array skor kedekatan relatif per tempat (0~1, makin tinggi makin baik)
    """
    if data_matrix.shape[0] == 0:
        return np.array([])

    # TC-06: Handle matrix dengan satu baris (edge case)
    if data_matrix.shape[0] == 1:
        return np.array([1.0])

    # 1. Normalisasi (vector normalization)
    norm_divisor = np.linalg.norm(data_matrix, axis=0)
    norm_divisor[norm_divisor == 0] = 1.0
    norm_matrix = data_matrix / norm_divisor

    # 2. Kalikan dengan bobot
    weighted = norm_matrix * weights

    # 3. Solusi Ideal Positif & Negatif (mempertimbangkan benefit vs cost)
    ideal_best = np.where(impacts == 1, np.max(weighted, axis=0), np.min(weighted, axis=0))
    ideal_worst = np.where(impacts == 1, np.min(weighted, axis=0), np.max(weighted, axis=0))

    # 4. Jarak Euclidean dari solusi ideal
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # 5. Skor Kedekatan Relatif
    denominator = dist_best + dist_worst
    denominator[denominator == 0] = 1.0

    scores = dist_worst / denominator
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# TOPSIS RANKING (standalone, tanpa spatial clustering)
# ─────────────────────────────────────────────────────────────────────────────

def rank_pois_by_topsis(
    pois: list,
    category: Literal["poi", "hotel", "restaurant"] = "poi",
    preference_mode: str = "standard",
    top_n: int = 10,
) -> list:
    """
    Meranking POI/hotel/restoran menggunakan TOPSIS multi-dimensi TANPA
    spatial clustering. Cocok untuk endpoint rekomendasi standalone.

    Args:
        pois: List dict hasil query database
        category: 'poi', 'hotel', atau 'restaurant' — menentukan bobot TOPSIS
        preference_mode: 'standard', 'hidden_gem', 'luxury', 'budget'
        top_n: Jumlah hasil teratas yang dikembalikan

    Returns:
        List tempat terbaik yang sudah diurutkan berdasarkan TOPSIS score
    """
    if not pois:
        return []

    import copy
    config = copy.deepcopy(TOPSIS_CONFIG.get(category, TOPSIS_CONFIG["poi"]))
    feature_keys = config["features"]

    # Adjust weights/impacts berdasarkan preference mode
    if preference_mode == "hidden_gem" and "popularity" in feature_keys:
        idx = feature_keys.index("popularity")
        config["impacts"][idx] = -1
        config["weights"][idx] = 0.25

    elif preference_mode == "luxury" and "price_value" in feature_keys:
        idx = feature_keys.index("price_value")
        config["impacts"][idx] = -1

    elif preference_mode == "budget" and "price_value" in feature_keys:
        idx = feature_keys.index("price_value")
        config["weights"][idx] = 0.35

    weights = np.array(config["weights"])
    impacts = np.array(config["impacts"])

    # Ekstraksi fitur
    feature_matrix_rows = []
    valid_pois = []

    for p in pois:
        try:
            features = extract_topsis_features(p, category)
            feature_row = [features.get(k, DEFAULT_FEATURE_VALUES.get(k, 0.5)) for k in feature_keys]
        except Exception as e:
            logger.warning(f"Gagal ekstrak fitur untuk '{p.get('name', 'unknown')}': {e}. Pakai default.")
            feature_row = [DEFAULT_FEATURE_VALUES.get(k, 0.5) for k in feature_keys]

        feature_matrix_rows.append(feature_row)
        valid_pois.append(p)

    if not valid_pois:
        return pois[:top_n]

    data_matrix = np.array(feature_matrix_rows, dtype=float)

    try:
        scores = topsis_score(data_matrix, weights, impacts)
    except Exception as e:
        logger.error(f"TOPSIS scoring error: {e}. Pakai score=1.0 untuk semua.")
        scores = np.ones(len(valid_pois))

    for i, p in enumerate(valid_pois):
        p["topsis_score"] = round(float(scores[i]), 4)
        p["topsis_category"] = category

    # Urutkan berdasarkan TOPSIS score
    valid_pois.sort(key=lambda x: x.get("topsis_score", 0), reverse=True)

    logger.info(
        f"rank_pois_by_topsis [{category}]: {len(pois)} raw → "
        f"{min(top_n, len(valid_pois))} selected (mode={preference_mode})"
    )
    return valid_pois[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# RADIUS & ANCHOR-FIRST CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def _get_place_id(place: dict) -> str:
    """Mengambil identifier unik dari sebuah tempat untuk deduplication."""
    return (
        place.get("place_id")
        or place.get("id")
        or place.get("name", "")
    )


def _select_day_anchors(
    ranked_candidates: list,
    primary_anchor_lat: float,
    primary_anchor_lng: float,
    num_days: int,
) -> list[dict]:
    """
    Memilih anchor point untuk setiap hari perjalanan.

    Day 1: Menggunakan primary anchor (POI peringkat tertinggi).
    Day 2..N: Pilih dari kandidat yang berjarak ≤ MAX_ANCHOR_DISTANCE_KM
              dari primary anchor, dengan jarak antar-anchor dimaksimalkan
              agar setiap hari mengeksplorasi area yang sedikit berbeda.

    CRITICAL FALLBACK: Jika tidak ada kandidat terdekat yang cukup,
    replikasi koordinat primary anchor.

    Returns:
        List of dicts: [{"day": int, "lat": float, "lng": float, "name": str, "place_id": str}, ...]
    """
    anchors = [{
        "day": 1,
        "lat": primary_anchor_lat,
        "lng": primary_anchor_lng,
        "name": ranked_candidates[0].get("name", "Primary Anchor") if ranked_candidates else "Unknown",
        "place_id": _get_place_id(ranked_candidates[0]) if ranked_candidates else None,
    }]

    if num_days <= 1:
        return anchors

    # Kumpulkan kandidat anchor yang dalam radius dari primary anchor
    # Skip index 0 (sudah jadi primary anchor)
    nearby_candidates = []
    for candidate in ranked_candidates[1:]:
        c_lat = candidate.get("latitude")
        c_lng = candidate.get("longitude")
        if c_lat is None or c_lng is None:
            continue

        dist = _haversine_km(primary_anchor_lat, primary_anchor_lng, c_lat, c_lng)
        if dist <= MAX_ANCHOR_DISTANCE_KM:
            nearby_candidates.append({
                "lat": c_lat,
                "lng": c_lng,
                "name": candidate.get("name", "Anchor"),
                "place_id": _get_place_id(candidate),
                "distance_from_primary": dist,
            })

    # Sortir agar yang paling jauh dari primary (tapi masih dalam radius) dipilih duluan
    # Ini memastikan setiap hari mengeksplorasi area yang sedikit berbeda
    nearby_candidates.sort(key=lambda x: x["distance_from_primary"], reverse=True)

    used_anchor_ids = {anchors[0].get("place_id")}

    for day_num in range(2, num_days + 1):
        anchor_found = False
        for candidate in nearby_candidates:
            cid = candidate.get("place_id")
            if cid and cid not in used_anchor_ids:
                anchors.append({
                    "day": day_num,
                    "lat": candidate["lat"],
                    "lng": candidate["lng"],
                    "name": candidate["name"],
                    "place_id": cid,
                })
                used_anchor_ids.add(cid)
                anchor_found = True
                break

        # CRITICAL FALLBACK: Replikasi koordinat Day 1 jika tidak ada kandidat
        if not anchor_found:
            logger.warning(
                f"Hari {day_num}: Tidak ada kandidat anchor dalam radius "
                f"{MAX_ANCHOR_DISTANCE_KM}km. Replikasi koordinat Day 1."
            )
            anchors.append({
                "day": day_num,
                "lat": primary_anchor_lat,
                "lng": primary_anchor_lng,
                "name": anchors[0]["name"],
                "place_id": anchors[0].get("place_id"),
            })

    return anchors


async def generate_clustered_pool_delivery(
    supabase_service,
    query: str,
    num_days: int,
    user_detected_location: Optional[str] = None,
    preference_mode: str = "standard",
    user_requested_pois: Optional[dict] = None,
) -> list[dict]:
    """
    Radius & Anchor-First clustering engine.

    Menghasilkan pool POI dan restoran per hari yang:
    - Secara geografis ketat (tight clustering)
    - Efisien (tidak ada backtracking lintas area)
    - Fleksibel (2-4 POI per hari, bisa lebih jika diminta user)
    - Bebas duplikasi lintas hari (anti-duplication filter)
    - SELALU menghasilkan tepat num_days objek hari

    Workflow:
    1. Global semantic search → Top 20-25 POIs
    2. Rank via TOPSIS
    3. Pilih anchor per hari (Day 1 = Rank 1, Day 2..N = dalam radius 20km)
    4. Per-day spatial search: POI nearby + restaurant nearby
    5. Deduplicate across days
    6. Return structured per-day pool

    Args:
        supabase_service: Instance SupabaseService untuk akses database
        query: Query tema wisata dari user (misal: "pantai sunset ubud")
        num_days: Jumlah hari perjalanan yang diminta
        user_detected_location: District/area yang eksplisit disebutkan user (opsional)
        preference_mode: 'standard', 'hidden_gem', 'luxury', 'budget'

    Returns:
        List of dicts, satu per hari:
        [
          {
            "day": 1,
            "anchor": {"name": str, "lat": float, "lng": float, "place_id": str},
            "pois": [...],         # TOPSIS-ranked attractions
            "restaurants": [...],  # Restoran terdekat dari anchor
          },
          ...
        ]
    """
    num_days = max(1, min(num_days, 14))  # Clamp: 1-14 hari

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Global Semantic Search
    # ═══════════════════════════════════════════════════════════════════════
    logger.info(
        f"generate_clustered_pool_delivery: query='{query}', "
        f"num_days={num_days}, location={user_detected_location}"
    )

    try:
        global_results = await supabase_service.search_pois_semantic(
            query=query,
            limit=GLOBAL_SEARCH_LIMIT,
            filter_district=user_detected_location,
        )
    except Exception as e:
        logger.error(f"Global semantic search gagal: {e}")
        global_results = []

    if not global_results:
        logger.warning("Global semantic search mengembalikan 0 hasil. Return pool kosong.")
        return [
            {
                "day": d + 1,
                "anchor": {"name": "Unknown", "lat": -8.4095, "lng": 115.1889, "place_id": None},
                "pois": [],
                "restaurants": [],
            }
            for d in range(num_days)
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: TOPSIS Ranking pada hasil global
    # ═══════════════════════════════════════════════════════════════════════
    ranked_global = rank_pois_by_topsis(
        global_results,
        category="poi",
        preference_mode=preference_mode,
        top_n=GLOBAL_SEARCH_LIMIT,
    )

    # Pastikan ada koordinat pada primary anchor
    primary = ranked_global[0]
    primary_lat = primary.get("latitude")
    primary_lng = primary.get("longitude")

    if primary_lat is None or primary_lng is None:
        # Cari kandidat pertama yang punya koordinat
        for candidate in ranked_global[1:]:
            if candidate.get("latitude") is not None and candidate.get("longitude") is not None:
                primary = candidate
                primary_lat = candidate["latitude"]
                primary_lng = candidate["longitude"]
                break
        else:
            # Fallback ke koordinat Bali tengah
            logger.warning("Tidak ada POI dengan koordinat valid. Pakai fallback Bali tengah.")
            primary_lat = -8.4095
            primary_lng = 115.1889

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Pilih Anchor untuk setiap hari
    # ═══════════════════════════════════════════════════════════════════════
    day_anchors = _select_day_anchors(ranked_global, primary_lat, primary_lng, num_days)

    logger.info(
        f"Day anchors selected: "
        + ", ".join(f"Day {a['day']}={a['name']} ({a['lat']:.4f},{a['lng']:.4f})" for a in day_anchors)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 & 5: Per-Day Spatial Search + Anti-Duplication
    # ═══════════════════════════════════════════════════════════════════════
    global_used_ids: set[str] = set()  # Anti-duplication across days
    daily_pools: list[dict] = []

    for anchor in day_anchors:
        day_num = anchor["day"]
        anchor_lat = anchor["lat"]
        anchor_lng = anchor["lng"]

        # ── POI Nearby Search ────────────────────────────────────────────
        day_pois: list[dict] = []
        try:
            nearby_pois = await supabase_service.search_pois_nearby(
                lat=anchor_lat,
                lng=anchor_lng,
                radius_m=POI_NEARBY_RADIUS_M,
                limit=NEARBY_POI_LIMIT,
            )
        except Exception as e:
            logger.warning(f"Hari {day_num}: search_pois_nearby gagal ({e})")
            nearby_pois = []

        # Deduplicate + filter
        for poi in nearby_pois:
            pid = _get_place_id(poi)
            if pid and pid not in global_used_ids:
                day_pois.append(poi)
                global_used_ids.add(pid)

        # Jika nearby tidak cukup, tambahkan dari global results yang belum terpakai
        if len(day_pois) < 2:
            for gp in ranked_global:
                gp_id = _get_place_id(gp)
                gp_lat = gp.get("latitude")
                gp_lng = gp.get("longitude")
                if gp_id and gp_id not in global_used_ids and gp_lat is not None and gp_lng is not None:
                    dist = _haversine_km(anchor_lat, anchor_lng, gp_lat, gp_lng)
                    if dist <= MAX_ANCHOR_DISTANCE_KM:
                        day_pois.append(gp)
                        global_used_ids.add(gp_id)
                        if len(day_pois) >= 3:
                            break

        # TOPSIS rank the day's POIs
        if day_pois:
            day_pois = rank_pois_by_topsis(
                day_pois,
                category="poi",
                preference_mode=preference_mode,
                top_n=NEARBY_POI_LIMIT,
            )

        # ── Restaurant Nearby Search ─────────────────────────────────────
        day_restaurants: list[dict] = []
        try:
            nearby_restaurants = await supabase_service.search_amenities_nearby(
                amenity_type="restaurant",
                lat=anchor_lat,
                lng=anchor_lng,
                radius_m=RESTAURANT_NEARBY_RADIUS_M,
                limit=NEARBY_RESTAURANT_LIMIT,
            )
        except Exception as e:
            logger.warning(f"Hari {day_num}: search_restaurants_nearby gagal ({e})")
            nearby_restaurants = []

        # Deduplicate restaurants
        for resto in nearby_restaurants:
            rid = _get_place_id(resto)
            if rid and rid not in global_used_ids:
                day_restaurants.append(resto)
                global_used_ids.add(rid)

        # Fallback: jika tidak ada restoran nearby, coba semantic search
        if not day_restaurants:
            try:
                _restaurant_query_map = {
                    "budget": f"warung makan murah {user_detected_location or 'Bali'} harga terjangkau",
                    "luxury": f"restoran fine dining mewah {user_detected_location or 'Bali'} premium",
                }
                rq = _restaurant_query_map.get(preference_mode, f"restoran {user_detected_location or 'Bali'}")
                semantic_restaurants = await supabase_service.search_amenities_semantic(
                    query=rq,
                    amenity_type="restaurant",
                    limit=NEARBY_RESTAURANT_LIMIT,
                )
                for sr in semantic_restaurants:
                    sr_id = _get_place_id(sr)
                    if sr_id and sr_id not in global_used_ids:
                        day_restaurants.append(sr)
                        global_used_ids.add(sr_id)
                        if len(day_restaurants) >= 2:
                            break
            except Exception as e:
                logger.warning(f"Hari {day_num}: semantic restaurant fallback gagal ({e})")

        # TOPSIS rank restaurants
        if day_restaurants:
            day_restaurants = rank_pois_by_topsis(
                day_restaurants,
                category="restaurant",
                preference_mode=preference_mode,
                top_n=NEARBY_RESTAURANT_LIMIT,
            )

        # ── Assemble day pool ────────────────────────────────────────────
        daily_pools.append({
            "day": day_num,
            "anchor": {
                "name": anchor["name"],
                "lat": anchor_lat,
                "lng": anchor_lng,
                "place_id": anchor.get("place_id"),
            },
            "pois": day_pois,
            "restaurants": day_restaurants,
        })

        logger.info(
            f"Hari {day_num}: anchor='{anchor['name']}' "
            f"→ {len(day_pois)} POIs, {len(day_restaurants)} restoran"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Final Validation & Return
    # ═══════════════════════════════════════════════════════════════════════

    # Pastikan selalu mengembalikan tepat num_days hari
    while len(daily_pools) < num_days:
        missing_day = len(daily_pools) + 1
        logger.warning(f"Hari {missing_day}: Pool kosong, tambahkan placeholder.")
        daily_pools.append({
            "day": missing_day,
            "anchor": {
                "name": daily_pools[0]["anchor"]["name"] if daily_pools else "Unknown",
                "lat": daily_pools[0]["anchor"]["lat"] if daily_pools else -8.4095,
                "lng": daily_pools[0]["anchor"]["lng"] if daily_pools else 115.1889,
                "place_id": daily_pools[0]["anchor"].get("place_id") if daily_pools else None,
            },
            "pois": [],
            "restaurants": [],
        })

    total_pois = sum(len(d["pois"]) for d in daily_pools)
    total_restos = sum(len(d["restaurants"]) for d in daily_pools)
    logger.info(
        f"generate_clustered_pool_delivery DONE: {num_days} hari, "
        f"{total_pois} total POIs, {total_restos} total restoran, "
        f"{len(global_used_ids)} unique IDs tracked"
    )

    structured_itinerary = build_deterministic_itinerary(daily_pools, user_requested_pois)
    return structured_itinerary


def map_to_place_item(item: dict, category: str, visit_time: str, visit_duration: int) -> dict:
    meta = item.get("metadata") or {}
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
            
    desc = item.get("content") or item.get("description")
    if not desc and meta:
        desc = meta.get("description") or meta.get("content")
    if not desc:
        parts = [f"{item.get('name', 'Tempat')}"]
        if item.get("district"):
            parts.append(f"berlokasi di {item.get('district')}")
        if item.get("rating"):
            parts.append(f"dengan rating {item.get('rating')}/5")
        desc = ". ".join(parts) + "."

    est_cost = item.get("estimated_cost_idr")
    if est_cost is None:
        if category == "restaurant":
            pl = item.get("price_level") or meta.get("price_level") or meta.get("priceLevel") or "PRICE_LEVEL_MODERATE"
            price_map = {
                "PRICE_LEVEL_FREE": 0,
                "PRICE_LEVEL_INEXPENSIVE": 50000,
                "PRICE_LEVEL_MODERATE": 100000,
                "PRICE_LEVEL_EXPENSIVE": 250000,
                "PRICE_LEVEL_VERY_EXPENSIVE": 500000,
            }
            if isinstance(pl, int):
                price_map_num = {0: 0, 1: 50000, 2: 120000, 3: 250000, 4: 500000}
                est_cost = price_map_num.get(pl, 120000)
            else:
                est_cost = price_map.get(pl, 120000)
        else:
            pl = item.get("price_level") or meta.get("price_level") or meta.get("priceLevel")
            if pl:
                price_map = {
                    "PRICE_LEVEL_FREE": 0,
                    "PRICE_LEVEL_INEXPENSIVE": 15000,
                    "PRICE_LEVEL_MODERATE": 35000,
                    "PRICE_LEVEL_EXPENSIVE": 75000,
                    "PRICE_LEVEL_VERY_EXPENSIVE": 150000,
                }
                est_cost = price_map.get(pl, 25000)
            else:
                est_cost = 25000

    tags = item.get("tags") or meta.get("tags")
    if not tags:
        if category == "restaurant":
            tags = ["kuliner", "makanan", "restoran"]
            if item.get("district"):
                tags.append(item.get("district").lower())
        else:
            tags = ["wisata", "atraksi", "populer"]
            if item.get("district"):
                tags.append(item.get("district").lower())
    elif isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    tips = item.get("tips") or meta.get("tips")
    if not tips:
        if category == "restaurant":
            tips = f"Coba menu andalan di {item.get('name')} untuk pengalaman kuliner terbaik."
        else:
            tips = f"Bawa kamera terbaikmu untuk mengabadikan momen indah di {item.get('name')}."

    return {
        "place_id": item.get("place_id"),
        "poi_id": str(item.get("id")) if item.get("id") is not None else (str(item.get("poi_id")) if item.get("poi_id") is not None else None),
        "name": item.get("name"),
        "category": "restaurant" if category == "restaurant" else "attraction",
        "description": desc,
        "latitude": item.get("latitude"),
        "longitude": item.get("longitude"),
        "district": item.get("district"),
        "image_url": item.get("image_url"),
        "rating": item.get("rating"),
        "user_rating_count": item.get("user_rating_count"),
        "estimated_cost_idr": est_cost,
        "tags": tags,
        "visit_duration_mins": visit_duration,
        "visit_time": visit_time,
        "opening_hours_note": item.get("opening_hours_note") or meta.get("opening_hours_note") or meta.get("openingHours"),
        "tips": tips,
        "route_to_next": None
    }


def build_deterministic_itinerary(daily_pools: list[dict], user_requested_pois: Optional[dict] = None) -> list[dict]:
    itinerary_days = []
    
    for day_pool in daily_pools:
        day_num = day_pool["day"]
        pois = day_pool["pois"]
        restaurants = day_pool["restaurants"]
        
        # 1. Determine target count of attractions
        target_count = 3
        if isinstance(user_requested_pois, dict):
            target_count = user_requested_pois.get(str(day_num), 3)
        try:
            target_count = max(1, min(int(target_count), 7))
        except (ValueError, TypeError):
            target_count = 3
            
        selected_pois = pois[:target_count]
        
        # 2. Select restaurants
        if not restaurants:
            # Fallback placeholder restaurant
            fallback_resto = {
                "place_id": None,
                "name": "Restoran Lokal Bali",
                "latitude": day_pool["anchor"]["lat"],
                "longitude": day_pool["anchor"]["lng"],
                "district": day_pool["anchor"].get("name", "Bali"),
                "rating": 4.5,
                "content": "Menyajikan hidangan lokal khas Bali yang lezat dengan bahan segar.",
                "metadata": {"price_level": "PRICE_LEVEL_MODERATE"}
            }
            restaurants = [fallback_resto, fallback_resto]
            
        lunch_resto = restaurants[0]
        dinner_resto = restaurants[1] if len(restaurants) > 1 else restaurants[0]
        
        # 3. Schedule places chronologically
        places = []
        if len(selected_pois) == 0:
            # No POIs, should not happen, but safeguard
            # Just add lunch and dinner
            l_item = map_to_place_item(lunch_resto, "restaurant", "12:00", 60)
            d_item = map_to_place_item(dinner_resto, "restaurant", "18:30", 60)
            places.extend([l_item, d_item])
        elif len(selected_pois) == 1:
            p1 = map_to_place_item(selected_pois[0], "attraction", "09:30", 90)
            l_item = map_to_place_item(lunch_resto, "restaurant", "12:00", 60)
            d_item = map_to_place_item(dinner_resto, "restaurant", "18:30", 60)
            places.extend([p1, l_item, d_item])
        elif len(selected_pois) == 2:
            p1 = map_to_place_item(selected_pois[0], "attraction", "09:00", 90)
            l_item = map_to_place_item(lunch_resto, "restaurant", "12:00", 60)
            p2 = map_to_place_item(selected_pois[1], "attraction", "14:30", 90)
            d_item = map_to_place_item(dinner_resto, "restaurant", "18:30", 60)
            places.extend([p1, l_item, p2, d_item])
        else:
            # 3 or more POIs
            p1 = map_to_place_item(selected_pois[0], "attraction", "08:30", 90)
            p2 = map_to_place_item(selected_pois[1], "attraction", "10:30", 60)
            places.extend([p1, p2])
            
            l_item = map_to_place_item(lunch_resto, "restaurant", "12:00", 60)
            places.append(l_item)
            
            afternoon_pois = selected_pois[2:]
            num_aft = len(afternoon_pois)
            for idx, poi in enumerate(afternoon_pois):
                start_total_mins = 13 * 60 + 30 + int(idx * (270 / num_aft))
                h = start_total_mins // 60
                m = start_total_mins % 60
                v_time = f"{h:02d}:{m:02d}"
                v_dur = max(45, min(90, int(270 / num_aft) - 15))
                p_aft = map_to_place_item(poi, "attraction", v_time, v_dur)
                places.append(p_aft)
                
            d_item = map_to_place_item(dinner_resto, "restaurant", "18:30", 60)
            places.append(d_item)
            
        # Determine theme based on attractions
        districts = []
        for p in places:
            if p.get("category") == "attraction" and p.get("district"):
                if p["district"] not in districts:
                    districts.append(p["district"])
        if districts:
            theme = f"Eksplorasi {', '.join(districts[:2])}"
        else:
            theme = f"Petualangan Menarik Hari {day_num}"
            
        itinerary_days.append({
            "day": day_num,
            "theme": theme,
            "places": places,
            "day_total_distance_km": None,
            "day_total_travel_time_mins": None,
            "day_full_polyline": None,
            "route_from_hotel": None,
            "odalan_warning": None,
            "weather_note": None
        })
        
    return itinerary_days