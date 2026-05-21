# app/engine/recommender.py
from app.services.tomtom_service import _haversine_km
import numpy as np
from sklearn.cluster import DBSCAN
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# BALI GEOGRAPHIC CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────
# Bali island bounding box (slightly expanded to include Nusa Penida etc.)
BALI_BBOX = {
    "lat_min": -8.90,
    "lat_max": -8.05,
    "lng_min": 114.35,
    "lng_max": 115.75,
}

# Maximum DBSCAN cluster radius (km).  A single day-trip cluster should never
# exceed this — it corresponds to ~40 min driving at Bali traffic speeds.
MAX_CLUSTER_RADIUS_KM = {
    "poi":        20.0,   # Attractions: tight, 1-day area
    "restaurant": 20.0,   # Restaurants: injected near attraction centroid anyway
    "hotel":      25.0,   # Hotels: slight slack — might be a hub for 2+ days
}

# Any individual POI further than this from its cluster centroid is ejected.
MAX_POI_DISTANCE_FROM_CENTROID_KM = 20.0

# KONFIGURASI TOPSIS PER KATEGORI
# Setiap kategori memiliki daftar fitur, bobot (weights), dan dampak
# (impacts: 1 = benefit/makin tinggi makin baik, -1 = cost/makin rendah makin baik)
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
    "rating":0.0,
    "popularity":0.0,
    "price_value":0.0,
    "strategic_score":0.0,
    "visual_interest_index":0.0,
    "comfort_index":0.0,
    "amenity_density":0.0,
    "accessibility_index":0.0,
    "menu_variety_index":0.0,
    "ambience_score":0.0,
    "payment_modern_index":0.0,
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


def _filter_bali_bbox(pois: list) -> list:
    """
    Drop any POI whose coordinates fall outside Bali's bounding box.
    This removes stray data points that would skew the DBSCAN radius calculation.
    """
    filtered = [
        p for p in pois
        if (
            p.get("latitude") is not None
            and p.get("longitude") is not None
            and BALI_BBOX["lat_min"] <= p["latitude"] <= BALI_BBOX["lat_max"]
            and BALI_BBOX["lng_min"] <= p["longitude"] <= BALI_BBOX["lng_max"]
        )
    ]
    removed = len(pois) - len(filtered)
    if removed > 0:
        logger.warning(f"_filter_bali_bbox: removed {removed} POIs outside Bali bbox.")
    return filtered


def cluster_and_rank_pois(
    pois: list,
    num_clusters: int = 1,
    top_n_per_cluster: int = 3,
    category: Literal["poi", "hotel", "restaurant"] = "poi",
    dbscan_radius_km: float = None,
    preference_mode: str = "standard",
) -> list:
    """
    Mengelompokkan POI secara spasial (DBSCAN) dan memilih yang terbaik
    menggunakan TOPSIS multi-dimensi sesuai kategori.

    Args:
        pois: List dict hasil query database
        num_clusters: Jumlah cluster (= jumlah hari perjalanan)
        top_n_per_cluster: Tempat terbaik yang dipilih per cluster
        category: 'poi', 'hotel', atau 'restaurant' — menentukan bobot TOPSIS
        dbscan_radius_km: Radius DBSCAN dalam km. If None, auto-calculated with a
                          hard cap of MAX_CLUSTER_RADIUS_KM[category].
    
    Returns:
        List tempat terbaik yang sudah diurutkan per cluster
    """
    # ── Step 0: Bali bbox pre-filter ─────────────────────────────────────────
    pois = _filter_bali_bbox(pois)

    if not pois:
        return []

    # ── Step 1: Determine DBSCAN radius ──────────────────────────────────────
    max_radius = MAX_CLUSTER_RADIUS_KM.get(category, 20.0)

    if dbscan_radius_km is None:
        if len(pois) > 1:
            lats = [p["latitude"] for p in pois if p.get("latitude")]
            lons = [p["longitude"] for p in pois if p.get("longitude")]
            spread_km = _haversine_km(min(lats), min(lons), max(lats), max(lons))
            # Dynamic formula but hard-capped: never allow a cluster to span more
            # than MAX_CLUSTER_RADIUS_KM — otherwise distant corners of Bali can
            # end up in the same cluster and produce 80+ km day itineraries.
            dynamic = spread_km / (num_clusters * 1.5)
            dbscan_radius_km = min(max(3.0, dynamic), max_radius)
        else:
            dbscan_radius_km = 5.0

    # Always enforce the hard cap even when caller passes a value explicitly.
    dbscan_radius_km = min(dbscan_radius_km, max_radius)
    logger.info(f"cluster_and_rank_pois [{category}]: DBSCAN radius = {dbscan_radius_km:.1f} km")

    config = TOPSIS_CONFIG.get(category, TOPSIS_CONFIG["poi"])

    import copy
    config = copy.deepcopy(config) 

    feature_keys = config["features"]

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

    feature_keys = config["features"]
    weights = np.array(config["weights"])
    impacts = np.array(config["impacts"])


    # 1. Ekstraksi Fitur & Filter Validitas

    coords, feature_matrix_rows, valid_pois = [], [], []

    for p in pois:
        lat = p.get("latitude")
        lon = p.get("longitude")
        if lat is None or lon is None:
            continue

        # TC-06: extract_topsis_features tidak boleh crash meski metadata kosong
        try:
            features = extract_topsis_features(p, category)
            feature_row = [features.get(k, DEFAULT_FEATURE_VALUES.get(k, 0.5)) for k in feature_keys]
        except Exception as e:
            logger.warning(f"Gagal ekstrak fitur untuk '{p.get('name', 'unknown')}': {e}. Pakai default.")
            feature_row = [DEFAULT_FEATURE_VALUES.get(k, 0.5) for k in feature_keys]

        coords.append([lat, lon])
        feature_matrix_rows.append(feature_row)
        valid_pois.append(p)

    if not valid_pois:
        logger.warning("Tidak ada POI dengan koordinat valid. Return raw pois[:limit].")
        return pois[: top_n_per_cluster * num_clusters]


    # 2. TOPSIS Scoring (hitung sebelum clustering agar bisa fallback)

    data_matrix = np.array(feature_matrix_rows, dtype=float)

    try:
        scores = topsis_score(data_matrix, weights, impacts)
    except Exception as e:
        logger.error(f"TOPSIS scoring error: {e}. Pakai score=1.0 untuk semua.")
        scores = np.ones(len(valid_pois))

    for i, p in enumerate(valid_pois):
        p["topsis_score"] = round(float(scores[i]), 4)
        p["topsis_category"] = category


    # 3. Clustering Spasial (DBSCAN Haversine)

    coords_rad = np.radians(coords)
    eps_rad = dbscan_radius_km / 6371.0  # Konversi km ke radian (radius bumi 6371 km)

    dbscan = DBSCAN(eps=eps_rad, min_samples=1, algorithm="ball_tree", metric="haversine")
    cluster_labels = dbscan.fit_predict(coords_rad)


    # 4. Grouping & Fallback

    clustered_result: dict[int, list] = {}
    outliers = []

    for i, p in enumerate(valid_pois):
        c_id = int(cluster_labels[i])
        if c_id == -1:
            outliers.append(p)  # Tandai sebagai outlier
        else:
            p["cluster_id"] = c_id
            clustered_result.setdefault(c_id, []).append(p)

    # Fallback 1: Semua outlier → coba dengan radius lebih besar (capped)
    if not clustered_result and outliers:
        fallback_r = min(max_radius, dbscan_radius_km * 1.5)
        logger.warning(f"DBSCAN: Semua titik adalah outlier. Mencoba radius {fallback_r:.1f}km...")
        eps_fallback = fallback_r / 6371.0
        dbscan_fallback = DBSCAN(eps=eps_fallback, min_samples=1, algorithm="ball_tree", metric="haversine")
        labels_fallback = dbscan_fallback.fit_predict(coords_rad)
        for i, p in enumerate(valid_pois):
            c_id = int(labels_fallback[i])
            if c_id != -1:
                p["cluster_id"] = c_id
                clustered_result.setdefault(c_id, []).append(p)

    # Fallback 2: Masih kosong → kembalikan top N berdasarkan TOPSIS
    if not clustered_result:
        logger.warning("DBSCAN Fallback: Tetap tidak ada cluster. Return top TOPSIS.")
        valid_pois_sorted = sorted(valid_pois, key=lambda x: x.get("topsis_score", 0), reverse=True)
        return valid_pois_sorted[: top_n_per_cluster * num_clusters]


    # ── Step 5: Iterative centroid distance filter ────────────────────────────
    # DBSCAN with a radius cap can still produce elongated "chain" clusters
    # (A→B→C each within 20km of the next, but A and C are 40km apart).
    # We run the centroid filter up to MAX_ITER times: each pass recomputes the
    # centroid from the surviving points and ejects any point > threshold away.
    # Iteration stops when no more points are ejected (convergence).
    MAX_CENTROID_ITER = 3
    for _pass in range(MAX_CENTROID_ITER):
        filtered_clusters: dict[int, list] = {}
        total_ejected_this_pass = 0

        for c_id, group in clustered_result.items():
            if len(group) <= 1:
                filtered_clusters[c_id] = group
                continue

            c_lats = [p["latitude"] for p in group if p.get("latitude")]
            c_lons = [p["longitude"] for p in group if p.get("longitude")]
            centroid_lat = sum(c_lats) / len(c_lats)
            centroid_lon = sum(c_lons) / len(c_lons)

            kept = [
                p for p in group
                if p.get("latitude") and p.get("longitude")
                and _haversine_km(p["latitude"], p["longitude"], centroid_lat, centroid_lon)
                <= MAX_POI_DISTANCE_FROM_CENTROID_KM
            ]
            if not kept:
                kept = group  # Safety: never empty a cluster entirely
            total_ejected_this_pass += len(group) - len(kept)
            filtered_clusters[c_id] = kept

        clustered_result = filtered_clusters
        if total_ejected_this_pass == 0:
            break  # Converged
        logger.info(
            f"cluster_and_rank_pois [{category}]: centroid pass {_pass+1} "
            f"ejected {total_ejected_this_pass} far POIs"
        )

    # ── Step 6: Pick top-N clusters by average TOPSIS score ───────────────────
    cluster_scores = []
    for c_id, group in clustered_result.items():
        avg_score = sum(p.get("topsis_score", 0) for p in group) / len(group)
        cluster_scores.append((avg_score, c_id))

    # Urutkan cluster dari yang rata-rata terbaiknya paling tinggi
    cluster_scores.sort(key=lambda x: x[0], reverse=True)
    best_cluster_ids = [c_id for _, c_id in cluster_scores[:num_clusters]]


    # ── Step 7: Take top-N per cluster ───────────────────────────────────────
    final_list = []
    for c_id in best_cluster_ids:
        sorted_group = sorted(
            clustered_result[c_id],
            key=lambda x: x.get("topsis_score", 0),
            reverse=True,
        )
        final_list.extend(sorted_group[:top_n_per_cluster])

    logger.info(
        f"cluster_and_rank_pois [{category}]: {len(pois)} raw → "
        f"{len(final_list)} selected from {len(clustered_result)} clusters "
        f"(radius={dbscan_radius_km:.1f} km)"
    )
    return final_list