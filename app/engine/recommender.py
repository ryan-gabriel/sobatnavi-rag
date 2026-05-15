# app/engine/recommender.py
import numpy as np
from sklearn.cluster import DBSCAN
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# =====================================================================
# KONFIGURASI TOPSIS PER KATEGORI
# Setiap kategori memiliki daftar fitur, bobot (weights), dan dampak
# (impacts: 1 = benefit/makin tinggi makin baik, -1 = cost/makin rendah makin baik)
# =====================================================================
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


def cluster_and_rank_pois(
    pois: list,
    num_clusters: int = 1,
    top_n_per_cluster: int = 3,
    category: Literal["poi", "hotel", "restaurant"] = "poi",
    dbscan_radius_km: float = 15.0,
) -> list:
    """
    Mengelompokkan POI secara spasial (DBSCAN) dan memilih yang terbaik
    menggunakan TOPSIS multi-dimensi sesuai kategori.

    Args:
        pois: List dict hasil query database
        num_clusters: Jumlah cluster (= jumlah hari perjalanan)
        top_n_per_cluster: Tempat terbaik yang dipilih per cluster
        category: 'poi', 'hotel', atau 'restaurant' — menentukan bobot TOPSIS
        dbscan_radius_km: Radius DBSCAN dalam km (default 15km)
    
    Returns:
        List tempat terbaik yang sudah diurutkan per cluster
    """
    if not pois:
        return []

    config = TOPSIS_CONFIG.get(category, TOPSIS_CONFIG["poi"])
    feature_keys = config["features"]
    weights = np.array(config["weights"])
    impacts = np.array(config["impacts"])

    # =====================================================================
    # 1. Ekstraksi Fitur & Filter Validitas
    # =====================================================================
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

    # =====================================================================
    # 2. TOPSIS Scoring (hitung sebelum clustering agar bisa fallback)
    # =====================================================================
    data_matrix = np.array(feature_matrix_rows, dtype=float)

    try:
        scores = topsis_score(data_matrix, weights, impacts)
    except Exception as e:
        logger.error(f"TOPSIS scoring error: {e}. Pakai score=1.0 untuk semua.")
        scores = np.ones(len(valid_pois))

    for i, p in enumerate(valid_pois):
        p["topsis_score"] = round(float(scores[i]), 4)
        p["topsis_category"] = category

    # =====================================================================
    # 3. Clustering Spasial (DBSCAN Haversine)
    # =====================================================================
    coords_rad = np.radians(coords)
    eps_rad = dbscan_radius_km / 6371.0  # Konversi km ke radian (radius bumi 6371 km)

    dbscan = DBSCAN(eps=eps_rad, min_samples=2, algorithm="ball_tree", metric="haversine")
    cluster_labels = dbscan.fit_predict(coords_rad)

    # =====================================================================
    # 4. Grouping & Fallback
    # =====================================================================
    clustered_result: dict[int, list] = {}
    outliers = []

    for i, p in enumerate(valid_pois):
        c_id = int(cluster_labels[i])
        if c_id == -1:
            outliers.append(p)  # Tandai sebagai outlier
        else:
            p["cluster_id"] = c_id
            clustered_result.setdefault(c_id, []).append(p)

    # Fallback 1: Semua outlier → coba dengan radius lebih besar (25km)
    if not clustered_result and outliers:
        logger.warning("DBSCAN: Semua titik adalah outlier. Mencoba radius 25km...")
        eps_fallback = 25.0 / 6371.0
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

    # =====================================================================
    # 5. Pilih Top-N Cluster Berdasarkan Skor TOPSIS Rata-rata
    # =====================================================================
    cluster_scores = []
    for c_id, group in clustered_result.items():
        avg_score = sum(p.get("topsis_score", 0) for p in group) / len(group)
        cluster_scores.append((avg_score, c_id))

    # Urutkan cluster dari yang rata-rata terbaiknya paling tinggi
    cluster_scores.sort(key=lambda x: x[0], reverse=True)
    best_cluster_ids = [c_id for _, c_id in cluster_scores[:num_clusters]]

    # =====================================================================
    # 6. Ambil Top-N per Cluster
    # =====================================================================
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
        f"{len(final_list)} terpilih dari {len(clustered_result)} cluster"
    )
    return final_list