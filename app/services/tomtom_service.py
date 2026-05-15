# app/services/tomtom_service.py
import httpx
import math
import asyncio
import logging
from typing import Optional
from app.core.config import settings
from app.core.resilience import retry_with_backoff

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Kalkulasi jarak Haversine antara 2 titik koordinat."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_tomtom_route(data: dict) -> dict:
    """
    Mem-parse respons JSON TomTom menjadi format terstandar yang siap
    dikonsumsi oleh frontend (polyline dalam format [{lat, lng}]).
    """
    route = data["routes"][0]
    summary = route["summary"]

    # Kumpulkan semua titik koordinat dari semua legs
    polyline_points = []
    for leg in route.get("legs", []):
        for point in leg.get("points", []):
            polyline_points.append({
                "lat": round(point["latitude"], 6),
                "lng": round(point["longitude"], 6),
            })

    return {
        "distance_km": round(summary["lengthInMeters"] / 1000, 2),
        "travel_time_mins": math.ceil(summary["travelTimeInSeconds"] / 60),
        "traffic_delay_mins": math.ceil(summary.get("trafficDelayInSeconds", 0) / 60),
        "polyline": polyline_points,  # Untuk frontend rendering (Leaflet/Google Maps)
        "status": "OK",
    }


class TomTomService:

    @retry_with_backoff(retries=2)
    async def calculate_route(
        self,
        origin: str,           # Format: "lat,lng"
        dest: str,             # Format: "lat,lng"
        avoid_areas: Optional[list[str]] = None,
        include_polyline: bool = True,
    ) -> dict:
        """
        Menghitung rute point-to-point via TomTom Routing API.

        Returns dict dengan keys:
            - distance_km (float)
            - travel_time_mins (int)
            - traffic_delay_mins (int)
            - polyline (list of {lat, lng}) — untuk render rute di peta
            - status (str): "OK" atau "DEGRADED (Estimasi)"
        """
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin}:{dest}/json"
        params = {
            "key": settings.tomtom_api_key,
            "routeType": "fastest",
            "traffic": "true",
        }

        # Sisipkan zona hindari Odalan jika ada
        if avoid_areas:
            # Format TomTom avoidAreas: bbox1~bbox2~...
            # Setiap bbox: minLon,minLat,maxLon,maxLat
            formatted_areas = []
            for area in avoid_areas:
                # area sudah dalam format: "minLon,minLat,maxLon,maxLat"
                parts = area.split(",")
                if len(parts) == 4:
                    formatted_areas.append(
                        f"{{{parts[1]},{parts[0]},{parts[3]},{parts[2]}}}"
                    )
            if formatted_areas:
                params["avoidAreas"] = "~".join(formatted_areas)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("routes"):
                    raise ValueError("TomTom API mengembalikan routes kosong")

                return _parse_tomtom_route(data)

        except httpx.HTTPStatusError as e:
            logger.error(f"TomTom HTTP Error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"TomTom route error ({origin} → {dest}): {e}")
            raise

    async def get_fallback_route(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> dict:
        """
        Fallback Haversine jika TomTom API tidak tersedia.
        Menghasilkan polyline lurus antara 2 titik (estimasi).
        """
        distance_km = _haversine_km(lat1, lon1, lat2, lon2)
        # Estimasi kecepatan rata-rata 30 km/jam di Bali
        travel_time_mins = math.ceil((distance_km / 30) * 60)

        return {
            "distance_km": round(distance_km, 2),
            "travel_time_mins": travel_time_mins,
            "traffic_delay_mins": 0,
            "polyline": [
                {"lat": lat1, "lng": lon1},
                {"lat": lat2, "lng": lon2},
            ],
            "status": "DEGRADED (Estimasi Haversine)",
        }

    async def calculate_batch_routes(
        self,
        waypoints: list[dict],   # [{lat, lng}, {lat, lng}, ...]
        avoid_areas: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Menghitung banyak rute secara paralel (satu panggilan per segmen).
        Satu panggilan untuk seluruh hari perjalanan (TC-08).

        Returns:
            List hasil rute per segmen (panjang = len(waypoints) - 1).
            Setiap item: {distance_km, travel_time_mins, traffic_delay_mins, polyline, status, from_index, to_index}
        """
        if len(waypoints) < 2:
            logger.warning("calculate_batch_routes: butuh minimal 2 waypoints.")
            return []

        tasks = []
        for i in range(len(waypoints) - 1):
            origin = f"{waypoints[i]['lat']},{waypoints[i]['lng']}"
            dest = f"{waypoints[i+1]['lat']},{waypoints[i+1]['lng']}"
            tasks.append(self.calculate_route(origin, dest, avoid_areas))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, res in enumerate(results):
            from_wp = waypoints[i]
            to_wp = waypoints[i + 1]

            if isinstance(res, Exception):
                logger.warning(
                    f"Batch route [{i}→{i+1}] gagal: {res}. Pakai fallback Haversine."
                )
                fallback = await self.get_fallback_route(
                    from_wp["lat"], from_wp["lng"], to_wp["lat"], to_wp["lng"]
                )
                fallback["from_index"] = i
                fallback["to_index"] = i + 1
                final_results.append(fallback)
            else:
                res["from_index"] = i
                res["to_index"] = i + 1
                final_results.append(res)

        logger.info(
            f"calculate_batch_routes: {len(waypoints)} waypoints → {len(final_results)} segmen"
        )
        return final_results

    async def get_full_day_route(
        self,
        waypoints: list[dict],   # Hotel → POI1 → POI2 → ... → POI_n → Hotel
        avoid_areas: Optional[list[str]] = None,
    ) -> dict:
        """
        Menghitung SATU rute penuh untuk seluruh perjalanan satu hari.
        Cocok untuk di-overlay di peta sebagai polyline keseluruhan hari.

        Returns:
            {
              "total_distance_km": float,
              "total_travel_time_mins": int,
              "segments": [...],         # per segmen
              "full_day_polyline": [...] # gabungan semua titik dalam urutan kunjungan
            }
        """
        segments = await self.calculate_batch_routes(waypoints, avoid_areas)

        total_distance = sum(s.get("distance_km", 0) for s in segments)
        total_time = sum(s.get("travel_time_mins", 0) for s in segments)

        # Gabungkan semua polyline menjadi satu jalur penuh (hindari duplikat titik)
        full_polyline = []
        for seg in segments:
            seg_poly = seg.get("polyline", [])
            if full_polyline and seg_poly:
                full_polyline.extend(seg_poly[1:])  # Skip titik pertama (duplikat)
            else:
                full_polyline.extend(seg_poly)

        return {
            "total_distance_km": round(total_distance, 2),
            "total_travel_time_mins": total_time,
            "segments": segments,
            "full_day_polyline": full_polyline,
        }


tomtom_service = TomTomService()