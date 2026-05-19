from pydantic import BaseModel
from typing import List, Optional

class OdalanCheckResult(BaseModel):
    status: str  # "CLEAR", "AFFECTED" (macet/ramai), "BLOCKED" (tutup)
    poi_id: Optional[str] = None
    poi_name: Optional[str] = None
    message: str

def evaluate_odalan_status(poi_id: str, active_odalans: list[dict]) -> OdalanCheckResult:
    """Mengecek status spesifik sebuah POI apakah terblokir Odalan."""
    for odalan in active_odalans:
        if odalan.get("poi_attraction_id") == poi_id:
            location_name = odalan.get("location_name", "Tempat ini")
            return OdalanCheckResult(
                status="BLOCKED",
                poi_id=poi_id,
                poi_name=location_name,
                message=f"{location_name} sedang ada upacara adat dan kemungkinan tertutup atau sangat padat."
            )
            
    return OdalanCheckResult(status="CLEAR", message="Aman dari Odalan.")

def extract_global_avoid_zones(active_odalans: List[dict]) -> List[str]:
    """
    Mengekstrak SEMUA bounding box dari daftar upacara yang aktif di tanggal tersebut
    untuk dikirimkan ke TomTom sebagai zona hindari (avoid_areas) global.
    """
    avoid_zones = []
    # Buffer ~100 meter agar rute benar-benar menjauh dari pusat keramaian
    BUFFER = 0.001 

    for odalan in active_odalans:
        sw_lat = odalan.get("south_west_latitude")
        sw_lng = odalan.get("south_west_longitude")
        ne_lat = odalan.get("north_east_latitude")
        ne_lng = odalan.get("north_east_longitude")

        # Pastikan data koordinatnya lengkap di database
        if sw_lat is not None and sw_lng is not None and ne_lat is not None and ne_lng is not None:
            # Format TomTom: minLon, minLat, maxLon, maxLat
            min_lon = sw_lng - BUFFER
            min_lat = sw_lat - BUFFER
            max_lon = ne_lng + BUFFER
            max_lat = ne_lat + BUFFER
            
            bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
            avoid_zones.append(bbox_str)

    return avoid_zones