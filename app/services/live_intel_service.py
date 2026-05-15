import httpx
from app.core.config import settings
from app.core.resilience import retry_with_backoff

class LiveIntelService:
    @retry_with_backoff(retries=2)
    async def get_weather_forecast(self, lat: float, lon: float) -> dict:
        """Cuaca 4 hari ke depan dari OpenWeatherMap."""
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": settings.openweathermap_api_key,
            "units": "metric",
            "cnt": 32 # Data per 3 jam, 32 = 4 hari
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    @retry_with_backoff(retries=2)
    async def search_tavily(self, query: str) -> str:
        """Berita/event real-time dari Tavily."""
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 2
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return "\n".join([f"- {res['content']}" for res in data.get('results', [])])

live_intel_service = LiveIntelService()