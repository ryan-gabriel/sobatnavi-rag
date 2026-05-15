# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI — required at runtime, optional at startup
    openai_api_key: str = ""
    openai_model_id: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Supabase — wajib ada
    supabase_url: str
    supabase_service_key: str

    # Third-party APIs — optional (fitur terkait akan fallback jika kosong)
    tomtom_api_key: str = ""
    tavily_api_key: str = ""
    openweathermap_api_key: str = ""

    # App
    log_level: str = "INFO"

settings = Settings()