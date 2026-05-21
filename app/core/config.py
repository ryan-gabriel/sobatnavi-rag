# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI — required at runtime, optional at startup
    openai_api_key: str = ""
    openai_model_id: str = "gpt-4.1-nano"
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

    # Security — CORS
    # Comma-separated list of allowed origins. Use "*" for development only.
    # Example: "https://app.sobatnavi.com,https://admin.sobatnavi.com"
    allowed_origins: str = "*"

    # Security — Trusted Hosts
    # Comma-separated list of allowed Host header values.
    # Example: "sobatnavi.com,api.sobatnavi.com,localhost"
    allowed_hosts: str = "*"

    # Security — Rate Limiting
    # Set to "false" to disable rate limiting (useful in tests / local dev)
    rate_limit_enabled: bool = True

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def allowed_hosts_list(self) -> List[str]:
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]

settings = Settings()
