from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_cloud_vision_api_key: str = ""
    google_application_credentials: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    max_upload_mb: int = 10
    rate_limit_per_minute: int = 8
    concurrency_limit: int = 4

    api_shared_secret: str = ""

    port: int = 8000

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
