from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_cloud_vision_api_key: str = ""
    google_application_credentials: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    # Empty = auto-derive "<gemini_model>-lite" (e.g. "gemini-3.5-flash" ->
    # "gemini-3.5-flash-lite"); set explicitly to use a different fallback.
    # Only used as a single extra attempt after the primary model exhausts
    # all its retries with a 503 (overloaded) — different model variants
    # often have separate capacity pools.
    gemini_fallback_model: str = ""
    gemini_timeout_seconds: int = 100
    # Comma-separated seconds to wait before each retry, e.g. "3,8,15" means
    # up to 3 retries (4 total attempts) with escalating backoff. Applies to
    # 429 (rate limit), 503 (overloaded), and timeout/504 (deadline exceeded).
    gemini_retry_backoff_seconds: str = "3,8,15"

    max_upload_mb: int = 10
    rate_limit_per_minute: int = 8
    concurrency_limit: int = 4

    api_shared_secret: str = ""

    port: int = 8000

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def gemini_retry_backoff_schedule(self) -> list[float]:
        return [float(part.strip()) for part in self.gemini_retry_backoff_seconds.split(",") if part.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
