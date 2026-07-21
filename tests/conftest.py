import os

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_CLOUD_VISION_API_KEY", "test-key")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")
os.environ.setdefault("CONCURRENCY_LIMIT", "4")

from app.config import get_settings  # noqa: E402
from app.security import limiter  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    limiter.reset()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():
    return get_settings()
