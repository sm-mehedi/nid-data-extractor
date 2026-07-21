import asyncio
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

settings = get_settings()

limiter = Limiter(key_func=get_remote_address, default_limits=[])

_concurrency_semaphore = asyncio.Semaphore(settings.concurrency_limit)


@asynccontextmanager
async def concurrency_guard():
    try:
        await asyncio.wait_for(_concurrency_semaphore.acquire(), timeout=0.01)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server is at capacity, please retry shortly.",
        )
    try:
        yield
    finally:
        _concurrency_semaphore.release()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'"
        )
        return response


def verify_shared_secret(request: Request) -> None:
    if not settings.api_shared_secret:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != settings.api_shared_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
