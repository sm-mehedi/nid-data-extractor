from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.models import ExtractResponse, HealthResponse
from app.routes.nid import router as nid_router
from app.security import SecurityHeadersMiddleware, limiter

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Bangladesh NID Extractor",
    description="Extracts structured, English-translated data from Bangladesh NID card photos.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_middleware(SecurityHeadersMiddleware)
# Cloud Run's front-end proxy is the only thing that can ever reach this
# container directly, so trusting all inbound connections' X-Forwarded-For
# is safe here. Without this, the per-IP rate limiter (keyed on
# request.client.host) sees Cloud Run's proxy connection instead of the real
# end-user's IP — added in code (not just the Dockerfile's uvicorn flags) so
# it applies regardless of how the ASGI app is invoked, including in tests.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    body = ExtractResponse(
        success=False, data=None, warnings=[], errors=["Rate limit exceeded, please retry shortly."]
    )
    return JSONResponse(status_code=429, content=body.model_dump())


app.include_router(nid_router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if (_FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
