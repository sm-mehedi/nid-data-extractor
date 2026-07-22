import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.helpers import clear_card_image, encode_jpg


def _files():
    img = encode_jpg(clear_card_image())
    return {
        "front_image": ("front.jpg", img, "image/jpeg"),
        "back_image": ("back.jpg", img, "image/jpeg"),
    }


def test_rate_limiter_triggers_past_threshold(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    get_settings.cache_clear()

    from app.main import app

    client = TestClient(app)
    statuses = [client.post("/api/v1/nid/extract", files={}).status_code for _ in range(3)]

    get_settings.cache_clear()

    # First 2 requests are under the cap (they still fail with 400 for missing
    # files, but that's a *validation* failure, not a rate-limit one); the 3rd
    # must be rejected by the limiter itself.
    assert statuses[:2] == [400, 400]
    assert statuses[2] == 429


def test_rate_limiter_response_shape_on_429(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    get_settings.cache_clear()

    from app.main import app

    client = TestClient(app)
    client.post("/api/v1/nid/extract", files={})
    resp = client.post("/api/v1/nid/extract", files={})

    get_settings.cache_clear()

    assert resp.status_code == 429
    body = resp.json()
    assert body["success"] is False
    assert body["data"] is None
    assert len(body["errors"]) >= 1


@pytest.mark.asyncio
async def test_concurrency_cap_returns_503_for_excess_requests(monkeypatch):
    import app.security as security_module
    from app.main import app

    monkeypatch.setattr(security_module, "_concurrency_semaphore", asyncio.Semaphore(1))

    def slow_extract(*args, **kwargs):
        time.sleep(0.3)
        from app.models import ExtractResponse, NidData

        return ExtractResponse(success=True, data=NidData(name="X"), warnings=[], errors=[])

    monkeypatch.setattr("app.routes.nid.pipeline.extract_nid", slow_extract)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        results = await asyncio.gather(
            client.post("/api/v1/nid/extract", files=_files()),
            client.post("/api/v1/nid/extract", files=_files()),
        )

    statuses = sorted(r.status_code for r in results)
    assert statuses == [200, 503]

    rejected = next(r for r in results if r.status_code == 503)
    assert rejected.json()["success"] is False


@pytest.mark.asyncio
async def test_concurrency_within_cap_both_succeed(monkeypatch):
    import app.security as security_module
    from app.main import app

    monkeypatch.setattr(security_module, "_concurrency_semaphore", asyncio.Semaphore(2))

    def fast_extract(*args, **kwargs):
        from app.models import ExtractResponse, NidData

        return ExtractResponse(success=True, data=NidData(name="X"), warnings=[], errors=[])

    monkeypatch.setattr("app.routes.nid.pipeline.extract_nid", fast_extract)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        results = await asyncio.gather(
            client.post("/api/v1/nid/extract", files=_files()),
            client.post("/api/v1/nid/extract", files=_files()),
        )

    assert all(r.status_code == 200 for r in results)


@pytest.mark.asyncio
async def test_rate_limiter_distinguishes_real_client_ip_behind_proxy(monkeypatch):
    # Regression test for the Cloud Run proxy-header gap: without
    # ProxyHeadersMiddleware (or the Dockerfile's --forwarded-allow-ips),
    # request.client.host reflects Cloud Run's own proxy connection, not the
    # real end-user's IP, so two different real users would incorrectly
    # share one rate-limit bucket. This sends X-Forwarded-For headers for two
    # distinct "real" client IPs through the actual app object (exercising
    # its real middleware stack, not just the route logic) and confirms each
    # gets its own independent budget rather than a global/shared one.
    from app.config import get_settings

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    get_settings.cache_clear()

    from app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        statuses_ip1 = []
        for _ in range(3):
            resp = await client.post(
                "/api/v1/nid/extract", files={}, headers={"X-Forwarded-For": "203.0.113.10"}
            )
            statuses_ip1.append(resp.status_code)

        statuses_ip2 = []
        for _ in range(3):
            resp = await client.post(
                "/api/v1/nid/extract", files={}, headers={"X-Forwarded-For": "203.0.113.20"}
            )
            statuses_ip2.append(resp.status_code)

        # IP1 again, in the same window — must still be limited, confirming
        # its budget wasn't reset or shared with IP2's usage.
        resp = await client.post(
            "/api/v1/nid/extract", files={}, headers={"X-Forwarded-For": "203.0.113.10"}
        )
        status_ip1_again = resp.status_code

    get_settings.cache_clear()

    assert statuses_ip1 == [400, 400, 429]
    assert statuses_ip2 == [400, 400, 429]
    assert status_ip1_again == 429


def test_repeated_rapid_requests_stay_stable(monkeypatch):
    """No crash / unbounded growth from many sequential requests to a failing case."""
    from app.main import app

    client = TestClient(app)
    for _ in range(15):
        resp = client.post("/api/v1/nid/extract", files={})
        assert resp.status_code in (400, 429)
