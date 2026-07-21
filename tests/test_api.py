import asyncio
import io

import pytest
from fastapi.testclient import TestClient

from app.services import gemini as gemini_module
from app.services import pipeline
from tests.helpers import clear_card_image, encode_jpg, tiny_image


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


def _files(front_bytes=None, back_bytes=None, front_name="front.jpg", back_name="back.jpg"):
    files = {}
    if front_bytes is not None:
        files["front_image"] = (front_name, io.BytesIO(front_bytes), "image/jpeg")
    if back_bytes is not None:
        files["back_image"] = (back_name, io.BytesIO(back_bytes), "image/jpeg")
    return files


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_missing_both_files(client):
    resp = client.post("/api/v1/nid/extract", files={})
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert body["data"] is None
    assert len(body["errors"]) >= 1


def test_missing_front_only(client):
    back = encode_jpg(clear_card_image())
    resp = client.post("/api/v1/nid/extract", files=_files(back_bytes=back))
    assert resp.status_code == 400
    assert "front_image" in resp.json()["errors"][0]


def test_missing_back_only(client):
    front = encode_jpg(clear_card_image())
    resp = client.post("/api/v1/nid/extract", files=_files(front_bytes=front))
    assert resp.status_code == 400
    assert "back_image" in resp.json()["errors"][0]


def test_wrong_extension(client):
    front = encode_jpg(clear_card_image())
    resp = client.post(
        "/api/v1/nid/extract",
        files=_files(front, front, front_name="front.gif", back_name="back.jpg"),
    )
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_corrupt_bytes_with_valid_extension(client):
    junk = b"not a real image, just some bytes" * 10
    resp = client.post("/api/v1/nid/extract", files=_files(junk, junk))
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_empty_file(client):
    resp = client.post("/api/v1/nid/extract", files=_files(b"", b""))
    assert resp.status_code == 400


def test_oversized_file(client, monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    big = b"\xff" * (2 * 1024 * 1024)
    resp = client.post("/api/v1/nid/extract", files=_files(big, big))
    assert resp.status_code == 400
    assert "exceeds" in resp.json()["errors"][0]
    get_settings.cache_clear()


def test_tiny_image(client):
    tiny = encode_jpg(tiny_image())
    resp = client.post("/api/v1/nid/extract", files=_files(tiny, tiny))
    assert resp.status_code == 400


def test_full_success_mocked(client, monkeypatch):
    def fake_extract_nid(*args, **kwargs):
        from app.models import ExtractResponse, NidData

        return ExtractResponse(
            success=True,
            data=NidData(
                name="Md. Rahim",
                fatherName="Abdul Karim",
                motherName="Amena Begum",
                dateOfBirth="1998-01-15",
                nidNumber="1234567890123",
                presentAddress="Dhaka, Bangladesh",
                permanentAddress="Cumilla, Bangladesh",
            ),
            warnings=[],
            errors=[],
        )

    monkeypatch.setattr("app.routes.nid.pipeline.extract_nid", fake_extract_nid)

    front = encode_jpg(clear_card_image())
    back = encode_jpg(clear_card_image())
    resp = client.post("/api/v1/nid/extract", files=_files(front, back))
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["nidNumber"] == "1234567890123"
    assert body["errors"] == []


def test_not_nid_card_returns_422(client, monkeypatch):
    def fake_extract_nid(*args, **kwargs):
        raise pipeline.NotNidCardError("not a card")

    monkeypatch.setattr("app.routes.nid.pipeline.extract_nid", fake_extract_nid)
    front = encode_jpg(clear_card_image())
    back = encode_jpg(clear_card_image())
    resp = client.post("/api/v1/nid/extract", files=_files(front, back))
    assert resp.status_code == 422
    assert resp.json()["success"] is False


def test_vision_ocr_error_returns_503(client, monkeypatch):
    from app.services import vision_ocr

    def fake_extract_nid(*args, **kwargs):
        raise vision_ocr.VisionOCRError("rate limited")

    monkeypatch.setattr("app.routes.nid.pipeline.extract_nid", fake_extract_nid)
    front = encode_jpg(clear_card_image())
    back = encode_jpg(clear_card_image())
    resp = client.post("/api/v1/nid/extract", files=_files(front, back))
    assert resp.status_code == 503
    assert resp.json()["success"] is False


def test_gemini_error_returns_503(client, monkeypatch):
    def fake_extract_nid(*args, **kwargs):
        raise gemini_module.GeminiError("timeout")

    monkeypatch.setattr("app.routes.nid.pipeline.extract_nid", fake_extract_nid)
    front = encode_jpg(clear_card_image())
    back = encode_jpg(clear_card_image())
    resp = client.post("/api/v1/nid/extract", files=_files(front, back))
    assert resp.status_code == 503


def test_unexpected_error_returns_500(client, monkeypatch):
    def fake_extract_nid(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.routes.nid.pipeline.extract_nid", fake_extract_nid)
    front = encode_jpg(clear_card_image())
    back = encode_jpg(clear_card_image())
    resp = client.post("/api/v1/nid/extract", files=_files(front, back))
    assert resp.status_code == 500
    assert resp.json()["success"] is False


def test_response_contract_success_shape(client, monkeypatch):
    def fake_extract_nid(*args, **kwargs):
        from app.models import ExtractResponse, NidData

        return ExtractResponse(success=True, data=NidData(name="X"), warnings=[], errors=[])

    monkeypatch.setattr("app.routes.nid.pipeline.extract_nid", fake_extract_nid)
    front = encode_jpg(clear_card_image())
    back = encode_jpg(clear_card_image())
    resp = client.post("/api/v1/nid/extract", files=_files(front, back))
    body = resp.json()
    assert body["success"] is True
    assert body["data"] is not None
    assert body["errors"] == []


def test_response_contract_failure_shape(client):
    resp = client.post("/api/v1/nid/extract", files={})
    body = resp.json()
    assert body["success"] is False
    assert body["data"] is None
    assert len(body["errors"]) >= 1


def test_malformed_non_multipart_body(client):
    resp = client.post(
        "/api/v1/nid/extract",
        content=b"this is not a multipart body at all",
        headers={"Content-Type": "application/json"},
    )
    # FastAPI/Starlette rejects this before our handler runs; it must not crash.
    assert resp.status_code in (400, 422)


def test_security_headers_present(client):
    resp = client.get("/health")
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert "content-security-policy" in resp.headers
