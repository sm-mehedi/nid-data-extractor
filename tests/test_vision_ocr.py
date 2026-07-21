import httpx
import pytest

from app.config import get_settings
from app.services import vision_ocr


class _FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


def test_detect_text_no_credentials_configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_VISION_API_KEY", "")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    get_settings.cache_clear()
    with pytest.raises(vision_ocr.VisionOCRError, match="No Cloud Vision credentials"):
        vision_ocr.detect_text(b"fake-bytes")
    get_settings.cache_clear()


def test_detect_text_success_via_rest(monkeypatch):
    def fake_post(url, params=None, json=None, timeout=None):
        return _FakeResponse(200, {"responses": [{"fullTextAnnotation": {"text": "HELLO WORLD"}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = vision_ocr._detect_via_rest(b"fake-bytes", "fake-key")
    assert result["fullTextAnnotation"]["text"] == "HELLO WORLD"


def test_detect_text_rate_limited(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(429))
    with pytest.raises(vision_ocr.VisionOCRError, match="rate limit"):
        vision_ocr._detect_via_rest(b"fake-bytes", "fake-key")


def test_detect_text_server_error(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(503))
    with pytest.raises(vision_ocr.VisionOCRError, match="server error"):
        vision_ocr._detect_via_rest(b"fake-bytes", "fake-key")


def test_detect_text_empty_responses(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(200, {"responses": []}))
    with pytest.raises(vision_ocr.VisionOCRError, match="empty response"):
        vision_ocr._detect_via_rest(b"fake-bytes", "fake-key")


def test_detect_text_error_in_response(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: _FakeResponse(200, {"responses": [{"error": {"message": "bad image"}}]}),
    )
    with pytest.raises(vision_ocr.VisionOCRError, match="API error"):
        vision_ocr._detect_via_rest(b"fake-bytes", "fake-key")


def test_detect_text_network_error(monkeypatch):
    def fake_post(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(vision_ocr.VisionOCRError, match="request failed"):
        vision_ocr._detect_via_rest(b"fake-bytes", "fake-key")


def test_detect_text_dispatches_to_rest_when_api_key_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_VISION_API_KEY", "fake-key")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    get_settings.cache_clear()

    monkeypatch.setattr(
        vision_ocr,
        "_detect_via_rest",
        lambda image_bytes, api_key: {"fullTextAnnotation": {"text": "OK"}},
    )
    result = vision_ocr.detect_text(b"fake-bytes")
    assert result.full_text == "OK"
    get_settings.cache_clear()
