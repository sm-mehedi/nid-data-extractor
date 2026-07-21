import json

import pytest

from app.config import get_settings
from app.services import gemini


class _FakeResponse:
    def __init__(self, text=None, candidates=None):
        self.text = text
        self.candidates = candidates if candidates is not None else [object()]


class _FakeModel:
    def __init__(self, behavior):
        self._behavior = behavior

    def generate_content(self, *args, **kwargs):
        return self._behavior()


def _valid_json_text():
    return json.dumps(
        {
            "name": "Md. Rahim",
            "fatherName": "Abdul Karim",
            "motherName": "Amena Begum",
            "dateOfBirth": "1998-01-15",
            "nidNumber": "1234567890123",
            "presentAddress": "Dhaka, Bangladesh",
            "permanentAddress": "Cumilla, Bangladesh",
            "isNidCard": True,
            "frontQualityNote": None,
            "backQualityNote": None,
            "lowConfidenceFields": [],
        }
    )


def test_structure_and_translate_no_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(gemini.GeminiError, match="No Gemini API key"):
        gemini.structure_and_translate(b"front", b"back", "front hint", "back hint")
    get_settings.cache_clear()


def test_structure_and_translate_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    monkeypatch.setattr(gemini.genai, "configure", lambda **kw: None)
    monkeypatch.setattr(
        gemini.genai,
        "GenerativeModel",
        lambda *a, **kw: _FakeModel(lambda: _FakeResponse(text=_valid_json_text())),
    )
    result = gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    assert result.name == "Md. Rahim"
    get_settings.cache_clear()


def test_structure_and_translate_rate_limit_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    monkeypatch.setattr(gemini.genai, "configure", lambda **kw: None)

    def _raise():
        raise Exception("429 Resource exhausted: quota")

    monkeypatch.setattr(gemini.genai, "GenerativeModel", lambda *a, **kw: _FakeModel(_raise))
    with pytest.raises(gemini.GeminiError, match="rate limit|quota"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()


def test_structure_and_translate_timeout_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    monkeypatch.setattr(gemini.genai, "configure", lambda **kw: None)

    def _raise():
        raise Exception("Deadline exceeded: timeout")

    monkeypatch.setattr(gemini.genai, "GenerativeModel", lambda *a, **kw: _FakeModel(_raise))
    with pytest.raises(gemini.GeminiError, match="timed out"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()


def test_structure_and_translate_no_candidates_blocked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    monkeypatch.setattr(gemini.genai, "configure", lambda **kw: None)
    monkeypatch.setattr(
        gemini.genai,
        "GenerativeModel",
        lambda *a, **kw: _FakeModel(lambda: _FakeResponse(text="", candidates=[])),
    )
    with pytest.raises(gemini.GeminiError, match="no candidates"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()


def test_structure_and_translate_empty_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    monkeypatch.setattr(gemini.genai, "configure", lambda **kw: None)
    monkeypatch.setattr(
        gemini.genai,
        "GenerativeModel",
        lambda *a, **kw: _FakeModel(lambda: _FakeResponse(text="   ")),
    )
    with pytest.raises(gemini.GeminiError, match="empty response"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()


def test_structure_and_translate_non_json_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    monkeypatch.setattr(gemini.genai, "configure", lambda **kw: None)
    monkeypatch.setattr(
        gemini.genai,
        "GenerativeModel",
        lambda *a, **kw: _FakeModel(lambda: _FakeResponse(text="I cannot help with that.")),
    )
    with pytest.raises(gemini.GeminiError):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()
