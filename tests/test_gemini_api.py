import json

import pytest

from app.config import get_settings
from app.services import gemini
from google.genai import errors


class _FakeResponse:
    def __init__(self, text=None, candidates=None):
        self.text = text
        self.candidates = candidates if candidates is not None else [object()]


class _FakeModels:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._behavior()


class _FakeClient:
    def __init__(self, behavior):
        self.models = _FakeModels(behavior)


def _install_fake_client(monkeypatch, behavior):
    fake_client = _FakeClient(behavior)
    monkeypatch.setattr(gemini.genai, "Client", lambda **kw: fake_client)
    return fake_client


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


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # Retry/backoff tests would otherwise really sleep for seconds at a time.
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: None)


def test_structure_and_translate_no_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(gemini.GeminiError, match="No Gemini API key"):
        gemini.structure_and_translate(b"front", b"back", "front hint", "back hint")
    get_settings.cache_clear()


def test_structure_and_translate_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    _install_fake_client(monkeypatch, lambda: _FakeResponse(text=_valid_json_text()))

    result = gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert result.name == "Md. Rahim"
    get_settings.cache_clear()


def test_structure_and_translate_passes_zero_thinking_budget_and_configured_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "77")
    get_settings.cache_clear()
    fake_client = _install_fake_client(monkeypatch, lambda: _FakeResponse(text=_valid_json_text()))

    gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert len(fake_client.models.calls) == 1
    config = fake_client.models.calls[0]["config"]
    assert config.thinking_config.thinking_budget == 0
    assert config.http_options.timeout == 77 * 1000
    get_settings.cache_clear()


def test_structure_and_translate_rate_limit_error_not_retried_past_max(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    get_settings.cache_clear()

    def _raise():
        raise errors.ClientError(
            code=429, response_json={"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
        )

    fake_client = _install_fake_client(monkeypatch, _raise)

    with pytest.raises(gemini.GeminiError, match="rate limit|quota"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    # 1 initial attempt + 1 retry = 2 total calls, then it gives up.
    assert len(fake_client.models.calls) == 2
    get_settings.cache_clear()


def test_structure_and_translate_timeout_error_not_retried_past_max(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    get_settings.cache_clear()

    def _raise():
        raise errors.ServerError(
            code=504, response_json={"error": {"message": "Deadline expired", "status": "DEADLINE_EXCEEDED"}}
        )

    fake_client = _install_fake_client(monkeypatch, _raise)

    with pytest.raises(gemini.GeminiError, match="timed out"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert len(fake_client.models.calls) == 2
    get_settings.cache_clear()


def test_structure_and_translate_retries_and_then_succeeds_on_rate_limit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "2")
    get_settings.cache_clear()

    call_count = {"n": 0}

    def _behavior():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise errors.ClientError(
                code=429, response_json={"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
            )
        return _FakeResponse(text=_valid_json_text())

    fake_client = _install_fake_client(monkeypatch, _behavior)

    result = gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert result.name == "Md. Rahim"
    assert len(fake_client.models.calls) == 3  # 2 failures + 1 success
    get_settings.cache_clear()


def test_structure_and_translate_retries_on_client_side_timeout_exception(monkeypatch):
    # A genuine client-side timeout (no response at all from the server) often
    # surfaces as an httpx-style exception with an empty message — the class
    # name is the only reliable signal, not the message text.
    class ReadTimeout(Exception):
        pass

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    get_settings.cache_clear()

    def _raise():
        raise ReadTimeout()

    fake_client = _install_fake_client(monkeypatch, _raise)

    with pytest.raises(gemini.GeminiError, match="timed out"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert len(fake_client.models.calls) == 2
    get_settings.cache_clear()


def test_structure_and_translate_non_retryable_error_fails_immediately(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "2")
    get_settings.cache_clear()

    def _raise():
        raise errors.ClientError(code=401, response_json={"error": {"message": "invalid API key"}})

    fake_client = _install_fake_client(monkeypatch, _raise)

    with pytest.raises(gemini.GeminiError, match="request failed"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    # Auth failures aren't retryable — must fail on the very first attempt.
    assert len(fake_client.models.calls) == 1
    get_settings.cache_clear()


def test_structure_and_translate_falls_back_when_model_rejects_thinking_config(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()

    call_count = {"n": 0}

    def _behavior():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise errors.ClientError(
                code=400,
                response_json={"error": {"message": "thinking_config is not supported for this model"}},
            )
        return _FakeResponse(text=_valid_json_text())

    fake_client = _install_fake_client(monkeypatch, _behavior)

    result = gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert result.name == "Md. Rahim"
    assert len(fake_client.models.calls) == 2
    # First attempt had thinking_config set, the retry must not.
    assert fake_client.models.calls[0]["config"].thinking_config is not None
    assert fake_client.models.calls[1]["config"].thinking_config is None
    get_settings.cache_clear()


def test_structure_and_translate_no_candidates_blocked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    _install_fake_client(monkeypatch, lambda: _FakeResponse(text="", candidates=[]))

    with pytest.raises(gemini.GeminiError, match="no candidates"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()


def test_structure_and_translate_empty_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    _install_fake_client(monkeypatch, lambda: _FakeResponse(text="   "))

    with pytest.raises(gemini.GeminiError, match="empty response"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()


def test_structure_and_translate_non_json_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    _install_fake_client(monkeypatch, lambda: _FakeResponse(text="I cannot help with that."))

    with pytest.raises(gemini.GeminiError):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()
