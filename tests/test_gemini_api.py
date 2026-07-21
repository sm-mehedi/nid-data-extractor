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
        return self._behavior(model)


class _FakeClient:
    def __init__(self, behavior):
        self.models = _FakeModels(behavior)


def _install_fake_client(monkeypatch, behavior):
    """`behavior` is called with the model name on each attempt; wrap a
    model-agnostic callable with `lambda model: fn()` where the model
    doesn't matter."""
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


def _rate_limited_error():
    return errors.ClientError(
        code=429, response_json={"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    )


def _overloaded_error():
    return errors.ServerError(
        code=503, response_json={"error": {"message": "model is overloaded", "status": "UNAVAILABLE"}}
    )


def _timeout_error():
    return errors.ServerError(
        code=504, response_json={"error": {"message": "Deadline expired", "status": "DEADLINE_EXCEEDED"}}
    )


@pytest.fixture(autouse=True)
def _recorded_sleep(monkeypatch):
    """Records every gemini.time.sleep() call instead of actually sleeping,
    so tests can assert on the exact backoff schedule used."""
    calls = []
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: calls.append(seconds))
    return calls


def test_structure_and_translate_no_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(gemini.GeminiError, match="No Gemini API key"):
        gemini.structure_and_translate(b"front", b"back", "front hint", "back hint")
    get_settings.cache_clear()


def test_structure_and_translate_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    _install_fake_client(monkeypatch, lambda model: _FakeResponse(text=_valid_json_text()))

    result = gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert result.name == "Md. Rahim"
    get_settings.cache_clear()


def test_structure_and_translate_passes_zero_thinking_budget_and_configured_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "77")
    get_settings.cache_clear()
    fake_client = _install_fake_client(monkeypatch, lambda model: _FakeResponse(text=_valid_json_text()))

    gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert len(fake_client.models.calls) == 1
    config = fake_client.models.calls[0]["config"]
    assert config.thinking_config.thinking_budget == 0
    assert config.http_options.timeout == 77 * 1000
    get_settings.cache_clear()


def test_structure_and_translate_rate_limit_error_not_retried_past_schedule(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()

    fake_client = _install_fake_client(monkeypatch, lambda model: (_ for _ in ()).throw(_rate_limited_error()))

    with pytest.raises(gemini.GeminiError, match="rate limit|quota"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    # Schedule "0" = 1 retry = 2 total calls, then it gives up.
    assert len(fake_client.models.calls) == 2
    get_settings.cache_clear()


def test_structure_and_translate_timeout_error_not_retried_past_schedule(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()

    fake_client = _install_fake_client(monkeypatch, lambda model: (_ for _ in ()).throw(_timeout_error()))

    with pytest.raises(gemini.GeminiError, match="timed out"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert len(fake_client.models.calls) == 2
    get_settings.cache_clear()


def test_structure_and_translate_503_is_retried_with_default_schedule(monkeypatch, _recorded_sleep):
    # This is the actual reported bug: 503 (overloaded) was previously not
    # classified as retryable at all, so it failed on the very first attempt
    # with no retry whatsoever. Using the real default schedule ("3,8,15")
    # here, not an overridden one, to guard against that regression directly.
    # Model ends in "-lite" so no auto-derived fallback attempt muddies this
    # test — fallback behavior has its own dedicated tests below.
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    monkeypatch.delenv("GEMINI_RETRY_BACKOFF_SECONDS", raising=False)
    get_settings.cache_clear()

    fake_client = _install_fake_client(monkeypatch, lambda model: (_ for _ in ()).throw(_overloaded_error()))

    with pytest.raises(gemini.GeminiError, match="overloaded"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    # Default schedule "3,8,15" = 3 retries = 4 total attempts against the
    # primary model, sleeping the exact escalating schedule between them —
    # not a single attempt, and not a flat/linear backoff either.
    assert len(fake_client.models.calls) == 4
    assert _recorded_sleep == [3.0, 8.0, 15.0]
    get_settings.cache_clear()


def test_structure_and_translate_retries_and_then_succeeds_on_rate_limit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0,0")
    get_settings.cache_clear()

    call_count = {"n": 0}

    def _behavior(model):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise _rate_limited_error()
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
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()

    fake_client = _install_fake_client(monkeypatch, lambda model: (_ for _ in ()).throw(ReadTimeout()))

    with pytest.raises(gemini.GeminiError, match="timed out"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert len(fake_client.models.calls) == 2
    get_settings.cache_clear()


def test_structure_and_translate_non_retryable_error_fails_immediately(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0,0")
    get_settings.cache_clear()

    def _raise(model):
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

    def _behavior(model):
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


def test_default_fallback_model_derivation():
    assert gemini._default_fallback_model("gemini-3.5-flash") == "gemini-3.5-flash-lite"
    assert gemini._default_fallback_model("gemini-3.5-flash-lite") == ""
    assert gemini._default_fallback_model("") == ""


def test_structure_and_translate_falls_back_to_lite_model_after_503_exhausts_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()

    def _behavior(model):
        if model == "gemini-3.5-flash":
            raise _overloaded_error()
        return _FakeResponse(text=_valid_json_text())

    fake_client = _install_fake_client(monkeypatch, _behavior)

    result = gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    assert result.name == "Md. Rahim"
    # 2 attempts against the primary model (schedule "0" = 1 retry), then
    # exactly 1 attempt against the auto-derived fallback — no retries on it.
    models_called = [c["model"] for c in fake_client.models.calls]
    assert models_called == ["gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
    get_settings.cache_clear()


def test_structure_and_translate_uses_explicit_fallback_model_override(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.0-flash")
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()

    def _behavior(model):
        if model == "gemini-3.5-flash":
            raise _overloaded_error()
        return _FakeResponse(text=_valid_json_text())

    fake_client = _install_fake_client(monkeypatch, _behavior)

    gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    models_called = [c["model"] for c in fake_client.models.calls]
    assert models_called[-1] == "gemini-3.0-flash"
    get_settings.cache_clear()


def test_structure_and_translate_fallback_model_also_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()

    fake_client = _install_fake_client(monkeypatch, lambda model: (_ for _ in ()).throw(_overloaded_error()))

    with pytest.raises(gemini.GeminiError, match="both.*gemini-3.5-flash.*fallback.*gemini-3.5-flash-lite"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    # 2 attempts on primary (schedule "0") + 1 on fallback (no retries) = 3.
    assert len(fake_client.models.calls) == 3
    get_settings.cache_clear()


def test_structure_and_translate_no_fallback_when_error_is_not_overloaded(monkeypatch):
    # Fallback is scoped specifically to 503 — a rate limit exhausting its
    # retries should not trigger a fallback-model attempt.
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()

    fake_client = _install_fake_client(monkeypatch, lambda model: (_ for _ in ()).throw(_rate_limited_error()))

    with pytest.raises(gemini.GeminiError, match="rate limit|quota"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")

    models_called = [c["model"] for c in fake_client.models.calls]
    assert models_called == ["gemini-3.5-flash", "gemini-3.5-flash"]
    get_settings.cache_clear()


def test_structure_and_translate_no_candidates_blocked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    _install_fake_client(monkeypatch, lambda model: _FakeResponse(text="", candidates=[]))

    with pytest.raises(gemini.GeminiError, match="no candidates"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()


def test_structure_and_translate_empty_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    _install_fake_client(monkeypatch, lambda model: _FakeResponse(text="   "))

    with pytest.raises(gemini.GeminiError, match="empty response"):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()


def test_structure_and_translate_non_json_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    get_settings.cache_clear()
    _install_fake_client(monkeypatch, lambda model: _FakeResponse(text="I cannot help with that."))

    with pytest.raises(gemini.GeminiError):
        gemini.structure_and_translate(b"front", b"back", "hint1", "hint2")
    get_settings.cache_clear()
