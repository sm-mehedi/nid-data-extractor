"""Gemini structuring + translation (Section 2, step 5 of build plan).

Cloud Vision already read the raw Bengali/English text (step 3); Gemini's job
here is different: understand *meaning* well enough to translate
name/parent-name/address fields without doing literal word-for-word
substitution, and make the holistic "does this look like a real NID" call
that a pure OCR+regex pipeline can't.

Uses the `google-genai` SDK (the legacy `google-generativeai` package has no
`thinking_config` support at all, in any version, and is fully deprecated).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import errors, types

from app.config import get_settings

EXPECTED_KEYS = {
    "name",
    "fatherName",
    "motherName",
    "dateOfBirth",
    "nidNumber",
    "presentAddress",
    "permanentAddress",
    "isNidCard",
    "frontQualityNote",
    "backQualityNote",
    "lowConfidenceFields",
}

PROMPT_TEMPLATE = """You are analyzing the front and back photos of a Bangladesh National ID (NID) card.

You are given OCR text extracted from each image as a hint (it may contain errors, especially in Bengali script) — use it as supporting context, not ground truth. Read the images directly to extract the actual data.

Front-image OCR hint:
---
{front_hint}
---

Back-image OCR hint:
---
{back_hint}
---

Extract the following fields and translate any Bengali text into natural, meaning-preserving English (NOT literal word-for-word translation — e.g. translate a village/upazila/district address the way it would naturally read in English, preserving the place names transliterated appropriately):

- name (the cardholder's full name)
- fatherName
- motherName
- dateOfBirth (ISO format YYYY-MM-DD)
- nidNumber (digits only, no spaces — convert any Bengali numerals ০-৯ to Latin 0-9)
- presentAddress (in English)
- permanentAddress (in English)

Also assess:
- isNidCard: true/false — does this genuinely look like a real Bangladesh NID card (front and back)?
- frontQualityNote: short note on the front image's readability (e.g. "clear", "partially obscured by glare", "cut off at edge") or null if fine
- backQualityNote: same, for the back image
- lowConfidenceFields: array of field names (from the list above) you are NOT confident about, because they were blurry, obscured, or ambiguous. Empty array if all fields are confidently read.

If a field is genuinely unreadable or absent, set it to null rather than guessing.

Respond with ONLY a single JSON object with exactly these keys: name, fatherName, motherName, dateOfBirth, nidNumber, presentAddress, permanentAddress, isNidCard, frontQualityNote, backQualityNote, lowConfidenceFields. No markdown, no commentary, no code fences.
"""


class GeminiError(Exception):
    """Raised on Gemini API failure (network, auth, timeout, quota) or unusable response."""


@dataclass
class GeminiResult:
    name: str | None = None
    fatherName: str | None = None
    motherName: str | None = None
    dateOfBirth: str | None = None
    nidNumber: str | None = None
    presentAddress: str | None = None
    permanentAddress: str | None = None
    isNidCard: bool | None = None
    frontQualityNote: str | None = None
    backQualityNote: str | None = None
    lowConfidenceFields: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(json)?", "", stripped.strip(), flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped.strip()).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise GeminiError(f"Gemini response was not valid JSON: {exc}") from exc

    raise GeminiError("Gemini response contained no parseable JSON object.")


def parse_gemini_response(text: str) -> GeminiResult:
    data = _extract_json_object(text)
    missing = sorted(EXPECTED_KEYS - set(data.keys()))

    low_confidence = data.get("lowConfidenceFields") or []
    if not isinstance(low_confidence, list):
        low_confidence = []

    return GeminiResult(
        name=data.get("name"),
        fatherName=data.get("fatherName"),
        motherName=data.get("motherName"),
        dateOfBirth=data.get("dateOfBirth"),
        nidNumber=data.get("nidNumber"),
        presentAddress=data.get("presentAddress"),
        permanentAddress=data.get("permanentAddress"),
        isNidCard=data.get("isNidCard"),
        frontQualityNote=data.get("frontQualityNote"),
        backQualityNote=data.get("backQualityNote"),
        lowConfidenceFields=[str(f) for f in low_confidence],
        missing_keys=missing,
    )


def _is_rate_limited(exc: Exception) -> bool:
    if isinstance(exc, errors.ClientError) and getattr(exc, "code", None) == 429:
        return True
    message = str(exc).lower()
    return "429" in message or "rate" in message or "quota" in message


def _is_overloaded(exc: Exception) -> bool:
    """503 UNAVAILABLE — the model is temporarily over capacity. Distinct
    from both 429 (caller is rate-limited) and 504 (deadline exceeded); this
    was previously not classified as retryable at all, so a single 503 fell
    straight through to a hard failure instead of retrying."""
    if isinstance(exc, errors.ServerError) and getattr(exc, "code", None) == 503:
        return True
    message = str(exc).lower()
    return "503" in message or "unavailable" in message


def _is_timeout(exc: Exception) -> bool:
    if isinstance(exc, errors.ServerError) and getattr(exc, "code", None) == 504:
        return True
    # httpx timeout exceptions (a true client-side timeout, no response from
    # the server at all) often carry an empty message — the class name
    # ("ReadTimeout", "ConnectTimeout", ...) is the reliable signal there.
    if "timeout" in type(exc).__name__.lower():
        return True
    message = str(exc).lower()
    return "timeout" in message or "deadline" in message


def _is_retryable(exc: Exception) -> bool:
    return _is_rate_limited(exc) or _is_overloaded(exc) or _is_timeout(exc)


def _classify_error(exc: Exception) -> GeminiError:
    if _is_rate_limited(exc):
        return GeminiError(f"Gemini rate limit/quota exceeded: {exc}")
    if _is_overloaded(exc):
        return GeminiError(f"Gemini model overloaded (503): {exc}")
    if _is_timeout(exc):
        return GeminiError(f"Gemini request timed out: {exc}")
    return GeminiError(f"Gemini request failed: {exc}")


def _default_fallback_model(model: str) -> str:
    if not model or model.endswith("-lite"):
        return ""
    return f"{model}-lite"


def _generate_with_retry(
    client: "genai.Client",
    model: str,
    contents: list,
    config: types.GenerateContentConfig,
    backoff_schedule: list[float],
):
    """Retries on 429 (rate limit), 503 (overloaded), and timeout/504
    (deadline exceeded) — failure modes actually worth a second attempt.
    Anything else (auth failure, bad request, safety block) fails
    immediately since a retry wouldn't change the outcome.

    `backoff_schedule` is a list of seconds to wait before each successive
    retry, e.g. [3, 8, 15] = up to 3 retries (4 total attempts)."""
    attempt = 0
    while True:
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            if not _is_retryable(exc) or attempt >= len(backoff_schedule):
                raise
            time.sleep(backoff_schedule[attempt])
            attempt += 1


def _build_config(
    http_options: types.HttpOptions, *, with_thinking: bool
) -> types.GenerateContentConfig:
    if with_thinking:
        # This is structured extraction + translation, not a task that
        # benefits from extended reasoning — a zero thinking budget cuts
        # both latency and cost for thinking-capable models.
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=http_options,
        )
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        http_options=http_options,
    )


def _attempt_model(
    client: "genai.Client",
    model: str,
    contents: list,
    http_options: types.HttpOptions,
    backoff_schedule: list[float],
):
    """Runs generate_content against `model` with the given retry schedule.
    Falls back to a no-thinking config (once, same retry schedule) if the
    model rejects thinking_config outright rather than hard-failing."""
    try:
        return _generate_with_retry(
            client, model, contents, _build_config(http_options, with_thinking=True), backoff_schedule
        )
    except errors.ClientError as exc:
        if getattr(exc, "code", None) == 400 and "thinking" in str(exc).lower():
            return _generate_with_retry(
                client, model, contents, _build_config(http_options, with_thinking=False), backoff_schedule
            )
        raise


def structure_and_translate(
    front_image_bytes: bytes,
    back_image_bytes: bytes,
    front_ocr_hint: str,
    back_ocr_hint: str,
) -> GeminiResult:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiError("No Gemini API key configured.")

    client = genai.Client(api_key=settings.gemini_api_key)

    prompt = PROMPT_TEMPLATE.format(
        front_hint=front_ocr_hint or "(no text detected)",
        back_hint=back_ocr_hint or "(no text detected)",
    )

    contents = [
        prompt,
        types.Part.from_bytes(data=front_image_bytes, mime_type="image/jpeg"),
        types.Part.from_bytes(data=back_image_bytes, mime_type="image/jpeg"),
    ]

    # google-genai's HttpOptions.timeout is in milliseconds.
    http_options = types.HttpOptions(timeout=settings.gemini_timeout_seconds * 1000)
    backoff_schedule = settings.gemini_retry_backoff_schedule

    try:
        response = _attempt_model(client, settings.gemini_model, contents, http_options, backoff_schedule)
    except Exception as primary_exc:
        fallback_model = settings.gemini_fallback_model or _default_fallback_model(settings.gemini_model)
        if fallback_model and _is_overloaded(primary_exc):
            # Different model variants often have separate capacity pools —
            # one extra attempt, no retries on the fallback itself.
            try:
                response = _attempt_model(client, fallback_model, contents, http_options, [])
            except Exception as fallback_exc:
                raise GeminiError(
                    f"Gemini request failed on both '{settings.gemini_model}' (after "
                    f"{len(backoff_schedule)} retries, 503 overloaded) and fallback "
                    f"'{fallback_model}': {fallback_exc}"
                ) from fallback_exc
        else:
            raise _classify_error(primary_exc) from primary_exc

    if not response.candidates:
        raise GeminiError("Gemini returned no candidates (likely blocked by safety filters).")

    text = response.text
    if not text or not text.strip():
        raise GeminiError("Gemini returned an empty response.")

    return parse_gemini_response(text)
