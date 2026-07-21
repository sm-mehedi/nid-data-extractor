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


def _classify_error(exc: Exception) -> GeminiError:
    if _is_rate_limited(exc):
        return GeminiError(f"Gemini rate limit/quota exceeded: {exc}")
    if _is_timeout(exc):
        return GeminiError(f"Gemini request timed out: {exc}")
    return GeminiError(f"Gemini request failed: {exc}")


def _generate_with_retry(
    client: "genai.Client",
    model: str,
    contents: list,
    config: types.GenerateContentConfig,
    max_retries: int,
    backoff_seconds: float,
):
    """Retries only on 429 (rate limit) and timeout/504 (deadline exceeded) —
    the two failure modes actually worth a second attempt. Anything else
    (auth failure, bad request, safety block) fails immediately since a
    retry wouldn't change the outcome."""
    attempt = 0
    while True:
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            if (not _is_rate_limited(exc) and not _is_timeout(exc)) or attempt >= max_retries:
                raise
            time.sleep(backoff_seconds * (attempt + 1))
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

    try:
        response = _generate_with_retry(
            client,
            settings.gemini_model,
            contents,
            _build_config(http_options, with_thinking=True),
            settings.gemini_max_retries,
            settings.gemini_retry_backoff_seconds,
        )
    except errors.ClientError as exc:
        if getattr(exc, "code", None) == 400 and "thinking" in str(exc).lower():
            # This model doesn't support thinking_config at all — retry once
            # without it rather than hard-failing the whole request.
            try:
                response = _generate_with_retry(
                    client,
                    settings.gemini_model,
                    contents,
                    _build_config(http_options, with_thinking=False),
                    settings.gemini_max_retries,
                    settings.gemini_retry_backoff_seconds,
                )
            except Exception as exc2:
                raise _classify_error(exc2) from exc2
        else:
            raise _classify_error(exc) from exc
    except Exception as exc:
        raise _classify_error(exc) from exc

    if not response.candidates:
        raise GeminiError("Gemini returned no candidates (likely blocked by safety filters).")

    text = response.text
    if not text or not text.strip():
        raise GeminiError("Gemini returned an empty response.")

    return parse_gemini_response(text)
