"""Gemini structuring + translation (Section 2, step 5 of build plan).

Cloud Vision already read the raw Bengali/English text (step 3); Gemini's job
here is different: understand *meaning* well enough to translate
name/parent-name/address fields without doing literal word-for-word
substitution, and make the holistic "does this look like a real NID" call
that a pure OCR+regex pipeline can't.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai

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


def structure_and_translate(
    front_image_bytes: bytes,
    back_image_bytes: bytes,
    front_ocr_hint: str,
    back_ocr_hint: str,
) -> GeminiResult:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiError("No Gemini API key configured.")

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model)

    prompt = PROMPT_TEMPLATE.format(
        front_hint=front_ocr_hint or "(no text detected)",
        back_hint=back_ocr_hint or "(no text detected)",
    )

    try:
        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": front_image_bytes},
                {"mime_type": "image/jpeg", "data": back_image_bytes},
            ],
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 30},
        )
    except Exception as exc:
        message = str(exc)
        if "429" in message or "rate" in message.lower() or "quota" in message.lower():
            raise GeminiError(f"Gemini rate limit/quota exceeded: {exc}") from exc
        if "timeout" in message.lower() or "deadline" in message.lower():
            raise GeminiError(f"Gemini request timed out: {exc}") from exc
        raise GeminiError(f"Gemini request failed: {exc}") from exc

    if not response.candidates:
        raise GeminiError("Gemini returned no candidates (likely blocked by safety filters).")

    text = response.text
    if not text or not text.strip():
        raise GeminiError("Gemini returned an empty response.")

    return parse_gemini_response(text)
