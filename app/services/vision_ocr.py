"""Google Cloud Vision OCR wrapper — DOCUMENT_TEXT_DETECTION on NID images.

Supports two auth modes (see .env.example): a service-account JSON via
GOOGLE_APPLICATION_CREDENTIALS (uses the official client library) or a raw
API key via GOOGLE_CLOUD_VISION_API_KEY (uses the plain REST endpoint, no
extra library needed). Service-account credentials take priority if both are set.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx

from app.config import get_settings

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


class VisionOCRError(Exception):
    """Raised on any Cloud Vision failure: auth, network, timeout, quota, or malformed response."""


@dataclass
class OcrResult:
    full_text: str
    raw_response: dict


def _detect_via_rest(image_bytes: bytes, api_key: str) -> dict:
    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode("utf-8")},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["bn", "en"]},
            }
        ]
    }
    try:
        resp = httpx.post(VISION_ENDPOINT, params={"key": api_key}, json=payload, timeout=30.0)
    except httpx.RequestError as exc:
        raise VisionOCRError(f"Cloud Vision request failed: {exc}") from exc

    if resp.status_code == 429:
        raise VisionOCRError("Cloud Vision rate limit exceeded (429).")
    if resp.status_code >= 500:
        raise VisionOCRError(f"Cloud Vision server error ({resp.status_code}).")
    if resp.status_code != 200:
        raise VisionOCRError(
            f"Cloud Vision request failed ({resp.status_code}): {resp.text[:300]}"
        )

    data = resp.json()
    responses = data.get("responses", [])
    if not responses:
        raise VisionOCRError("Cloud Vision returned an empty response.")
    if "error" in responses[0]:
        raise VisionOCRError(f"Cloud Vision API error: {responses[0]['error']}")
    return responses[0]


def _detect_via_client_library(image_bytes: bytes) -> dict:
    from google.cloud import vision  # lazy import: only needed for this auth path

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    image_context = vision.ImageContext(language_hints=["bn", "en"])
    try:
        response = client.document_text_detection(image=image, image_context=image_context)
    except Exception as exc:  # google-api-core raises a family of exceptions here
        raise VisionOCRError(f"Cloud Vision request failed: {exc}") from exc

    if response.error.message:
        raise VisionOCRError(f"Cloud Vision API error: {response.error.message}")

    return {"fullTextAnnotation": {"text": response.full_text_annotation.text}}


def detect_text(image_bytes: bytes) -> OcrResult:
    settings = get_settings()
    if settings.google_application_credentials:
        raw = _detect_via_client_library(image_bytes)
    elif settings.google_cloud_vision_api_key:
        raw = _detect_via_rest(image_bytes, settings.google_cloud_vision_api_key)
    else:
        raise VisionOCRError("No Cloud Vision credentials configured.")

    full_text = raw.get("fullTextAnnotation", {}).get("text", "")
    return OcrResult(full_text=full_text, raw_response=raw)
