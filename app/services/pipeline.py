"""Orchestrates the full extraction pipeline (Section 2 of build plan):
quality checks -> Cloud Vision OCR -> MRZ parse/checksum -> Gemini
structuring/translation -> merge -> response.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from app.models import ExtractResponse, NidData
from app.services import gemini, image_checks, mrz, translit, vision_ocr

BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


class NotNidCardError(Exception):
    """Raised when Gemini's holistic judgment concludes the images aren't an NID at all."""


@dataclass
class SideResult:
    ocr_text: str
    quality_warnings: list[str]


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise image_checks.ImageQualityError("Failed to re-encode processed image.")
    return buf.tobytes()


def _normalize_digits(value: str | None) -> str | None:
    if value is None:
        return None
    translated = value.translate(BENGALI_DIGITS)
    return translated


def _digits_only(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_digits(value)
    digits = re.sub(r"\D", "", normalized)
    return digits or None


def _ensure_translated(value: str | None, field_label: str, warnings: list[str]) -> str | None:
    """Deterministic safety net: Gemini's translation is not reliably
    consistent across calls (observed returning raw Bengali script for one
    name field while correctly translating others in the same response).
    If Bengali script survives into a name field, apply the local
    transliteration fallback rather than ever returning untranslated script."""
    if value and translit.contains_bengali(value):
        warnings.append(
            f"{field_label} was not translated by the AI model; applied a local "
            "phonetic transliteration fallback (approximate, not a full translation)."
        )
        return translit.transliterate_bengali(value)
    return value


def _mirror_addresses(
    present_address: str | None, permanent_address: str | None, warnings: list[str]
) -> tuple[str | None, str | None]:
    """Deterministic safety net: real Bangladesh NID cards print a single
    address for both fields, but Gemini doesn't reliably mirror it into both
    output fields on every call (observed returning null with a
    low-confidence flag for one field on one run, while correctly mirroring
    on another run for the same card design). Enforced here in code,
    unconditionally, rather than relying on the model to signal it."""
    if not present_address and permanent_address:
        warnings.append(
            "presentAddress was empty; mirrored from permanentAddress (this card "
            "prints a single address for both fields)."
        )
        return permanent_address, permanent_address
    if not permanent_address and present_address:
        warnings.append(
            "permanentAddress was empty; mirrored from presentAddress (this card "
            "prints a single address for both fields)."
        )
        return present_address, present_address
    return present_address, permanent_address


def _process_side(content: bytes, filename: str, max_upload_bytes: int) -> tuple[SideResult, np.ndarray]:
    quality = image_checks.run_quality_pipeline(content, filename, max_upload_bytes)
    encoded = _encode_jpeg(quality.image)
    ocr = vision_ocr.detect_text(encoded)
    return SideResult(ocr_text=ocr.full_text, quality_warnings=quality.warnings), quality.image


def extract_nid(
    front_content: bytes,
    front_filename: str,
    back_content: bytes,
    back_filename: str,
    max_upload_bytes: int,
) -> ExtractResponse:
    warnings: list[str] = []

    front_side, front_image = _process_side(front_content, front_filename, max_upload_bytes)
    back_side, back_image = _process_side(back_content, back_filename, max_upload_bytes)
    warnings.extend(front_side.quality_warnings)
    warnings.extend(back_side.quality_warnings)

    mrz_result = mrz.parse_mrz(back_side.ocr_text)
    if not mrz_result.parsed:
        warnings.append("Could not locate a machine-readable zone (MRZ) on the back image.")
    elif not mrz_result.all_checks_passed:
        warnings.append("MRZ was found but one or more checksum validations failed; back-side data may be unreliable.")

    gemini_result = gemini.structure_and_translate(
        _encode_jpeg(front_image),
        _encode_jpeg(back_image),
        front_side.ocr_text,
        back_side.ocr_text,
    )

    if gemini_result.isNidCard is False:
        raise NotNidCardError(
            "The uploaded images do not appear to be a Bangladesh NID card (front and/or back)."
        )

    if gemini_result.frontQualityNote:
        warnings.append(f"Front: {gemini_result.frontQualityNote}")
    if gemini_result.backQualityNote:
        warnings.append(f"Back: {gemini_result.backQualityNote}")
    for field_name in gemini_result.lowConfidenceFields:
        warnings.append(f"Low confidence on field: {field_name}")

    nid_number = _digits_only(gemini_result.nidNumber)
    date_of_birth = gemini_result.dateOfBirth

    if mrz_result.parsed and mrz_result.all_checks_passed:
        if mrz_result.document_number and mrz_result.document_number != nid_number:
            if nid_number:
                warnings.append(
                    "Front/back may not match: NID number from Gemini differs from the MRZ-verified document number."
                )
            nid_number = mrz_result.document_number or nid_number
        if mrz_result.date_of_birth and mrz_result.date_of_birth != date_of_birth:
            if date_of_birth:
                warnings.append(
                    "Front/back may not match: date of birth differs from the MRZ-verified value."
                )
            date_of_birth = mrz_result.date_of_birth or date_of_birth

    name = _ensure_translated(gemini_result.name, "name", warnings)
    father_name = _ensure_translated(gemini_result.fatherName, "fatherName", warnings)
    mother_name = _ensure_translated(gemini_result.motherName, "motherName", warnings)

    present_address, permanent_address = _mirror_addresses(
        gemini_result.presentAddress, gemini_result.permanentAddress, warnings
    )

    data = NidData(
        name=name,
        fatherName=father_name,
        motherName=mother_name,
        dateOfBirth=date_of_birth,
        nidNumber=nid_number,
        presentAddress=present_address,
        permanentAddress=permanent_address,
    )

    return ExtractResponse(success=True, data=data, warnings=warnings, errors=[])
