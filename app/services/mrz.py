"""ICAO 9303 TD1 MRZ parsing + checksum validation (Section 4 / 8 of build plan).

TD1 is 3 lines x 30 characters, printed on the back of the Bangladesh Smart
NID card. This module only parses that machine-readable zone — it proves
front/back *internal consistency* (checksum-verified fields), not government
database authenticity. The separate 1D barcode on real cards is out of scope
(no documented open standard), per the build plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

MRZ_LINE_LENGTH = 30
MRZ_CHARSET_RE = re.compile(r"^[A-Z0-9<]+$")


def _char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 10
    return 0  # '<' and anything unrecognized counts as 0 per ICAO 9303


def compute_check_digit(data: str) -> int:
    weights = (7, 3, 1)
    total = 0
    for i, c in enumerate(data):
        total += _char_value(c) * weights[i % 3]
    return total % 10


def extract_mrz_lines(raw_text: str) -> list[str]:
    """Finds the 3 MRZ lines inside noisy OCR output of the full back image."""
    candidates = []
    for line in raw_text.splitlines():
        cleaned = line.strip().upper().replace(" ", "")
        if len(cleaned) < 28:
            continue
        if not MRZ_CHARSET_RE.match(cleaned):
            continue
        candidates.append(cleaned)

    if len(candidates) < 3:
        return []

    lines = candidates[-3:]
    normalized = []
    for line in lines:
        if len(line) < MRZ_LINE_LENGTH:
            line = line.ljust(MRZ_LINE_LENGTH, "<")
        elif len(line) > MRZ_LINE_LENGTH:
            line = line[:MRZ_LINE_LENGTH]
        normalized.append(line)
    return normalized


def _yymmdd_to_iso(yymmdd: str, *, is_birth_date: bool, reference_year: int) -> str | None:
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None

    century_cutoff = reference_year % 100
    if is_birth_date:
        century = 1900 if yy > century_cutoff else 2000
    else:
        # Expiry dates on ID cards are almost always in the future relative to issue.
        century = 2000
    year = century + yy
    try:
        return date(year, mm, dd).isoformat()
    except ValueError:
        return None


def _clean_name_field(field_text: str) -> str:
    return " ".join(part for part in field_text.replace("<", " ").split()).strip()


@dataclass
class MrzResult:
    parsed: bool
    lines: list[str] = field(default_factory=list)
    document_number: str | None = None
    document_number_valid: bool | None = None
    date_of_birth: str | None = None
    date_of_birth_valid: bool | None = None
    sex: str | None = None
    expiry_date: str | None = None
    expiry_date_valid: bool | None = None
    name_raw: str | None = None
    composite_valid: bool | None = None
    all_checks_passed: bool = False
    errors: list[str] = field(default_factory=list)


def parse_mrz(raw_text: str, *, reference_year: int | None = None) -> MrzResult:
    if reference_year is None:
        reference_year = date.today().year

    lines = extract_mrz_lines(raw_text)
    if len(lines) != 3:
        return MrzResult(parsed=False, errors=["Could not locate a 3-line MRZ block in the back image OCR text."])

    line1, line2, line3 = lines
    errors: list[str] = []

    doc_number_field = line1[5:14]
    doc_number_check = line1[14]
    optional_data_1 = line1[15:30]
    document_number = doc_number_field.rstrip("<")
    try:
        doc_check_expected = compute_check_digit(doc_number_field)
        document_number_valid = str(doc_check_expected) == doc_number_check
    except Exception:
        document_number_valid = False
        errors.append("Could not validate document number checksum.")

    dob_field = line2[0:6]
    dob_check = line2[6]
    sex_char = line2[7]
    expiry_field = line2[8:14]
    expiry_check = line2[14]
    optional_data_2 = line2[18:29]

    date_of_birth = _yymmdd_to_iso(dob_field, is_birth_date=True, reference_year=reference_year)
    dob_check_expected = compute_check_digit(dob_field)
    date_of_birth_valid = str(dob_check_expected) == dob_check and date_of_birth is not None
    if date_of_birth is None:
        errors.append("Date of birth field in MRZ is not a valid date.")

    expiry_date = _yymmdd_to_iso(expiry_field, is_birth_date=False, reference_year=reference_year)
    expiry_check_expected = compute_check_digit(expiry_field)
    expiry_date_valid = str(expiry_check_expected) == expiry_check

    sex = {"M": "M", "F": "F"}.get(sex_char)

    composite_data = (
        doc_number_field + doc_number_check + optional_data_1
        + dob_field + dob_check
        + expiry_field + expiry_check
        + optional_data_2
    )
    composite_check_char = line2[29]
    composite_expected = compute_check_digit(composite_data)
    composite_valid = str(composite_expected) == composite_check_char

    name_raw = _clean_name_field(line3)

    all_checks_passed = bool(
        document_number_valid and date_of_birth_valid and expiry_date_valid and composite_valid
    )

    return MrzResult(
        parsed=True,
        lines=lines,
        document_number=document_number or None,
        document_number_valid=document_number_valid,
        date_of_birth=date_of_birth,
        date_of_birth_valid=date_of_birth_valid,
        sex=sex,
        expiry_date=expiry_date,
        expiry_date_valid=expiry_date_valid,
        name_raw=name_raw or None,
        composite_valid=composite_valid,
        all_checks_passed=all_checks_passed,
        errors=errors,
    )
