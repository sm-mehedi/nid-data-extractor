from app.services import mrz
from tests.helpers import build_td1_mrz


def test_compute_check_digit_known_values():
    # Worked examples from the official ICAO 9303 Part 4 TD3 sample MRZ
    # (L898902C36UTO7408122F1204159ZE184226B<<<<<10): DOB field "740812" -> 2,
    # and document number "L898902C3" -> 6.
    assert mrz.compute_check_digit("740812") == 2
    assert mrz.compute_check_digit("L898902C3") == 6


def test_extract_mrz_lines_from_noisy_text():
    block = build_td1_mrz()
    noisy = "SOME HEADER TEXT\nBangladesh National ID\n" + block + "trailing junk\n"
    lines = mrz.extract_mrz_lines(noisy)
    assert len(lines) == 3
    assert all(len(l) == 30 for l in lines)


def test_extract_mrz_lines_not_found_in_plain_text():
    lines = mrz.extract_mrz_lines("just some regular OCR text\nwith no MRZ block at all\n")
    assert lines == []


def test_parse_mrz_valid_full_success():
    block = build_td1_mrz(doc_number="987654321", dob="980115", expiry="300101")
    result = mrz.parse_mrz(block, reference_year=2026)
    assert result.parsed is True
    assert result.document_number == "987654321"
    assert result.document_number_valid is True
    assert result.date_of_birth == "1998-01-15"
    assert result.date_of_birth_valid is True
    assert result.expiry_date_valid is True
    assert result.composite_valid is True
    assert result.all_checks_passed is True
    assert result.sex == "M"


def test_parse_mrz_corrupted_document_number_checksum():
    block = build_td1_mrz(corrupt_doc_check=True)
    result = mrz.parse_mrz(block, reference_year=2026)
    assert result.parsed is True
    assert result.document_number_valid is False
    assert result.all_checks_passed is False


def test_parse_mrz_corrupted_dob_checksum():
    block = build_td1_mrz(corrupt_dob_check=True)
    result = mrz.parse_mrz(block, reference_year=2026)
    assert result.parsed is True
    assert result.date_of_birth_valid is False
    assert result.all_checks_passed is False


def test_parse_mrz_corrupted_composite_checksum():
    block = build_td1_mrz(corrupt_composite=True)
    result = mrz.parse_mrz(block, reference_year=2026)
    assert result.parsed is True
    assert result.composite_valid is False
    assert result.all_checks_passed is False


def test_parse_mrz_garbled_unparseable():
    result = mrz.parse_mrz("this is not an MRZ block\njust regular text\nno structure here\n")
    assert result.parsed is False
    assert result.errors


def test_parse_mrz_dob_century_heuristic_old_person():
    # yy=98 with reference_year=2026 (cutoff 26) -> 1998, a plausible adult birth year.
    block = build_td1_mrz(dob="980115")
    result = mrz.parse_mrz(block, reference_year=2026)
    assert result.date_of_birth == "1998-01-15"


def test_parse_mrz_dob_century_heuristic_recent_person():
    # yy=05 with reference_year=2026 (cutoff 26) -> 2005.
    block = build_td1_mrz(dob="050615")
    result = mrz.parse_mrz(block, reference_year=2026)
    assert result.date_of_birth == "2005-06-15"


def test_parse_mrz_name_field_cleaned():
    block = build_td1_mrz(surname="RAHIM", given_names="MD")
    result = mrz.parse_mrz(block, reference_year=2026)
    assert result.name_raw == "RAHIM MD"
