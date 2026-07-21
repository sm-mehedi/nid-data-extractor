import json

import pytest

from app.services import gemini


def _valid_payload(**overrides):
    payload = {
        "name": "Md. Rahim",
        "fatherName": "Abdul Karim",
        "motherName": "Amena Begum",
        "dateOfBirth": "1998-01-15",
        "nidNumber": "1234567890123",
        "presentAddress": "Village Rampur, Upazila Debidwar, District Cumilla",
        "permanentAddress": "Village Rampur, Upazila Debidwar, District Cumilla",
        "isNidCard": True,
        "frontQualityNote": None,
        "backQualityNote": None,
        "lowConfidenceFields": [],
    }
    payload.update(overrides)
    return payload


def test_parse_gemini_response_plain_json():
    text = json.dumps(_valid_payload())
    result = gemini.parse_gemini_response(text)
    assert result.name == "Md. Rahim"
    assert result.isNidCard is True
    assert result.missing_keys == []


def test_parse_gemini_response_wrapped_in_markdown_fences():
    text = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    result = gemini.parse_gemini_response(text)
    assert result.nidNumber == "1234567890123"


def test_parse_gemini_response_missing_keys_filled_with_none():
    payload = _valid_payload()
    del payload["motherName"]
    del payload["backQualityNote"]
    result = gemini.parse_gemini_response(json.dumps(payload))
    assert result.motherName is None
    assert set(result.missing_keys) == {"motherName", "backQualityNote"}


def test_parse_gemini_response_not_nid_card():
    payload = _valid_payload(isNidCard=False, name=None, nidNumber=None)
    result = gemini.parse_gemini_response(json.dumps(payload))
    assert result.isNidCard is False


def test_parse_gemini_response_non_json_raises():
    with pytest.raises(gemini.GeminiError):
        gemini.parse_gemini_response("Sorry, I cannot process this request.")


def test_parse_gemini_response_low_confidence_fields():
    payload = _valid_payload(lowConfidenceFields=["fatherName", "presentAddress"])
    result = gemini.parse_gemini_response(json.dumps(payload))
    assert result.lowConfidenceFields == ["fatherName", "presentAddress"]


def test_parse_gemini_response_embedded_json_with_surrounding_text():
    text = "Here is the result:\n" + json.dumps(_valid_payload()) + "\nLet me know if you need anything else."
    result = gemini.parse_gemini_response(text)
    assert result.name == "Md. Rahim"
