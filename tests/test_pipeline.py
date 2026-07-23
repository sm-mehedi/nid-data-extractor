import pytest

from app.services import gemini as gemini_module
from app.services import pipeline, translit, vision_ocr
from tests.helpers import blurry_card_image, build_td1_mrz, clear_card_image, encode_jpg, glare_image


FRONT_BYTES = encode_jpg(clear_card_image())
BACK_BYTES = encode_jpg(clear_card_image())


def _make_ocr_stub(front_text: str, back_text: str):
    calls = {"n": 0}

    def _fake_detect_text(image_bytes: bytes):
        calls["n"] += 1
        text = front_text if calls["n"] == 1 else back_text
        return vision_ocr.OcrResult(full_text=text, raw_response={})

    return _fake_detect_text


def _base_gemini_result(**overrides) -> gemini_module.GeminiResult:
    defaults = dict(
        name="Md. Rahim",
        fatherName="Abdul Karim",
        motherName="Amena Begum",
        dateOfBirth="1998-01-15",
        nidNumber="987654321",
        presentAddress="Village Rampur, Upazila Debidwar, District Cumilla",
        permanentAddress="Village Rampur, Upazila Debidwar, District Cumilla",
        isNidCard=True,
        frontQualityNote=None,
        backQualityNote=None,
        lowConfidenceFields=[],
    )
    defaults.update(overrides)
    return gemini_module.GeminiResult(**defaults)


def test_extract_nid_full_success(monkeypatch):
    back_text_with_mrz = "NATIONAL ID CARD\n" + build_td1_mrz(doc_number="987654321", dob="980115")
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text, no mrz here", back_text_with_mrz),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.success is True
    assert result.errors == []
    assert result.data.nidNumber == "987654321"
    assert result.data.dateOfBirth == "1998-01-15"
    assert result.warnings == []


def test_extract_nid_blurry_and_glare_photos_still_succeed_end_to_end(monkeypatch):
    # End-to-end regression: blur/exposure/glare must never block a request —
    # a real (if imperfect) front/back pair should still reach Cloud
    # Vision/Gemini and come back as success:true with descriptive warnings,
    # not a 400-equivalent hard failure.
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(),
    )

    blurry_front = encode_jpg(blurry_card_image())
    glare_back = encode_jpg(glare_image())

    result = pipeline.extract_nid(blurry_front, "front.jpg", glare_back, "back.jpg", 10_000_000)

    assert result.success is True
    assert result.errors == []
    assert any("blur" in w.lower() for w in result.warnings)
    assert any("glare" in w.lower() for w in result.warnings)


def test_extract_nid_mrz_cross_check_mismatch_warns_but_succeeds(monkeypatch):
    back_text_with_mrz = "NATIONAL ID CARD\n" + build_td1_mrz(doc_number="111111111", dob="980115")
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", back_text_with_mrz),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(nidNumber="222222222"),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.success is True
    assert result.data.nidNumber == "111111111"  # MRZ-verified value wins
    assert any("Front/back may not match" in w for w in result.warnings)


def test_extract_nid_not_a_card_raises(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("random text", "more random text"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(isNidCard=False, name=None, nidNumber=None),
    )

    with pytest.raises(pipeline.NotNidCardError):
        pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)


def test_extract_nid_low_confidence_fields_produce_warnings(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(lowConfidenceFields=["fatherName"]),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.success is True
    assert "Low confidence on field: fatherName" in result.warnings


def test_extract_nid_bengali_digits_normalized(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(nidNumber="১২৩৪৫৬৭৮৯০১২৩"),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.data.nidNumber == "1234567890123"


def test_extract_nid_mrz_unparseable_produces_warning(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text with no mrz block at all"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.success is True
    assert any("machine-readable zone" in w for w in result.warnings)


def test_extract_nid_propagates_vision_ocr_error(monkeypatch):
    def _raise(*a, **kw):
        raise vision_ocr.VisionOCRError("simulated timeout")

    monkeypatch.setattr("app.services.vision_ocr.detect_text", _raise)

    with pytest.raises(vision_ocr.VisionOCRError):
        pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)


def test_extract_nid_propagates_gemini_error(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text"),
    )

    def _raise(*a, **kw):
        raise gemini_module.GeminiError("simulated 429")

    monkeypatch.setattr("app.services.gemini.structure_and_translate", _raise)

    with pytest.raises(gemini_module.GeminiError):
        pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)


def test_extract_nid_same_image_both_sides_does_not_crash(monkeypatch):
    # Neither side will contain a real MRZ, since both are the same "front".
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text, no mrz", "front text, no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", FRONT_BYTES, "front.jpg", 10_000_000)

    assert result.success is True
    assert any("machine-readable zone" in w for w in result.warnings)


def test_extract_nid_front_back_swapped_does_not_crash(monkeypatch):
    # Swapping which field the real front/back bytes land in shouldn't matter to
    # the pipeline — it just processes whatever bytes it's given per side.
    back_text_with_mrz = "NATIONAL ID CARD\n" + build_td1_mrz(doc_number="555555555", dob="900101")
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub(back_text_with_mrz, "front text, no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(),
    )

    # Here "front_content" is actually back-side bytes/text and vice versa —
    # simulating the user swapping the two uploads.
    result = pipeline.extract_nid(BACK_BYTES, "back.jpg", FRONT_BYTES, "front.jpg", 10_000_000)
    assert result.success is True


def test_extract_nid_empty_ocr_text_both_sides_still_succeeds_via_gemini(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("", ""),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)
    assert result.success is True
    assert any("machine-readable zone" in w for w in result.warnings)


def test_extract_nid_one_side_unreadable_produces_side_specific_warning(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(
            backQualityNote="obscured by glare, several fields illegible",
            motherName=None,
        ),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.success is True
    assert any("Back: obscured by glare" in w for w in result.warnings)
    assert not any(w.startswith("Front:") for w in result.warnings)
    assert result.data.motherName is None


def test_extract_nid_long_multiline_address_preserved(monkeypatch):
    long_address = (
        "House 12, Road 5, Block C, Section 2, Mirpur, Dhaka-1216, "
        "near Central Mosque, opposite Green View School, Bangladesh"
    )
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(presentAddress=long_address, permanentAddress=long_address),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.data.presentAddress == long_address
    assert result.data.permanentAddress == long_address


def test_extract_nid_name_with_honorifics_passed_through(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(name="Md. Rahim", fatherName="Mst. Amena Begum"),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)
    assert result.data.name == "Md. Rahim"
    assert result.data.fatherName == "Mst. Amena Begum"


def test_extract_nid_partial_translation_failure_gets_transliteration_fallback(monkeypatch):
    # Regression test for a real observed failure mode: Gemini translated
    # presentAddress correctly in the same response where it left fatherName
    # as raw, untranslated Bengali script. This is not reproduced via a live
    # call (which wouldn't be deterministic) — it's mocked directly, and the
    # code-level safety net must catch it regardless of whether Gemini's
    # inconsistency is reproducible on demand.
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(
            name="Md. Rahim",  # correctly translated
            fatherName="আব্দুল করিম",  # left untranslated by the model
            motherName="আমেনা বেগম",  # left untranslated by the model
            presentAddress="Dhaka, Bangladesh",  # correctly translated
        ),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.success is True
    # The hard invariant: no Bengali script anywhere in the final output,
    # no matter what the model returned.
    assert not translit.contains_bengali(result.data.name)
    assert not translit.contains_bengali(result.data.fatherName)
    assert not translit.contains_bengali(result.data.motherName)
    # The field that was already correctly translated must be untouched.
    assert result.data.name == "Md. Rahim"
    assert any("fatherName" in w and "transliteration fallback" in w for w in result.warnings)
    assert any("motherName" in w and "transliteration fallback" in w for w in result.warnings)
    # name was fine — must not have triggered a fallback warning for it.
    assert not any(w.startswith("name was not translated") for w in result.warnings)


def test_extract_nid_fully_translated_names_produce_no_fallback_warning(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert not any("transliteration fallback" in w for w in result.warnings)


def test_extract_nid_missing_present_address_mirrors_from_permanent(monkeypatch):
    # Regression test for a real observed inconsistency: one run correctly
    # mirrored the single-address card design into both fields, another run
    # (same card design) returned null with a low-confidence flag for one
    # field. Mocked directly rather than relying on a live call reproducing
    # the inconsistency on demand.
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(presentAddress=None, permanentAddress="Dhaka, Bangladesh"),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.data.presentAddress == "Dhaka, Bangladesh"
    assert result.data.permanentAddress == "Dhaka, Bangladesh"
    assert any("presentAddress was empty" in w and "mirrored" in w for w in result.warnings)


def test_extract_nid_missing_permanent_address_mirrors_from_present(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(presentAddress="Dhaka, Bangladesh", permanentAddress=None),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.data.presentAddress == "Dhaka, Bangladesh"
    assert result.data.permanentAddress == "Dhaka, Bangladesh"
    assert any("permanentAddress was empty" in w and "mirrored" in w for w in result.warnings)


def test_extract_nid_both_addresses_present_no_mirroring_warning(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert not any("mirrored" in w for w in result.warnings)


def test_extract_nid_both_addresses_missing_stay_none_no_mirroring_warning(monkeypatch):
    monkeypatch.setattr(
        "app.services.vision_ocr.detect_text",
        _make_ocr_stub("front text", "back text no mrz"),
    )
    monkeypatch.setattr(
        "app.services.gemini.structure_and_translate",
        lambda *a, **kw: _base_gemini_result(presentAddress=None, permanentAddress=None),
    )

    result = pipeline.extract_nid(FRONT_BYTES, "front.jpg", BACK_BYTES, "back.jpg", 10_000_000)

    assert result.data.presentAddress is None
    assert result.data.permanentAddress is None
    assert not any("mirrored" in w for w in result.warnings)
