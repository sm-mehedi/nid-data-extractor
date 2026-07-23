from app.services import translit


def test_contains_bengali_detects_bengali_script():
    assert translit.contains_bengali("রহিম") is True
    assert translit.contains_bengali("Md. Rahim") is False
    assert translit.contains_bengali("") is False
    assert translit.contains_bengali(None) is False


def test_contains_bengali_detects_mixed_script():
    # Partial-failure case: some words translated, one word still Bengali.
    assert translit.contains_bengali("Abdul করিম") is True


def test_transliterate_bengali_never_leaves_bengali_characters():
    samples = [
        "রহিম",
        "আব্দুল করিম",
        "আমেনা বেগম",
        "মোঃ আব্দুল্লাহ হাসান",
        "মোসাঃ আনোয়ারা বেগম",  # contains a nukta combination the library may not fully handle
    ]
    for text in samples:
        result = translit.transliterate_bengali(text)
        assert not translit.contains_bengali(result), f"leaked Bengali script in: {result!r}"
        assert result.strip() != ""


def test_transliterate_bengali_produces_legible_latin_output():
    result = translit.transliterate_bengali("আব্দুল করিম")
    assert result.isascii()
    assert any(c.isalpha() for c in result)


def test_transliterate_bengali_empty_input():
    assert translit.transliterate_bengali("") == ""
    assert translit.transliterate_bengali(None) == ""


# --- Concern 1: recognized prefix/title detection --------------------------


def test_recognized_prefixes_detected_and_replaced_directly():
    assert translit.transliterate_bengali("মোঃ রহিম").startswith("Md.")
    assert translit.transliterate_bengali("মোসাঃ বেগম").startswith("Mst.")
    assert translit.transliterate_bengali("সৈয়দ রহিম").startswith("Syed")
    assert translit.transliterate_bengali("শেখ হাসিনা").startswith("Sheikh")
    assert translit.transliterate_bengali("আলহাজ্ব রহিম").startswith("Alhaj")
    assert translit.transliterate_bengali("আলহাজ রহিম").startswith("Alhaj")
    assert translit.transliterate_bengali("হাজী করিম").startswith("Haji")
    assert translit.transliterate_bengali("মৃত রহিম").startswith("Late")


def test_recognized_prefixes_are_never_phonetically_transliterated():
    # If these leaked into the phonetic path they'd come out garbled (e.g.
    # "মোঃ" alone would transliterate to something like "Moh") rather than
    # the clean, direct replacement.
    assert "Moh" not in translit.transliterate_bengali("মোঃ রহিম")
    assert "Mrrita" not in translit.transliterate_bengali("মৃত রহিম")


def test_stacked_prefixes_compose_in_sequence():
    # The specific case this was built for: multiple recognized prefixes in
    # a row must all resolve, not just the first one.
    assert translit.transliterate_bengali("মৃত মোঃ রহিম") == "Late Md. Rahima"
    assert translit.transliterate_bengali("সৈয়দ মোঃ রহিম") == "Syed Md. Rahima"


def test_longer_prefix_variant_not_shadowed_by_shorter_substring():
    # আলহাজ is a leading substring of আলহাজ্ব — the longer one must still
    # match in full, not get cut short leaving a dangling remainder.
    result = translit.transliterate_bengali("আলহাজ্ব রহিম")
    assert result == "Alhaj Rahima"
    assert not translit.contains_bengali(result)


def test_prefix_only_input_produces_no_dangling_output():
    result = translit.transliterate_bengali("মোঃ")
    assert result == "Md."


# --- Concern 2: systematic phonetic transliteration errors ------------------


def test_ba_letter_renders_as_b_not_v():
    # Confirmed example from the report: "Veoya" should be "Bewa".
    result = translit.transliterate_bengali("বেওয়া")
    assert result.startswith("B")
    assert "v" not in result.lower()


def test_bha_letter_still_renders_correctly_after_v_to_b_fix():
    # ভ (bha) must still come out as "Bh...", not be affected by the ব->b
    # correction in a way that breaks its own (already-correct) "bh" mapping.
    result = translit.transliterate_bengali("ভালো")
    assert result.lower().startswith("bh")


def test_known_unfixed_limitation_inherent_vowel_insertion():
    # Documents the confirmed, NOT-fixed limitation: the library inserts
    # inherent vowels Bengali pronunciation drops. This test exists to make
    # the limitation explicit and regression-visible, not to assert correct
    # output — if this ever starts passing with exact matches, the
    # module docstring's claim that this is unfixed should be revisited.
    result = translit.transliterate_bengali("জামসেদ")
    assert result != "Jamsed"  # the linguistically correct form
    assert not translit.contains_bengali(result)  # but still no Bengali leak
