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
