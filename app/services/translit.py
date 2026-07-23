"""Local, deterministic, no-network fallback for names Gemini failed to
translate out of Bengali script.

This is a safety net, not a quality-equivalent substitute for Gemini's own
translation: `indic-transliteration` targets Sanskrit-style scholarly
Romanization schemes, not Bengali colloquial name-spelling conventions, so
its output reads noticeably rougher than Gemini's (e.g. "Avdula Karima"
rather than "Abdul Karim"). It also does not reliably transliterate every
Bengali character on its own — some conjunct/nukta combinations have been
observed passing straight through untouched — so a second pass strips any
Bengali characters that survive. The one guarantee this module makes is
that raw Bengali script never reaches the final output; it makes no
guarantee about matching Gemini's translation quality.

Transliteration (phonetic Romanization), not translation, is also the
linguistically correct operation for personal names in the first place —
names don't have meaning-preserving "translations", which is why this is a
reasonable fallback specifically for name/fatherName/motherName and not for
address fields.
"""
from __future__ import annotations

import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

BENGALI_RANGE = re.compile(r"[ঀ-৿]")


def contains_bengali(text: str | None) -> bool:
    if not text:
        return False
    return bool(BENGALI_RANGE.search(text))


def transliterate_bengali(text: str) -> str:
    """Best-effort phonetic Bengali -> Roman transliteration, with a strict
    guarantee: the result never contains a Bengali Unicode character
    (U+0980-U+09FF), even in the rare case where the underlying library
    fails to transliterate part of the input."""
    if not text:
        return text or ""

    raw = transliterate(text, sanscript.BENGALI, sanscript.ITRANS)
    cleaned = " ".join(word.capitalize() for word in raw.split())

    # Backstop: strip anything the library didn't actually transliterate,
    # rather than ever letting Bengali script leak into the final output.
    cleaned = BENGALI_RANGE.sub("", cleaned)
    cleaned = " ".join(cleaned.split())  # collapse any double-spaces left behind

    return cleaned
