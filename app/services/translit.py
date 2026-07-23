"""Local, deterministic, no-network fallback for names Gemini failed to
translate out of Bengali script.

This is a safety net, not a quality-equivalent substitute for Gemini's own
translation. It runs in two ordered stages:

1. Recognized prefix/title stripping (`_strip_recognized_prefixes`) — titles
   and status markers (Md., Mst., Syed, Sheikh, Alhaj, Haji, Late) are
   matched and replaced directly, never phonetically transliterated. These
   can stack (e.g. "মৃত মোঃ..." -> "Late Md...."), so this repeatedly
   strips recognized prefixes from the front of the string until none
   remain, before anything is handed to the transliteration library.

2. Phonetic transliteration of whatever remains (`transliterate_bengali`),
   via `indic-transliteration`. Two systematic corrections are applied to
   the library's raw output, both because it applies Sanskrit/Devanagari
   Romanization conventions to Bengali text rather than Bengali-specific
   ones (confirmed by testing every scheme the library offers — KOLKATA,
   ITRANS, HK, IAST, OPTITRANS all showed the identical issues, so this is
   not a matter of picking a different preset):

   - The Bengali letter ব is rendered as "v" in every scheme; corrected to
     "b" via a direct character replacement. This is safe as a blanket
     substitution: the other letter that could plausibly produce a "v"-like
     sound, ভ (bha), consistently transliterates to "bh" in this library,
     never to a bare "v" — there's no ambiguity to disambiguate.
   - Extra inherent vowels are inserted that Bengali pronunciation drops
     (schwa deletion) — e.g. "Jamsed" comes out as "Jamaseda". This is a
     genuine, unsolved limitation: no scheme this library offers accounts
     for Bengali's schwa-deletion rules (they're context-dependent — syllable
     position, word-final position, consonant clusters — and were confirmed
     absent by direct testing, not assumed). Building a custom rule set for
     this would be its own significant linguistic effort with real risk of
     introducing new errors, so it has deliberately NOT been attempted here.
     This module does not claim to fix it.

   The library also does not reliably transliterate every Bengali character
   on its own — some conjunct/nukta combinations have been observed passing
   straight through untouched — so a final pass strips any Bengali
   characters that survive. The one strict guarantee this module makes is
   that raw Bengali script never reaches the final output; it makes no
   guarantee about matching Gemini's translation quality or fixing every
   phonetic inaccuracy.

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

# Titles and status markers to detect and replace directly, never
# phonetically transliterate. Ordered longest-Bengali-string-first so a
# longer prefix (e.g. আলহাজ্ব) can never be partially shadowed by a shorter
# one that happens to be one of its own leading substrings (আলহাজ) — sorted
# defensively below rather than relying purely on this list's order.
_RECOGNIZED_PREFIXES: list[tuple[str, str]] = sorted(
    [
        ("মোঃ", "Md."),
        ("মোসাঃ", "Mst."),
        ("সৈয়দ", "Syed"),
        ("শেখ", "Sheikh"),
        ("আলহাজ্ব", "Alhaj"),
        ("আলহাজ", "Alhaj"),
        ("হাজী", "Haji"),
        # Status marker, not a name — indicates the named person (typically
        # a father/mother) is deceased. Confirmed via the Bangladesh
        # Election Commission's own NID correction documentation.
        ("মৃত", "Late"),
    ],
    key=lambda pair: -len(pair[0]),
)


def contains_bengali(text: str | None) -> bool:
    if not text:
        return False
    return bool(BENGALI_RANGE.search(text))


def _strip_recognized_prefixes(text: str) -> tuple[list[str], str]:
    """Repeatedly matches known title/status prefixes at the start of the
    (remaining) text, so stacked prefixes (e.g. "মৃত মোঃ...", "সৈয়দ
    মোঃ...") all resolve in sequence rather than just the first one.
    Returns (matched English replacements in order, unmatched remainder)."""
    remainder = text.strip()
    matched: list[str] = []
    progressed = True
    while progressed:
        progressed = False
        for bengali, english in _RECOGNIZED_PREFIXES:
            if remainder.startswith(bengali):
                matched.append(english)
                remainder = remainder[len(bengali):].lstrip()
                progressed = True
                break
    return matched, remainder


def _fix_known_transliteration_errors(raw: str) -> str:
    """Corrects the one systematic, unambiguous error confirmed across
    every scheme this library offers: ব is rendered as "v" instead of "b".
    Does NOT attempt to fix inherent-vowel insertion (schwa deletion) — see
    module docstring for why that's a genuinely harder, unfixed problem."""
    return raw.replace("v", "b").replace("V", "B")


def transliterate_bengali(text: str) -> str:
    """Best-effort phonetic Bengali -> Roman transliteration, with a strict
    guarantee: the result never contains a Bengali Unicode character
    (U+0980-U+09FF), even in the rare case where the underlying library
    fails to transliterate part of the input. Recognized titles/status
    markers are stripped and replaced directly first (see
    `_strip_recognized_prefixes`); only the remainder is phonetically
    transliterated."""
    if not text:
        return text or ""

    prefixes, remainder = _strip_recognized_prefixes(text)

    transliterated_remainder = ""
    if remainder:
        raw = transliterate(remainder, sanscript.BENGALI, sanscript.ITRANS)
        raw = _fix_known_transliteration_errors(raw)
        cleaned = " ".join(word.capitalize() for word in raw.split())

        # Backstop: strip anything the library didn't actually transliterate,
        # rather than ever letting Bengali script leak into the final output.
        cleaned = BENGALI_RANGE.sub("", cleaned)
        transliterated_remainder = " ".join(cleaned.split())

    parts = prefixes + ([transliterated_remainder] if transliterated_remainder else [])
    return " ".join(parts)
