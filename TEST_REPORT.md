# Test Coverage Report

**Summary: the build plan's Section 11 matrix maps to 46 sub-cases below**
(one bullet — rotation — is split into two rows: real pixel rotation vs.
EXIF-only rotation, since they turned out to have different outcomes; four
further rows were added for deterministic code-level guarantees introduced
after real observed model inconsistencies — see #42-#45). **40 are fully
automated, 3 are automated with a noted caveat (not airtight), 1 is a mixed
row (one half fixed and automated, one half explicitly confirmed-unfixed —
see #45), and 2 are not covered** (one a known implementation gap, one
explicitly out-of-scope per the build plan itself). 144 tests pass in total.
No real Cloud Vision/Gemini API keys or network access were needed to run
the suite — every upstream call is mocked/monkeypatched.

Run it yourself: `pip install -r requirements-dev.txt && pytest -v`

Legend: ✅ Automated · ⚠️ Manual only / partial · ❌ Not covered

---

## Upload / request shape

| # | Case | Status | Notes |
|---|---|---|---|
| 1 | Missing front only | ✅ | `test_missing_front_only` (test_api.py) |
| 2 | Missing back only | ✅ | `test_missing_back_only` |
| 3 | Missing both | ✅ | `test_missing_both_files` |
| 4 | Wrong extension (.gif/.bmp/.webp/.pdf/.txt) | ✅ | `test_validate_extension_rejects_disallowed[...]` (parametrized, test_image_checks.py) + `test_wrong_extension` (API) |
| 5 | Extension says image, content isn't | ✅ | `test_decode_image_corrupt_bytes`, `test_decode_image_txt_renamed_jpg`, `test_corrupt_bytes_with_valid_extension` |
| 6 | Empty file (0 bytes) | ✅ | `test_validate_size_empty`, `test_empty_file` |
| 7 | File at exactly the size limit | ✅ | `test_validate_size_at_exact_limit_ok` |
| 8 | File over the limit | ✅ | `test_validate_size_over_limit`, `test_oversized_file`, `test_run_quality_pipeline_oversized` |
| 9 | Tiny image (10×10px) | ✅ | `test_decode_image_tiny`, `test_tiny_image` |
| 10 | Malformed/non-multipart request body | ✅ | `test_malformed_non_multipart_body` — confirms a clean 400/422, not a crash |
| 11 | Same image for both front and back | ✅ | `test_extract_nid_same_image_both_sides_does_not_crash` |
| 12 | Front and back swapped | ✅ | `test_extract_nid_front_back_swapped_does_not_crash` |

## Image quality

| # | Case | Status | Notes |
|---|---|---|---|
| 13 | Heavily blurred | ✅ | **Soft check (200/warning, not a 400).** `test_blur_check_warns_on_blurry_image_but_does_not_raise`, `test_run_quality_pipeline_blurry_photo_succeeds_with_warning`, `test_extract_nid_blurry_and_glare_photos_still_succeed_end_to_end`. Changed from a hard 400 after real WhatsApp-compressed (legible) photos were false-positive rejected. |
| 14 | Under-exposed (too dark) | ✅ | **Soft check (200/warning).** `test_exposure_check_warns_on_dark_image_but_does_not_raise`, `test_run_quality_pipeline_dark_photo_succeeds_with_warning` |
| 15 | Over-exposed (washed out) | ✅ | **Soft check (200/warning).** `test_exposure_check_warns_on_overexposed_image_but_does_not_raise`, `test_run_quality_pipeline_overexposed_photo_succeeds_with_warning` |
| 16 | Localized glare over key text | ✅ | **Soft check (200/warning).** `test_glare_check_warns_on_localized_bright_blob_but_does_not_raise`, `test_run_quality_pipeline_glare_photo_succeeds_with_warning`, `test_extract_nid_blurry_and_glare_photos_still_succeed_end_to_end`. Real laminated cards routinely show some glare as a physical property of the material even when fully legible, so this no longer hard-rejects. |
| 17a | Rotated 90°/180°/270° (real pixel rotation) | ✅ | `test_run_quality_pipeline_handles_rotated_images[90\|180\|270]` |
| 17b | Rotated via EXIF orientation flag only | ❌ | **Not covered.** `cv2.imdecode` does not reliably apply EXIF orientation across OpenCV builds, and the pipeline does no explicit EXIF-tag correction. A photo that is only EXIF-rotated (pixels unrotated) may be processed in the wrong orientation. Deprioritized for this build; a real fix would use Pillow's `ImageOps.exif_transpose()` before handing the image to OpenCV. |
| 18 | Cut off at edge vs. small/unzoomed (must differ) | ✅ | `test_card_boundary_cut_off_at_edge`/`test_run_quality_pipeline_cut_off_raises_400` (hard 400) vs. `test_card_boundary_not_found_for_small_unzoomed_card`/`test_run_quality_pipeline_small_unzoomed_produces_warning_not_error` (soft warning) — confirmed to produce genuinely different outcomes |
| 19 | Low-resolution/upscaled photo | ⚠️ | Only the hard floor (`MIN_DIMENSION_PX` rejection, i.e. "tiny image") is asserted. A photo that's low-res but above that floor is not given a dedicated resolution-quality warning/test — it will simply proceed through the same path as any other image. |

## Content / authenticity

| # | Case | Status | Notes |
|---|---|---|---|
| 20 | Not an NID at all (random object/receipt/passport) | ✅ | `test_extract_nid_not_a_card_raises`, `test_not_nid_card_returns_422`, `test_run_quality_pipeline_non_card_image_falls_through_to_warning` |
| 21 | Front/back belong to two different cards | ✅ | `test_extract_nid_mrz_cross_check_mismatch_warns_but_succeeds` |
| 22 | MRZ checksum fails (corrupted/misread) | ✅ | `test_parse_mrz_corrupted_document_number_checksum`, `..._dob_checksum`, `..._composite_checksum` |
| 23 | MRZ present but unparseable/garbled | ✅ | `test_parse_mrz_garbled_unparseable`, `test_extract_nid_mrz_unparseable_produces_warning` |
| 24 | Photo of a screen/photocopy, not the physical card | ❌ | **Explicitly out of scope.** No moiré-pattern or screen-glare-signature detection is implemented (per the build plan's Section 6 scope: "no forgery/tamper detection"). Would require a dedicated model/heuristic beyond this build's budget. |

## Field-level extraction

| # | Case | Status | Notes |
|---|---|---|---|
| 25 | One side unreadable, other fine | ✅ | `test_extract_nid_one_side_unreadable_produces_side_specific_warning` — confirms the warning names the correct side and the affected field is `null` |
| 26 | Some fields illegible, others fine (partial success) | ✅ | `test_extract_nid_low_confidence_fields_produce_warnings`, `test_extract_nid_partial_translation_failure_gets_transliteration_fallback` (real observed case: one name field translated correctly, another left as raw Bengali script, in the same response) |
| 27 | Honorifics (Md., Mst., Dr.) | ✅ | `test_extract_nid_name_with_honorifics_passed_through` |
| 28 | Bengali numerals (০-৯) normalized to Latin | ✅ | `test_extract_nid_bengali_digits_normalized` |
| 29 | Long, multi-line addresses | ✅ | `test_extract_nid_long_multiline_address_preserved` |

## Upstream API failures

| # | Case | Status | Notes |
|---|---|---|---|
| 30 | Missing/invalid Vision/Gemini API keys | ✅ | `test_detect_text_no_credentials_configured`, `test_structure_and_translate_no_api_key` |
| 31 | Cloud Vision returns empty text | ✅ | `test_detect_text_empty_responses` (API-level), `test_extract_nid_empty_ocr_text_both_sides_still_succeeds_via_gemini` (pipeline still completes via Gemini) |
| 32 | Gemini returns non-JSON or JSON missing keys | ✅ | `test_parse_gemini_response_non_json_raises`, `test_parse_gemini_response_missing_keys_filled_with_none`, `test_structure_and_translate_non_json_response` |
| 33 | Simulated timeout, 429, 5xx from either API | ✅ | `test_detect_text_rate_limited`, `test_detect_text_server_error`, `test_detect_text_network_error`, `test_structure_and_translate_rate_limit_error_not_retried_past_schedule`, `test_structure_and_translate_timeout_error_not_retried_past_schedule`, `test_structure_and_translate_503_is_retried_with_default_schedule`, `test_structure_and_translate_retries_and_then_succeeds_on_rate_limit`, `test_structure_and_translate_retries_on_client_side_timeout_exception`, `test_structure_and_translate_non_retryable_error_fails_immediately`, `test_structure_and_translate_falls_back_to_lite_model_after_503_exhausts_retries`, `test_structure_and_translate_uses_explicit_fallback_model_override`, `test_structure_and_translate_fallback_model_also_fails`, `test_structure_and_translate_no_fallback_when_error_is_not_overloaded`. Gemini 429/503/timeout now retry with an escalating backoff schedule (default `3,8,15`s — verified as the *exact* sleep sequence used, not just "a retry happened"); 503 was previously not retried at all (the actual reported bug). A fallback model gets one extra attempt if the primary exhausts retries on a 503 specifically; non-retryable errors (e.g. auth) fail on the first attempt with no fallback. |
| 34 | APIs disagree (Vision reads text, Gemini says not-a-card) | ✅ | `test_extract_nid_not_a_card_raises` |

## Concurrency / abuse

| # | Case | Status | Notes |
|---|---|---|---|
| 35 | Rate limiter triggers past threshold → clean 429 | ✅ | `test_rate_limiter_triggers_past_threshold`, `test_rate_limiter_response_shape_on_429` |
| 36 | Concurrent requests don't exceed the concurrency cap | ✅ | `test_concurrency_cap_returns_503_for_excess_requests` (real `asyncio.gather` against a `Semaphore(1)`), `test_concurrency_within_cap_both_succeed` |
| 37 | Repeated rapid requests stay stable, no memory growth | ⚠️ | `test_repeated_rapid_requests_stay_stable` confirms no crash/exception across 15 sequential requests. Actual **memory-growth measurement** (e.g. via `tracemalloc` over thousands of requests) is manual-only — see checklist below. |

## Response contract

| # | Case | Status | Notes |
|---|---|---|---|
| 38 | `success: true` ⇒ `data` non-null, `errors: []`, always | ✅ | `test_response_contract_success_shape` + holds across every success-path test |
| 39 | `success: false` ⇒ `data: null`, `errors` non-empty, always | ✅ | `test_response_contract_failure_shape` + holds across every failure-path test |
| 40 | `warnings` empty on a normal single-address card | ✅ | `test_extract_nid_full_success` asserts `warnings == []` |
| 41 | `nidNumber` always digits-only; `dateOfBirth` always ISO | ✅ / ⚠️ | `nidNumber` normalization is enforced in code (`_digits_only`) and tested (`test_extract_nid_bengali_digits_normalized`) regardless of what Gemini returns. `dateOfBirth` is **guaranteed** ISO only when MRZ checksums pass (computed directly by our own code); when MRZ is absent/invalid, the date comes from Gemini's own output, which is instructed via prompt to use ISO format but not independently re-validated/reformatted in code. This is a real gap, not just an untested one. |
| 42 | `name`/`fatherName`/`motherName` never contain untranslated Bengali script, regardless of what Gemini returns | ✅ | `test_extract_nid_partial_translation_failure_gets_transliteration_fallback`, `test_extract_nid_fully_translated_names_produce_no_fallback_warning`, plus unit tests in `tests/test_translit.py` (`test_transliterate_bengali_never_leaves_bengali_characters` — including a nukta-combination input the underlying transliteration library does not fully handle on its own, confirming the strip-backstop actually catches what the library misses). Enforced deterministically in code (`pipeline._ensure_translated`), not left to Gemini's translation being reliable every call — added after a real observed case of `fatherName` returning raw Bengali script in the same response where `presentAddress` translated correctly. |
| 43 | `presentAddress`/`permanentAddress` mirrored deterministically when one is empty | ✅ | `test_extract_nid_missing_present_address_mirrors_from_permanent`, `test_extract_nid_missing_permanent_address_mirrors_from_present`, `test_extract_nid_both_addresses_present_no_mirroring_warning`, `test_extract_nid_both_addresses_missing_stay_none_no_mirroring_warning`. Enforced unconditionally in code (`pipeline._mirror_addresses`) rather than relying on Gemini to mirror the single-address card design correctly every call — added after two real runs of the same card design produced different results (one correctly mirrored, one returned `null` with a low-confidence flag). |
| 44 | Recognized name prefixes/titles (Md., Mst., Syed, Sheikh, Alhaj, Haji, Late) detected and replaced directly, including stacked prefixes | ✅ | `test_recognized_prefixes_detected_and_replaced_directly`, `test_recognized_prefixes_are_never_phonetically_transliterated`, `test_stacked_prefixes_compose_in_sequence` (the specific "মৃত মোঃ" -> "Late Md." case), `test_longer_prefix_variant_not_shadowed_by_shorter_substring`, `test_prefix_only_input_produces_no_dangling_output`. Confirmed the pre-existing Md./Mst. pass-through test (`test_extract_nid_name_with_honorifics_passed_through`) and the nukta-leak backstop test both still pass unaffected. |
| 45 | Systematic transliteration errors in the phonetic fallback (ব→v, inherent vowel insertion) | ✅ / ❌ | **ব→b fix: automated and verified.** `test_ba_letter_renders_as_b_not_v`, `test_bha_letter_still_renders_correctly_after_v_to_b_fix` (confirms the blanket substitution doesn't clobber ভ's already-correct "bh" mapping). Verified empirically across every scheme the library offers (KOLKATA, ITRANS, HK, IAST, OPTITRANS all showed the identical ব→v issue) before applying a fix, rather than assuming a different preset would solve it. **Inherent-vowel/schwa-deletion: confirmed NOT fixed, deliberately.** `test_known_unfixed_limitation_inherent_vowel_insertion` documents this as a known limitation rather than silently omitting it — no scheme this library offers accounts for Bengali's schwa-deletion rules, and a custom rule set was judged too risky to attempt reliably in the time available. |

---

## Manual-only checklist (needs real API keys / network — not simulated here)

These require an actual Google Cloud Vision + Gemini account and cannot be
meaningfully faked without reimplementing Google's services:

1. **Real NID photo, full happy path.** Take an actual (or sample/test)
   Bangladesh NID front + back photo, set real `GOOGLE_CLOUD_VISION_API_KEY`
   and `GEMINI_API_KEY` in `.env`, `docker compose up`, and POST both images
   to `/api/v1/nid/extract`. Confirm all 7 fields populate correctly and
   `warnings: []`.
2. **Real Cloud Vision Bengali OCR quality** on an actual printed card —
   confirm Bengali script is read accurately enough for Gemini's translation
   step to work from real (not synthetic) OCR text.
3. **Real Gemini translation quality** — verify translations of a real
   name/address preserve meaning rather than reading as literal/awkward
   word-for-word output, per the build plan's translation requirement.
4. **Real rate-limit behavior against Google's actual free-tier quota** —
   confirm Cloud Vision (1,000 free units/month) and Gemini
   (~10–15 req/min) quota errors surface through our 503 path exactly as
   they do when mocked.
5. **Railway deployment** — confirm `$PORT` binding, cold-start time, and
   TLS termination behave as expected once actually deployed (not just
   `docker run` locally).

## Explicitly out of scope (not attempted, by design)

- **Forgery/tamper detection** and **government NID registry verification** —
  stated as out of scope in the build plan itself (Section 6). This system
  verifies internal consistency (MRZ checksums, front/back cross-check, AI
  plausibility judgment) only.
- **Face-match between the card photo and a live selfie** — not part of the
  assignment's functional requirements.
- **The 1D barcode block** on the back of real cards (separate from the
  MRZ) — no documented open standard for it, so it is not decoded.
