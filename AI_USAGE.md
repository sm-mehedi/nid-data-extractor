# AI Usage

## Tool(s) used

**Claude** (Anthropic), via **Claude Code** (the CLI agent). Used for both an
earlier planning/design pass (producing `BUILD_PLAN.md`) and this
implementation pass (writing the code, tests, and docs in this repository).

## Example prompts

Planning phase (prior session, condensed):
> "I need to build the Bangladesh NID Extractor from the case study. I don't
> want to depend on a paid/expiring Anthropic API key, and I want to host it
> on Railway. Design the final architecture, response schema, error handling,
> and edge cases we need to test, and write it up as a build plan I can hand
> to a fresh Claude Code session."

This implementation phase:
> "Build according to build plan and give the outputs we want" (pointing at
> `BUILD_PLAN.md` and the original case-study PDF).

Within this session, work was directed module-by-module rather than as one
giant generation — e.g. asking for the image-quality-checks module, then the
MRZ parser, then the Gemini wrapper, then the pipeline orchestrator, each
followed by real test runs against synthetic images before moving on.

## How AI-generated code was verified

- **101 automated tests** (`pytest`) were written and actually executed
  against the real implementation — not just generated and assumed to pass.
  Failures were iterated on until genuinely green (see below).
- The Docker image was built and run end-to-end (`docker build` /
  `docker compose up`), and `/health` plus `/docs` were checked in a real
  browser.
- Every module was read through manually (config, security, image checks,
  OCR wrapper, MRZ parser, Gemini wrapper, pipeline, routes) to confirm the
  logic actually matched the intended design, not just that it "looked
  plausible."

## Bugs found during verification and how they were fixed

These were caught by actually running the test suite and Docker build, not
assumed away:

1. **FastAPI startup crash** — `UploadFile | None` under
   `from __future__ import annotations` made FastAPI treat the type as an
   unresolved forward reference at route-registration time
   (`FastAPIError: Invalid args for response field!`). Fixed by dropping the
   future-annotations import in that file and using `typing.Optional`
   explicitly.
2. **Concurrency cap was not actually enforcing concurrency** — the route
   called the synchronous, CPU/network-bound `pipeline.extract_nid()`
   directly inside `async def`, which blocks the single event loop
   regardless of the semaphore guarding it. Fixed by offloading it via
   Starlette's `run_in_threadpool`, then added a real async test
   (`httpx.AsyncClient` + `asyncio.gather`) that fires two concurrent
   requests against a `Semaphore(1)` and confirms one gets `503` while the
   other completes.
3. **Rate limit value baked in at import time** — the `slowapi` decorator was
   originally `@limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")`,
   evaluated once at module import, so changing the env var later (including
   in tests) had no effect. Switched to a zero-argument callable
   (`@limiter.limit(_rate_limit_value)`) so the limit is read fresh per
   request — matching slowapi's actual dynamic-limit calling convention
   (discovered by reading the `TypeError` it raised when the callable's
   signature didn't match what slowapi expected).
4. **A real `503` was silently becoming a `500`** — the concurrency guard
   raises a FastAPI `HTTPException(503)`, but the route's catch-all
   `except Exception` caught it before it could propagate, logging it as an
   "unexpected error" and returning `500`. Fixed by adding an explicit
   `except HTTPException` branch ahead of the generic catch-all.
5. **Cut-off-card detection didn't work at all, then over-fired** — a card
   whose edge is genuinely outside the photo frame has no edge pixels there
   for Canny to trace, so the original contour search legitimately could
   never find it (verified against a synthetic test image). Added a
   threshold-based fallback for that case — which then falsely flagged
   *the entire background* as a "cut off card" once the binary mask was
   inverted (the background trivially touches all four sides). Fixed by
   excluding near-full-frame blobs and requiring exactly 1–2 sides touched
   (not 3–4, which is the background's signature).
6. **A hand-picked MRZ checksum test value was simply wrong** — an initial
   unit test asserted a made-up "known" check digit that didn't match the
   algorithm. Replaced it with the actual published ICAO 9303 Part 4 TD3
   worked example (`L898902C3` → check digit `6`), computed by hand and
   confirmed against the implementation.
7. **Frontend fell back to a native GET form submission after a real
   deployment** (found post-deploy, on Cloud Run) — image filenames were
   appearing as plain text in the URL query string
   (`?front_image=photo.jpg&back_image=photo.jpg`) instead of the images
   being POSTed as multipart data. Root cause: `script.js` called
   `setupPreview()` for the file-input preview thumbnails *before*
   registering the form's `submit` listener. If `setupPreview` ever threw
   (e.g. a null element lookup), the whole IIFE aborted right there and the
   `event.preventDefault()` handler was never attached at all — so clicking
   "Extract Information" fell through to the browser's default behavior: a
   plain HTML form GET submit, which serializes file inputs as their
   filename string, never actual bytes. Fixed by moving the `submit`
   listener registration to the very first thing the script does
   (unconditional on `form` existing), and wrapping the non-essential
   preview-thumbnail setup in `try/catch` with null-guards so it can never
   again take down the critical submit handler. Verified in a real browser
   (not just read): programmatically attached files via `DataTransfer`,
   clicked the actual button, and confirmed via the network panel that a
   real `POST /api/v1/nid/extract` fired and `window.location.href` never
   changed (no native-submit fallback).
8. **Upload preview thumbnails showed a broken-image icon** — reported
   post-deploy. The suspicion going in was the `URL.createObjectURL(file)` /
   `<img>` wiring in `script.js`, but that code was actually correct. The
   real cause was the response security header:
   `Content-Security-Policy: img-src 'self' data:` — this list of allowed
   image sources didn't include `blob:`, and `URL.createObjectURL()`
   produces a `blob:` URL. The browser silently blocked the `<img>` load as
   a CSP violation, rendering as a broken-image icon, with the JS itself
   never seeing an error. Fixed by adding `blob:` to the `img-src` directive
   in `app/security.py`. Verified in a real browser: generated a real JPEG
   via `canvas.toBlob()`, attached it to the file input via `DataTransfer`,
   and confirmed the resulting `<img>` had non-zero `naturalWidth`/
   `naturalHeight` and `complete: true` — not just that no error was thrown.
9. **Blur threshold false-positived on real WhatsApp-compressed photos** —
   the original `BLUR_VARIANCE_THRESHOLD = 60.0` was calibrated only against
   a synthetic heavy-Gaussian-blur test case, never against realistic
   compression artifacts. The user didn't have file-accessible copies of the
   actual reported photos, so this was recalibrated using a simulated stand-in
   instead of guessed blindly: a mild 5×5 Gaussian blur (representing normal
   handheld focus softness on a real photo, not the heavy blur used to
   simulate genuine blur) run through a WhatsApp-style downscale +
   aggressive JPEG re-encode. That measured ~47-52 Laplacian variance —
   comfortably legible, but below the old 60.0 threshold, reproducing the
   false positive. The existing heavy-blur test case measures ~1.04.
   Lowered the threshold to **35.0**, which clears the legible-but-compressed
   case with margin while still failing the genuinely-blurry case by roughly
   35x. Added `whatsapp_compressed_card_image()` to `tests/helpers.py` and
   two new regression tests locking in both ends of this: one asserting the
   heavy-blur case fails with a wide margin below the threshold (not just
   barely), one asserting the simulated compressed-but-legible case passes.
   This is explicitly a simulated calibration, not one verified against the
   user's literal source photos — flagged as such rather than presented as
   fully confirmed.
10. **Blur/exposure/glare hard-rejected real legible photos even after the
    threshold recalibration** — reported with a live screenshot showing
    "Glare detected over part of the card, please retake." on a real card.
    Recalibrating the blur threshold (#9) wasn't enough on its own: exposure
    and glare had the same structural problem — thresholds tuned against
    synthetic worst-case images, hard-rejecting on any real photo that
    happened to trip them, and real laminated ID cards routinely show *some*
    glare as a physical property of the material even when perfectly
    legible. Changed `check_blur`, `check_exposure`, and `check_glare` from
    raising `ImageQualityError` to returning `str | None` (a warning message,
    or `None` if the check passes); `run_quality_pipeline` now collects
    whichever fire into `warnings` and always continues to Cloud
    Vision/Gemini rather than stopping. Card-cutoff detection was
    deliberately left as a hard rejection — a card missing from the frame
    edge is missing data, not just noisy data, so no amount of downstream
    processing recovers it; that's a difference in kind from blur/exposure/
    glare, not just degree. Updated the corresponding unit tests to assert
    the returned warning string instead of `pytest.raises`, and added new
    pipeline-level and full end-to-end tests
    (`test_extract_nid_blurry_and_glare_photos_still_succeed_end_to_end`)
    confirming a blurry-front + glare-back pair now returns `success: true`
    with both warnings present, not a 400.
11. **Gemini calls timing out on real two-image requests** (`"Gemini request
    timed out: 504 Deadline expired"`) — checked the code first rather than
    assuming: the per-call timeout was hardcoded to `request_options={"timeout":
    30}` inside `structure_and_translate` — 30 seconds, not configurable via
    any env var (there wasn't one), confirming the user's suspicion exactly.
    Separately, checking whether "thinking" could be disabled surfaced a
    bigger issue: the pinned SDK, `google-generativeai==0.8.3`, has no
    `thinking_config`/`thinking_budget` support in *any* version — verified
    by inspecting `GenerationConfig`'s fields directly on both the installed
    version and the latest release (0.8.6), which also now raises a
    `FutureWarning` that the entire package is deprecated in favor of
    `google-genai`. Migrated `app/services/gemini.py` to the new `google-genai`
    SDK (`genai.Client(...)` / `client.models.generate_content(...)`) rather
    than working around a dead-end package. This required bumping
    `pydantic` from `2.10.4` to `2.12.5` in `requirements.txt` — `google-genai`
    hard-requires `pydantic>=2.12.5`, discovered via a real
    `ResolutionImpossible` from pip, not guessed. Added `GEMINI_TIMEOUT_SECONDS`
    (default 100s), `GEMINI_MAX_RETRIES` (default 1), and
    `GEMINI_RETRY_BACKOFF_SECONDS` (default 2.0) as configurable settings;
    `thinking_config=ThinkingConfig(thinking_budget=0)` is applied by default
    and the call transparently retries once without it if a model rejects
    the field with a 400 (verified this exact fallback path with a test that
    asserts the first attempt's config had `thinking_config` set and the
    retry's didn't). Retry-with-backoff applies only to 429 (rate limit) and
    timeout/504 (deadline exceeded) — verified both the "retries then
    succeeds" and "retries exhausted then fails" paths, plus that a
    non-retryable error (401 auth failure) fails on the very first attempt
    rather than wasting retries on an error a retry can't fix. Also confirmed
    the arithmetic on retry-timeout interaction: worst case for the Gemini
    stage alone is `(GEMINI_MAX_RETRIES + 1) × GEMINI_TIMEOUT_SECONDS` = 200s
    with the defaults, leaving roughly 40s of headroom under a 300s platform
    request timeout (e.g. Cloud Run) once the Cloud Vision stage's own worst
    case (~60s, unchanged) is added — documented in the README so this isn't
    silently outgrown by a future config change. Verified end-to-end: full
    test suite (118 tests) passes, and the Docker image was rebuilt and
    smoke-tested (`/health` responds, container starts cleanly) with the new
    dependency actually installed, not just assumed to resolve.
12. **503 (model overloaded) wasn't actually being retried at all** — a real
    request logged `"Gemini request failed: 503 UNAVAILABLE... Spikes in
    demand are usually temporary"` after ~39 seconds. Read the actual retry
    code before assuming it was working: `_is_rate_limited` only matched 429,
    `_is_timeout` only matched 504 — 503 matched neither, so it fell straight
    through `_classify_error` to a hard failure on the very first attempt.
    The ~39s was Gemini's own server-side response time for that one failed
    attempt, not evidence of multiple retries happening. Added `_is_overloaded`
    for 503 specifically, and replaced the old single-multiplier backoff
    (`backoff_seconds * (attempt + 1)`, max 1 retry) with a schedule-based one
    driven by `GEMINI_RETRY_BACKOFF_SECONDS` (default `"3,8,15"`, a
    comma-separated list — length of the list *is* the retry count, one
    source of truth instead of two settings that could drift apart). Verified
    with a test that captures every `time.sleep()` call and asserts the exact
    sequence `[3.0, 8.0, 15.0]` was used — not just that *a* retry happened,
    which the previous test suite's assertions would have let slide. Also
    added a fallback-model attempt (`GEMINI_FALLBACK_MODEL`, auto-derives
    `<model>-lite` if unset) that fires exactly once, with no retries of its
    own, only when the primary model's retries are exhausted *and* the final
    error is specifically a 503 — verified this doesn't fire for a rate-limit
    exhaustion (`test_structure_and_translate_no_fallback_when_error_is_not_
    overloaded`), and that both the auto-derived and explicitly-overridden
    fallback model names are actually passed through to the API call, not
    just computed and discarded. Recomputed the Cloud Run timeout headroom
    math from scratch for the new, much longer worst case (~586s with
    defaults, up from ~260s) and updated the README prominently rather than
    letting the old, now-wrong number stand — this schedule genuinely
    requires raising the platform's request timeout well past its default.
13. **Rate limiter likely not scoping by real client IP on Cloud Run** —
    found during a requested read-only investigation (not a reported bug):
    the Dockerfile's `uvicorn` invocation had no `--proxy-headers`/
    `--forwarded-allow-ips` flags, and nothing in `app/` handled
    `X-Forwarded-For`. Verified precisely rather than assuming: uvicorn's
    `proxy_headers` defaults to `True` already, but `forwarded_allow_ips`
    defaults to trusting only `127.0.0.1` (confirmed by reading
    `uvicorn/config.py` directly) — Cloud Run's front-end proxy doesn't
    connect from there, so `X-Forwarded-For` was being silently ignored and
    `request.client.host` resolved to the proxy's own connection. Fixed two
    ways: `--proxy-headers --forwarded-allow-ips='*'` added to the
    Dockerfile's `CMD`, and `ProxyHeadersMiddleware` added directly in
    `app/main.py` — the latter matters because the CLI flags only take
    effect when uvicorn itself launches the app; a pytest test hitting the
    ASGI `app` object directly (via `TestClient`/`ASGITransport`, as this
    whole suite does) bypasses that layer entirely, so the middleware had to
    be in the app itself to be testable at all. Verified with a real
    red/green check, not just a passing test: temporarily reverted the
    middleware line, re-ran the new regression test, and confirmed it failed
    with both simulated IPs collapsing into a single `127.0.0.1` bucket in
    the logs (proving the exact bug) — then restored the fix and confirmed
    it passed. All 125 tests pass; Docker rebuilt and smoke-tested.
14. **Gemini's own consistency isn't reliable enough to trust for two
    specific fields, so both got a deterministic code-level safety net**
    instead of a stronger prompt:
    - *Name translation*: a real test showed `fatherName`/`motherName`
      returned as raw untranslated Bengali script while `presentAddress`
      translated correctly in that same response — not a broken prompt,
      the model just doesn't follow the instruction reliably every call.
      Added `app/services/translit.py`: after the response comes back, each
      of `name`/`fatherName`/`motherName` is checked for Bengali Unicode
      characters (U+0980–U+09FF); anything that still has them gets a
      local, offline phonetic transliteration via `indic-transliteration`
      instead of being returned untranslated. Investigated the library's
      actual output quality before trusting it rather than assuming a
      transliteration library "just works" — it targets Sanskrit-style
      scholarly Romanization, not Bengali colloquial name spelling, so raw
      output reads oddly (e.g. "Avdula Karima" for "আব্দুল করিম", expected
      "Abdul Karim" — the "a" endings and b/v swap come from the library's
      Sanskrit-oriented rules, not a bug in how it's called here). Also
      found, by actually testing multiple real Bengali name samples rather
      than a single happy-path input, that the library itself doesn't
      reliably transliterate every character — a name containing a nukta
      combination left one raw Bengali character behind untouched. Added a
      second, stricter pass that strips any Bengali characters surviving
      the library's output, so the one guarantee this module makes (no
      Bengali script ever reaches the final response) holds even when the
      library itself only partially succeeds. This is documented plainly as
      a rough safety net, not a quality-equivalent substitute for Gemini's
      translation.
    - *Address mirroring*: confirmed by two different real test runs of the
      identical single-address card design — mirrored correctly on one run,
      returned `null` with a low-confidence flag on another. Added
      `pipeline._mirror_addresses`: if either `presentAddress` or
      `permanentAddress` is empty and the other has a value, the value is
      copied across unconditionally in code, with a warning noting both
      reflect the same source address. When both already match (the
      expected case), nothing fires — same reasoning as the existing
      single-address documentation, firing on every request would stop
      being a useful signal.
    - Both were tested without depending on a live Gemini call reproducing
      the inconsistency on demand — the exact reported scenarios (one name
      field translated, one left in Bengali, in the same response; one
      address field present, one null) are mocked directly, so the fallback
      is verified deterministically on every test run regardless of
      whether the live model happens to be consistent that day.
      `tests/test_translit.py` also verifies the transliteration module in
      isolation, including the nukta-leak case, independent of the pipeline.
      All 136 tests pass; Docker rebuilt and smoke-tested with the new
      `indic-transliteration` dependency actually installed.
15. **Two follow-up refinements to the transliteration fallback (#14),
    handled as separate, ordered concerns rather than one blended change:**
    - *Prefix/title recognition*: no prefix-detection code actually existed
      anywhere in the codebase before this — the earlier honorifics test
      only confirmed that Gemini's own already-translated "Md. Rahim" passed
      through unmodified, not that there was dedicated logic behind it. Built
      `_strip_recognized_prefixes` in `app/services/translit.py`: matches
      Md., Mst., Syed, Sheikh, Alhaj (both "আলহাজ্ব" and "আলহাজ" variants),
      Haji, and Late (a status marker for a deceased father/mother, confirmed
      against the Bangladesh Election Commission's own NID correction
      documentation — not part of the name itself) and replaces them
      directly, never phonetically. These stack (e.g. "মৃত মোঃ..." ->
      "Late Md...."), so the matcher repeats until no more recognized
      prefixes remain at the front of the string, rather than matching only
      once. The prefix list is sorted longest-Bengali-string-first so
      "আলহাজ্ব" can't be shadowed by "আলহাজ" (its own leading substring) —
      verified with a dedicated test, not just asserted.
    - *Systematic phonetic errors*: rather than guess at fixes, tested the
      two reported issues against every scheme the library actually offers
      (KOLKATA, ITRANS, HK, IAST, OPTITRANS) first. Confirmed ব→"v" is
      identical across all five (not a matter of picking a better preset),
      and confirmed it's safe to correct as a blanket substitution by
      checking whether ভ (bha) — the other letter that could plausibly
      produce a "v" sound — ever collides with the fix; it consistently
      transliterates to "bh", never a bare "v", so there's no ambiguity.
      Applied the ব→b correction with a test that specifically guards
      against it breaking ভ's already-correct mapping. For the
      inherent-vowel/schwa-deletion issue, the same scheme-by-scheme testing
      confirmed none of the five accounts for it — this was verified
      empirically, not assumed, before concluding it wasn't a "pick a
      different preset" fix. Per explicit instruction not to force a
      fragile rule-based fix without something reliable to build on, this
      was left unfixed and documented plainly as a known limitation, with a
      test (`test_known_unfixed_limitation_inherent_vowel_insertion`) that
      makes the gap regression-visible rather than silently absent from the
      suite.
    - Confirmed both the pre-existing Md./Mst. honorifics pass-through test
      and the nukta-leak backstop test still pass, unaffected by either
      change — run explicitly by name, not just inferred from the full
      suite passing. All 144 tests pass; Docker rebuilt and smoke-tested.

## Modifications made to AI-generated code, and why

Beyond the bug fixes above (all of which required actual code changes, not
just re-prompting):

- The MRZ cross-check logic was deliberately placed in the pipeline's merge
  step rather than inside the MRZ parser itself, since it inherently needs
  both the MRZ result *and* Gemini's front-derived fields to compare — the
  parser alone has no way to see both sides at once.
- The frontend's rendering code was written to use `textContent` exclusively
  (never `innerHTML`) when displaying extracted card data, per the security
  checklist in the build plan — extracted fields are untrusted OCR/LLM
  output and must not be interpreted as HTML.
