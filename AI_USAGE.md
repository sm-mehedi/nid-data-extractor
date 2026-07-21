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
