# API Documentation

Base URL (local): `http://localhost:8000`
Interactive Swagger UI: `GET /docs` · ReDoc: `GET /redoc`

---

## `GET /health`

Liveness check.

**Response `200`**
```json
{ "status": "ok" }
```

---

## `POST /api/v1/nid/extract`

Extracts structured data from a Bangladesh NID card's front and back photos.

### Request

`multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `front_image` | file | yes | JPG, JPEG, or PNG |
| `back_image` | file | yes | JPG, JPEG, or PNG |

Optional header:

| Header | Purpose |
|---|---|
| `X-API-Key` | Required only if `API_SHARED_SECRET` is configured server-side |

**curl example**
```bash
curl -X POST http://localhost:8000/api/v1/nid/extract \
  -F "front_image=@/path/to/front.jpg" \
  -F "back_image=@/path/to/back.jpg"
```

### Response body shape

```json
{
  "success": true,
  "data": {
    "name": "Md. Rahim",
    "fatherName": "Abdul Karim",
    "motherName": "Amena Begum",
    "dateOfBirth": "1998-01-15",
    "nidNumber": "1234567890123",
    "presentAddress": "Village Rampur, Upazila Debidwar, District Cumilla",
    "permanentAddress": "Village Rampur, Upazila Debidwar, District Cumilla"
  },
  "warnings": [],
  "errors": []
}
```

- `success: true` ⇒ `data` is present (individual fields may still be `null`
  if genuinely unreadable), `errors` is `[]`.
- `success: false` ⇒ `data` is `null`, `errors` has at least one message.
- `warnings` flags things unusual to *this specific* upload — a
  low-confidence/unreadable field, or a front/back MRZ mismatch. It is not
  used for the expected single-address behavior described in the README.
- `nidNumber` is always digits-only; `dateOfBirth` is always ISO
  (`YYYY-MM-DD`) — regardless of the digit script or date format printed on
  the card.

### Error responses

| Situation | HTTP status | `success` |
|---|---|---|
| Missing `front_image` and/or `back_image` | 400 | false |
| Wrong extension / corrupt / empty / oversized file | 400 | false |
| Card cropped/cut off at the edge of the original photo | 400 | false |
| No card-like content found at all (per AI judgment) | 422 | false |
| Blurry / dark / overexposed / glare on the card | 200 | true (warning, best-effort data still returned) |
| Front/back MRZ cross-check mismatch | 200 | true (warning, best-effort data still returned) |
| Some fields unreadable, others fine | 200 | true (warnings list which fields) |
| Cloud Vision or Gemini error/timeout/rate-limit | 503 | false |
| Server at concurrency capacity | 503 | false |
| Per-IP rate limit exceeded | 429 | false |
| Unexpected server error | 500 | false |

Blur, exposure, and glare are deliberately **soft** checks, not hard
rejections: real photos (especially recompressed by messaging apps, or a
laminated card catching some glare) routinely trip these thresholds while
still being perfectly legible to Cloud Vision/Gemini, so the request
proceeds anyway and a descriptive note is added to `warnings` instead of
blocking it. A cut-off card is different in kind — it's genuinely missing
data no downstream processing can recover — so that one still hard-rejects.

**Example — validation failure (400)**
```json
{
  "success": false,
  "data": null,
  "warnings": [],
  "errors": ["Card appears cut off at the edge of the photo; please retake showing the full card."]
}
```

**Example — blurry but legible photo (200, warning only)**
```json
{
  "success": true,
  "data": { "...": "..." },
  "warnings": ["Photo may be blurry — some fields may be less reliable."],
  "errors": []
}
```

**Example — not an NID card (422)**
```json
{
  "success": false,
  "data": null,
  "warnings": [],
  "errors": ["The uploaded images do not appear to be a Bangladesh NID card (front and/or back)."]
}
```

**Example — MRZ mismatch (200, warning only)**
```json
{
  "success": true,
  "data": { "...": "..." },
  "warnings": ["Front/back may not match: NID number from Gemini differs from the MRZ-verified document number."],
  "errors": []
}
```

**Example — a name field needed the local transliteration fallback (200, warning only)**
```json
{
  "success": true,
  "data": { "fatherName": "Avdula Karima", "...": "..." },
  "warnings": ["fatherName was not translated by the AI model; applied a local phonetic transliteration fallback (approximate, not a full translation)."],
  "errors": []
}
```

**Example — one address field needed mirroring (200, warning only)**
```json
{
  "success": true,
  "data": { "presentAddress": "Dhaka, Bangladesh", "permanentAddress": "Dhaka, Bangladesh", "...": "..." },
  "warnings": ["presentAddress was empty; mirrored from permanentAddress (this card prints a single address for both fields)."],
  "errors": []
}
```

### Rate limiting & concurrency

- Per-IP limit: `RATE_LIMIT_PER_MINUTE` (default 8/min). Exceeding it returns
  `429` in the same response shape.
- Server-side concurrency cap: `CONCURRENCY_LIMIT` (default 4). Excess
  concurrent requests get an immediate `503` rather than queueing/piling up.
