"""Synthetic image generators for tests — no real NID photos are used or needed."""
from __future__ import annotations

import cv2
import numpy as np

CARD_W, CARD_H = 700, 441  # ~1.586 aspect ratio
CANVAS_W, CANVAS_H = 900, 700


def _draw_textured_card(canvas: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Draws a card-shaped rectangle with enough internal edges (text-like bars,
    border) to produce a high Laplacian variance, so it reads as 'sharp'."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (235, 235, 235), thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 40, 40), thickness=3)
    rng = np.random.default_rng(42)
    for i in range(8):
        line_y = y + 40 + i * (h - 80) // 8
        line_w = int(w * rng.uniform(0.3, 0.75))
        cv2.rectangle(
            canvas,
            (x + 30, line_y),
            (x + 30 + line_w, line_y + 12),
            (20, 20, 20),
            thickness=-1,
        )


def encode_jpg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def clear_card_image() -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 200, dtype=np.uint8)
    x = (CANVAS_W - CARD_W) // 2
    y = (CANVAS_H - CARD_H) // 2
    _draw_textured_card(canvas, x, y, CARD_W, CARD_H)
    return canvas


def blurry_card_image() -> np.ndarray:
    image = clear_card_image()
    return cv2.GaussianBlur(image, (35, 35), 15)


def whatsapp_compressed_card_image() -> np.ndarray:
    """Simulates a realistic legible-but-degraded phone photo: mild
    handheld-focus softness (a 5x5 Gaussian blur — subtle, not the heavy
    blur used in blurry_card_image) followed by WhatsApp-style downscale +
    aggressive JPEG re-encode. Measures ~47-52 Laplacian variance, well
    above BLUR_VARIANCE_THRESHOLD (35) but well below a pristine synthetic
    image (~1489) — representative of the false positives reported against
    real WhatsApp-sent NID photos."""
    image = clear_card_image()
    softened = cv2.GaussianBlur(image, (5, 5), 0)
    h, w = softened.shape[:2]
    small = cv2.resize(softened, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", upscaled, [cv2.IMWRITE_JPEG_QUALITY, 35])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def dark_image() -> np.ndarray:
    image = clear_card_image()
    return (image.astype(np.float32) * 0.08).astype(np.uint8)


def overexposed_image() -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 250, dtype=np.uint8)
    x = (CANVAS_W - CARD_W) // 2
    y = (CANVAS_H - CARD_H) // 2
    cv2.rectangle(canvas, (x, y), (x + CARD_W, y + CARD_H), (252, 252, 252), thickness=-1)
    return canvas


def glare_image() -> np.ndarray:
    image = clear_card_image()
    cv2.circle(image, (CANVAS_W // 2, CANVAS_H // 2), 60, (255, 255, 255), thickness=-1)
    return image


def cut_off_card_image() -> np.ndarray:
    """Card quad whose edge touches the frame boundary — unrecoverable crop."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 200, dtype=np.uint8)
    x = -100
    y = (CANVAS_H - CARD_H) // 2
    _draw_textured_card(canvas, x, y, CARD_W, CARD_H)
    return canvas


def small_unzoomed_card_image() -> np.ndarray:
    """Card present but small relative to frame — should NOT be treated as cut off,
    just falls through to AI judgment (no confident local quad detected)."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 210, dtype=np.uint8)
    small_w, small_h = 220, 139
    x = (CANVAS_W - small_w) // 2
    y = (CANVAS_H - small_h) // 2
    _draw_textured_card(canvas, x, y, small_w, small_h)
    return canvas


def tiny_image() -> np.ndarray:
    return np.full((10, 10, 3), 128, dtype=np.uint8)


def non_card_image() -> np.ndarray:
    """A plausible photo that is not a card at all (e.g. a random object) — still
    a valid, sharp, well-exposed image so it passes local checks; only Gemini's
    holistic judgment can reject it."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 180, dtype=np.uint8)
    cv2.circle(canvas, (CANVAS_W // 2, CANVAS_H // 2), 150, (90, 60, 30), thickness=-1)
    rng = np.random.default_rng(7)
    for _ in range(20):
        p1 = (int(rng.uniform(0, CANVAS_W)), int(rng.uniform(0, CANVAS_H)))
        p2 = (int(rng.uniform(0, CANVAS_W)), int(rng.uniform(0, CANVAS_H)))
        cv2.line(canvas, p1, p2, (20, 20, 20), thickness=2)
    return canvas


def _mrz_check_digit(data: str) -> int:
    from app.services.mrz import compute_check_digit

    return compute_check_digit(data)


def build_td1_mrz(
    doc_number: str = "123456789",
    optional_data_1: str = "1234567890123<<",
    dob: str = "980115",
    sex: str = "M",
    expiry: str = "300101",
    optional_data_2: str = "<<<<<<<<<<<",
    surname: str = "RAHIM",
    given_names: str = "MD",
    *,
    corrupt_doc_check: bool = False,
    corrupt_dob_check: bool = False,
    corrupt_composite: bool = False,
) -> str:
    """Builds a valid (or deliberately corrupted) synthetic ICAO 9303 TD1 MRZ
    block, 3 lines of 30 chars each, for testing checksum validation."""
    doc_number = doc_number.ljust(9, "<")[:9]
    optional_data_1 = optional_data_1.ljust(15, "<")[:15]
    doc_check = _mrz_check_digit(doc_number)
    if corrupt_doc_check:
        doc_check = (doc_check + 1) % 10
    line1 = f"ID{'BGD'}{doc_number}{doc_check}{optional_data_1}"

    dob_check = _mrz_check_digit(dob)
    if corrupt_dob_check:
        dob_check = (dob_check + 1) % 10
    expiry_check = _mrz_check_digit(expiry)
    nationality = "BGD"
    optional_data_2 = optional_data_2.ljust(11, "<")[:11]

    composite_data = (
        doc_number + str(_mrz_check_digit(doc_number)) + optional_data_1
        + dob + str(dob_check)
        + expiry + str(expiry_check)
        + optional_data_2
    )
    composite_check = _mrz_check_digit(composite_data)
    if corrupt_composite:
        composite_check = (composite_check + 1) % 10

    line2 = f"{dob}{dob_check}{sex}{expiry}{expiry_check}{nationality}{optional_data_2}{composite_check}"

    name_field = f"{surname}<<{given_names}".ljust(30, "<")[:30]
    line3 = name_field

    assert len(line1) == 30, len(line1)
    assert len(line2) == 30, len(line2)
    assert len(line3) == 30, len(line3)
    return f"{line1}\n{line2}\n{line3}\n"


def rotated(image: np.ndarray, angle_degrees: int) -> np.ndarray:
    if angle_degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle_degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle_degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("angle must be 90, 180, or 270")
