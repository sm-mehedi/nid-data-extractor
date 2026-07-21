"""Cheap, local, pre-OCR image validation: format/size, blur, exposure, glare,
and card-boundary detection with auto-crop + deskew.

Every check here is meant to run before any paid API call, so a bad photo is
rejected (or repaired) for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ID-1 card aspect ratio (85.6mm x 53.98mm)
CARD_ASPECT_RATIO = 1.586
CARD_ASPECT_TOLERANCE = 0.35

BLUR_VARIANCE_THRESHOLD = 60.0
DARK_MEAN_THRESHOLD = 45.0
OVEREXPOSED_MEAN_THRESHOLD = 225.0
OVEREXPOSED_WHITE_RATIO_THRESHOLD = 0.6

GLARE_BRIGHTNESS_THRESHOLD = 240
GLARE_MIN_AREA_RATIO = 0.01
GLARE_MAX_AREA_RATIO = 0.35

MIN_DIMENSION_PX = 200

FRAME_EDGE_MARGIN_PX = 3


class ImageQualityError(Exception):
    """Raised when an uploaded image fails a cheap local check.

    `status_code` distinguishes a malformed/rejected upload (400) from
    "no card-like content found at all" (422), per the build plan's error table.
    """

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class QualityCheckResult:
    image: np.ndarray
    warnings: list[str] = field(default_factory=list)
    auto_cropped: bool = False


def validate_extension(filename: str) -> str:
    if "." not in filename:
        raise ImageQualityError(
            f"File '{filename}' has no extension; expected one of {sorted(ALLOWED_EXTENSIONS)}."
        )
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageQualityError(
            f"Unsupported file extension '.{ext}'; expected one of {sorted(ALLOWED_EXTENSIONS)}."
        )
    return ext


def validate_size(content: bytes, max_bytes: int) -> None:
    if len(content) == 0:
        raise ImageQualityError("Uploaded file is empty (0 bytes).")
    if len(content) > max_bytes:
        raise ImageQualityError(
            f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)}MB."
        )


def decode_image(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageQualityError(
            "File could not be decoded as an image (corrupt or not a real image file)."
        )
    h, w = image.shape[:2]
    if h < MIN_DIMENSION_PX or w < MIN_DIMENSION_PX:
        raise ImageQualityError(
            f"Image is too small ({w}x{h}px) to reliably read; please retake at a higher resolution."
        )
    return image


def compute_blur_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_blur(gray: np.ndarray) -> None:
    variance = compute_blur_variance(gray)
    if variance < BLUR_VARIANCE_THRESHOLD:
        raise ImageQualityError("Photo appears blurry, please retake.")


def compute_brightness_stats(gray: np.ndarray) -> tuple[float, float]:
    mean = float(gray.mean())
    white_ratio = float(np.mean(gray > 240))
    return mean, white_ratio


def check_exposure(gray: np.ndarray) -> None:
    mean, white_ratio = compute_brightness_stats(gray)
    if mean < DARK_MEAN_THRESHOLD:
        raise ImageQualityError("Photo is too dark, please retake in better lighting.")
    if mean > OVEREXPOSED_MEAN_THRESHOLD or white_ratio > OVEREXPOSED_WHITE_RATIO_THRESHOLD:
        raise ImageQualityError("Photo is overexposed/washed out, please retake.")


def detect_glare(gray: np.ndarray) -> bool:
    """Localized bright blob (glare), as opposed to uniform overexposure."""
    _, thresh = cv2.threshold(gray, GLARE_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    total_area = gray.shape[0] * gray.shape[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area_ratio = cv2.contourArea(c) / total_area
        if GLARE_MIN_AREA_RATIO < area_ratio < GLARE_MAX_AREA_RATIO:
            return True
    return False


def check_glare(gray: np.ndarray) -> None:
    if detect_glare(gray):
        raise ImageQualityError("Glare detected over part of the card, please retake.")


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


@dataclass
class CardBoundaryResult:
    found: bool
    touches_frame_edge: bool = False
    quad: np.ndarray | None = None


def find_card_quad(gray: np.ndarray) -> CardBoundaryResult:
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = h * w
    best_quad = None
    best_area = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * frame_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2).astype("float32")
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect
        width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        if height == 0:
            continue
        ratio = width / height
        if ratio < 1:
            ratio = 1 / ratio
        if abs(ratio - CARD_ASPECT_RATIO) > CARD_ASPECT_TOLERANCE:
            continue
        if area > best_area:
            best_area = area
            best_quad = rect

    if best_quad is not None:
        touches_edge = bool(
            np.any(best_quad[:, 0] <= FRAME_EDGE_MARGIN_PX)
            or np.any(best_quad[:, 0] >= w - FRAME_EDGE_MARGIN_PX)
            or np.any(best_quad[:, 1] <= FRAME_EDGE_MARGIN_PX)
            or np.any(best_quad[:, 1] >= h - FRAME_EDGE_MARGIN_PX)
        )
        return CardBoundaryResult(found=True, touches_frame_edge=touches_edge, quad=best_quad)

    # No fully-closed quad: a card whose physical edge falls outside the photo
    # has no edge pixels to trace there at all, so Canny-based contour search
    # can never close a polygon for it. Fall back to a coarser brightness-blob
    # check: a large, roughly card-shaped region whose bounding box runs into
    # the frame border is exactly that "cut off" situation.
    if _cutoff_blob_touches_border(gray):
        return CardBoundaryResult(found=True, touches_frame_edge=True)

    return CardBoundaryResult(found=False)


def _cutoff_blob_touches_border(gray: np.ndarray) -> bool:
    h, w = gray.shape[:2]
    frame_area = h * w
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for candidate in (thresh, cv2.bitwise_not(thresh)):
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.15 * frame_area or area > 0.85 * frame_area:
                # Too small to be the card, or too large — almost certainly the
                # background itself (which trivially touches every side).
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            touches_left = x <= FRAME_EDGE_MARGIN_PX
            touches_top = y <= FRAME_EDGE_MARGIN_PX
            touches_right = x + cw >= w - FRAME_EDGE_MARGIN_PX
            touches_bottom = y + ch >= h - FRAME_EDGE_MARGIN_PX
            num_sides_touched = sum([touches_left, touches_top, touches_right, touches_bottom])
            # A card cut off by the frame runs off through 1-2 sides, not 3-4 —
            # 3+ sides touched is a signature of picking up the background blob.
            if not (1 <= num_sides_touched <= 2):
                continue
            ratio = cw / ch if cw > ch else ch / cw
            if abs(ratio - CARD_ASPECT_RATIO) <= CARD_ASPECT_TOLERANCE:
                return True
    return False


def run_quality_pipeline(content: bytes, filename: str, max_upload_bytes: int) -> QualityCheckResult:
    """Runs the full cheap-local-checks pipeline (Section 2, step 2 of the build plan).

    Raises ImageQualityError on any hard rejection. Returns the (possibly
    auto-cropped/deskewed) image plus any soft warnings otherwise.
    """
    validate_extension(filename)
    validate_size(content, max_upload_bytes)
    image = decode_image(content)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    check_blur(gray)
    check_exposure(gray)
    check_glare(gray)

    boundary = find_card_quad(gray)
    warnings: list[str] = []
    result_image = image
    auto_cropped = False

    if boundary.found and boundary.touches_frame_edge:
        raise ImageQualityError(
            "Card appears cut off at the edge of the photo; please retake showing the full card."
        )
    elif boundary.found and boundary.quad is not None:
        result_image = four_point_transform(image, boundary.quad)
        auto_cropped = True
    else:
        warnings.append(
            "Could not confidently detect the card boundary; falling back to AI visual judgment."
        )

    return QualityCheckResult(image=result_image, warnings=warnings, auto_cropped=auto_cropped)
