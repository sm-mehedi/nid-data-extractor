import pytest

from app.services import image_checks
from tests.helpers import (
    blurry_card_image,
    clear_card_image,
    cut_off_card_image,
    dark_image,
    encode_jpg,
    encode_png,
    glare_image,
    non_card_image,
    overexposed_image,
    rotated,
    small_unzoomed_card_image,
    tiny_image,
    whatsapp_compressed_card_image,
)


def test_validate_extension_accepts_allowed():
    assert image_checks.validate_extension("front.jpg") == "jpg"
    assert image_checks.validate_extension("front.JPEG") == "jpeg"
    assert image_checks.validate_extension("front.png") == "png"


@pytest.mark.parametrize("filename", ["front.gif", "front.bmp", "front.webp", "front.pdf", "front.txt"])
def test_validate_extension_rejects_disallowed(filename):
    with pytest.raises(image_checks.ImageQualityError):
        image_checks.validate_extension(filename)


def test_validate_extension_no_extension():
    with pytest.raises(image_checks.ImageQualityError):
        image_checks.validate_extension("frontimage")


def test_validate_size_empty():
    with pytest.raises(image_checks.ImageQualityError, match="empty"):
        image_checks.validate_size(b"", max_bytes=1000)


def test_validate_size_over_limit():
    with pytest.raises(image_checks.ImageQualityError, match="exceeds"):
        image_checks.validate_size(b"x" * 2000, max_bytes=1000)


def test_validate_size_at_exact_limit_ok():
    image_checks.validate_size(b"x" * 1000, max_bytes=1000)


def test_decode_image_corrupt_bytes():
    with pytest.raises(image_checks.ImageQualityError, match="could not be decoded|corrupt"):
        image_checks.decode_image(b"this is not an image, just text renamed to .jpg")


def test_decode_image_txt_renamed_jpg():
    with pytest.raises(image_checks.ImageQualityError):
        image_checks.decode_image(b"plain text content" * 20)


def test_decode_image_tiny():
    tiny_bytes = encode_jpg(tiny_image())
    with pytest.raises(image_checks.ImageQualityError, match="too small"):
        image_checks.decode_image(tiny_bytes)


def test_decode_image_valid_png_and_jpg():
    img = clear_card_image()
    decoded_jpg = image_checks.decode_image(encode_jpg(img))
    decoded_png = image_checks.decode_image(encode_png(img))
    assert decoded_jpg is not None
    assert decoded_png is not None


def test_blur_check_passes_on_clear_image():
    import cv2

    img = clear_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_checks.check_blur(gray)  # should not raise


def test_blur_check_rejects_blurry_image():
    import cv2

    img = blurry_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with pytest.raises(image_checks.ImageQualityError, match="blurry"):
        image_checks.check_blur(gray)


def test_blur_check_rejects_blurry_image_with_wide_margin_below_threshold():
    # Guards the recalibration: a genuinely blurry photo must fail by a large
    # margin, not just barely — confirms lowering the threshold to fix real
    # WhatsApp-compression false positives didn't quietly let real blur through.
    import cv2

    img = blurry_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = image_checks.compute_blur_variance(gray)
    assert variance < image_checks.BLUR_VARIANCE_THRESHOLD / 5


def test_blur_check_passes_on_whatsapp_compressed_but_legible_photo():
    # Regression test for a real false positive: a real phone photo with
    # normal handheld focus softness, after WhatsApp-style downscale +
    # aggressive JPEG re-compression, must still pass — it's legible to a
    # human (and to Cloud Vision/Gemini), just not pixel-perfect sharp.
    import cv2

    img = whatsapp_compressed_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_checks.check_blur(gray)  # should not raise
    variance = image_checks.compute_blur_variance(gray)
    assert variance > image_checks.BLUR_VARIANCE_THRESHOLD


def test_exposure_check_rejects_dark_image():
    import cv2

    img = dark_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with pytest.raises(image_checks.ImageQualityError, match="dark"):
        image_checks.check_exposure(gray)


def test_exposure_check_rejects_overexposed_image():
    import cv2

    img = overexposed_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with pytest.raises(image_checks.ImageQualityError, match="overexposed"):
        image_checks.check_exposure(gray)


def test_exposure_check_passes_on_clear_image():
    import cv2

    img = clear_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_checks.check_exposure(gray)  # should not raise


def test_glare_check_detects_localized_bright_blob():
    import cv2

    img = glare_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with pytest.raises(image_checks.ImageQualityError, match="Glare"):
        image_checks.check_glare(gray)


def test_glare_check_passes_on_clear_image():
    import cv2

    img = clear_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_checks.check_glare(gray)  # should not raise


def test_card_boundary_found_and_cropped():
    import cv2

    img = clear_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = image_checks.find_card_quad(gray)
    assert result.found is True
    assert result.touches_frame_edge is False


def test_card_boundary_cut_off_at_edge():
    import cv2

    img = cut_off_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = image_checks.find_card_quad(gray)
    assert result.found is True
    assert result.touches_frame_edge is True


def test_card_boundary_not_found_for_small_unzoomed_card():
    import cv2

    img = small_unzoomed_card_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = image_checks.find_card_quad(gray)
    # A small, unzoomed card is a materially different situation from a
    # cropped/cut-off one: it must NOT be classified as touching the frame edge.
    assert result.touches_frame_edge is False


def test_run_quality_pipeline_full_success_autocrops():
    content = encode_jpg(clear_card_image())
    result = image_checks.run_quality_pipeline(content, "front.jpg", max_upload_bytes=10_000_000)
    assert result.auto_cropped is True
    assert result.warnings == []


def test_run_quality_pipeline_accepts_whatsapp_compressed_photo():
    content = encode_jpg(whatsapp_compressed_card_image())
    # Must not raise "photo appears blurry" for a realistically-compressed
    # but legible photo — this is the exact end-to-end path a real upload
    # goes through, not just the isolated check_blur() unit.
    result = image_checks.run_quality_pipeline(content, "front.jpg", max_upload_bytes=10_000_000)
    assert result is not None


def test_run_quality_pipeline_cut_off_raises_400():
    content = encode_jpg(cut_off_card_image())
    with pytest.raises(image_checks.ImageQualityError) as exc_info:
        image_checks.run_quality_pipeline(content, "front.jpg", max_upload_bytes=10_000_000)
    assert exc_info.value.status_code == 400
    assert "retake" in exc_info.value.message


def test_run_quality_pipeline_small_unzoomed_produces_warning_not_error():
    content = encode_jpg(small_unzoomed_card_image())
    result = image_checks.run_quality_pipeline(content, "front.jpg", max_upload_bytes=10_000_000)
    assert result.auto_cropped is False
    assert len(result.warnings) == 1


def test_run_quality_pipeline_non_card_image_falls_through_to_warning():
    content = encode_jpg(non_card_image())
    result = image_checks.run_quality_pipeline(content, "front.jpg", max_upload_bytes=10_000_000)
    assert result.auto_cropped is False
    assert len(result.warnings) == 1


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_run_quality_pipeline_handles_rotated_images(angle):
    content = encode_jpg(rotated(clear_card_image(), angle))
    # Should not raise — rotation alone is not a rejection condition.
    image_checks.run_quality_pipeline(content, "front.jpg", max_upload_bytes=10_000_000)


def test_run_quality_pipeline_wrong_extension():
    content = encode_jpg(clear_card_image())
    with pytest.raises(image_checks.ImageQualityError):
        image_checks.run_quality_pipeline(content, "front.gif", max_upload_bytes=10_000_000)


def test_run_quality_pipeline_empty_file():
    with pytest.raises(image_checks.ImageQualityError, match="empty"):
        image_checks.run_quality_pipeline(b"", "front.jpg", max_upload_bytes=10_000_000)


def test_run_quality_pipeline_oversized():
    content = encode_jpg(clear_card_image())
    with pytest.raises(image_checks.ImageQualityError, match="exceeds"):
        image_checks.run_quality_pipeline(content, "front.jpg", max_upload_bytes=100)
