import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import ExtractResponse
from app.security import concurrency_guard, limiter, verify_shared_secret
from app.services import gemini, image_checks, pipeline, vision_ocr

logger = logging.getLogger("nid_extractor")

router = APIRouter()


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Reads the upload in chunks, aborting as soon as the size cap is exceeded
    instead of buffering the entire (potentially huge) body first."""
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise image_checks.ImageQualityError(
                f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)}MB."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _error_response(status_code: int, message: str) -> JSONResponse:
    body = ExtractResponse(success=False, data=None, warnings=[], errors=[message])
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _rate_limit_value() -> str:
    return f"{get_settings().rate_limit_per_minute}/minute"


@router.post("/api/v1/nid/extract", response_model=ExtractResponse)
@limiter.limit(_rate_limit_value)
async def extract_nid_endpoint(
    request: Request,
    front_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None),
    _: None = Depends(verify_shared_secret),
):
    settings = get_settings()

    if front_image is None and back_image is None:
        return _error_response(400, "Both front_image and back_image are required.")
    if front_image is None:
        return _error_response(400, "front_image is required.")
    if back_image is None:
        return _error_response(400, "back_image is required.")

    try:
        front_content = await _read_limited(front_image, settings.max_upload_bytes)
        back_content = await _read_limited(back_image, settings.max_upload_bytes)
    except image_checks.ImageQualityError as exc:
        return _error_response(exc.status_code, exc.message)

    try:
        async with concurrency_guard():
            result = await run_in_threadpool(
                pipeline.extract_nid,
                front_content,
                front_image.filename or "front",
                back_content,
                back_image.filename or "back",
                settings.max_upload_bytes,
            )
        return JSONResponse(status_code=200, content=result.model_dump())

    except image_checks.ImageQualityError as exc:
        return _error_response(exc.status_code, exc.message)
    except pipeline.NotNidCardError as exc:
        return _error_response(422, str(exc))
    except (vision_ocr.VisionOCRError, gemini.GeminiError) as exc:
        logger.warning("Upstream AI service failure: %s", type(exc).__name__)
        return _error_response(503, f"Upstream AI service error: {exc}")
    except HTTPException as exc:
        return _error_response(exc.status_code, str(exc.detail))
    except Exception:
        logger.exception("Unexpected error during NID extraction")
        return _error_response(500, "Internal server error.")
