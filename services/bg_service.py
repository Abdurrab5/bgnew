from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import uuid
import warnings

from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError
from rembg import new_session, remove

from config import settings

logger = logging.getLogger(__name__)
_session = None
_session_lock = threading.Lock()
_inference_slots = asyncio.Semaphore(settings.max_concurrent_jobs)

_MIME_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
}


def model_ready() -> bool:
    return _session is not None


def _get_model_session():
    """Initialize once per process; keep model loading off the event loop."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                started = time.perf_counter()
                logger.info("Initializing rembg model=%s", settings.model_name)
                _session = new_session(
                    settings.model_name,
                    providers=["CPUExecutionProvider"],
                )
                logger.info(
                    "Model initialized model=%s duration_seconds=%.2f",
                    settings.model_name,
                    time.perf_counter() - started,
                )
    return _session


async def _read_limited(file: UploadFile) -> bytes:
    # Read at most one byte beyond the limit, avoiding an unbounded in-memory read.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Maximum upload size is {settings.max_upload_bytes // (1024 * 1024)} MB.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    return data


def _decode_and_prepare(data: bytes, declared_type: str | None) -> tuple[Image.Image, tuple[int, int]]:
    if declared_type not in _MIME_FORMATS:
        raise HTTPException(status_code=415, detail="Only PNG, JPEG and WebP images are supported.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                image_format = probe.format
                width, height = probe.size
                if width < 1 or height < 1 or width * height > settings.max_image_pixels:
                    raise HTTPException(status_code=413, detail="Image dimensions exceed the allowed limit.")
                probe.verify()

            if image_format != _MIME_FORMATS[declared_type]:
                raise HTTPException(status_code=415, detail="Image content does not match its declared format.")

            with Image.open(io.BytesIO(data)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                original_size = oriented.size
                prepared = oriented.convert("RGB")
                prepared.thumbnail(
                    (settings.max_inference_dimension, settings.max_inference_dimension),
                    Image.Resampling.LANCZOS,
                )
                return prepared, original_size
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc
    except Exception as exc:
        logger.warning("Image decoding failed: %s", type(exc).__name__)
        raise HTTPException(status_code=400, detail="Unable to read the uploaded image.") from exc


def _infer_and_encode(image: Image.Image, original_size: tuple[int, int], session) -> bytes:
    result = remove(image, session=session)
    if isinstance(result, bytes):
        result_image = Image.open(io.BytesIO(result))
    elif isinstance(result, Image.Image):
        result_image = result
    else:
        raise TypeError("rembg returned an unsupported image result")

    rgba = None
    try:
        rgba = result_image.convert("RGBA")
        if rgba.size != original_size:
            resized = rgba.resize(original_size, Image.Resampling.LANCZOS)
            rgba.close()
            rgba = resized
        output = io.BytesIO()
        rgba.save(output, format="PNG", optimize=False, compress_level=6)
        return output.getvalue()
    finally:
        if rgba is not None:
            rgba.close()
        result_image.close()


async def remove_background(file: UploadFile) -> bytes:
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:10]
    image = None
    acquired = False
    try:
        data = await _read_limited(file)
        image, original_size = await run_in_threadpool(_decode_and_prepare, data, file.content_type)
        del data

        if _inference_slots.locked():
            logger.info("Request rejected as capacity is full request_id=%s", request_id)
            raise HTTPException(status_code=503, detail="Background removal is busy. Please retry shortly.")
        await _inference_slots.acquire()
        acquired = True

        try:
            session = await run_in_threadpool(_get_model_session)
        except Exception as exc:
            logger.exception("Model initialization failed")
            raise HTTPException(status_code=503, detail="Background removal is temporarily unavailable.") from exc

        inference_started = time.perf_counter()
        result = await run_in_threadpool(_infer_and_encode, image, original_size, session)
        logger.info(
            "Image processed model=%s inference_seconds=%.2f total_seconds=%.2f output_bytes=%d",
            settings.model_name,
            time.perf_counter() - inference_started,
            time.perf_counter() - started,
            len(result),
        )
        return result
    except HTTPException as exc:
        logger.info("Request rejected request_id=%s status=%d", request_id, exc.status_code)
        raise
    except Exception as exc:
        logger.exception("Background removal failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Background removal failed during image processing.") from exc
    finally:
        await file.close()
        if image is not None:
            image.close()
        if acquired:
            _inference_slots.release()
        logger.info(
            "Request finished request_id=%s total_seconds=%.2f",
            request_id,
            time.perf_counter() - started,
        )
