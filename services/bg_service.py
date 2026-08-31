from __future__ import annotations

import gc
import io
import logging
import os
import time
import uuid
from typing import Final

import psutil
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError
from rembg import new_session, remove


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME: Final[str] = os.getenv(
    "BG_MODEL",
    "u2netp",
)

MAX_FILE_SIZE: Final[int] = int(
    os.getenv(
        "BG_MAX_FILE_SIZE",
        str(10 * 1024 * 1024),
    )
)

MAX_IMAGE_PIXELS: Final[int] = int(
    os.getenv(
        "BG_MAX_IMAGE_PIXELS",
        str(25_000_000),
    )
)

MAX_INFERENCE_DIMENSION: Final[int] = int(
    os.getenv(
        "BG_MAX_INFERENCE_DIMENSION",
        "2048",
    )
)

MAX_CONCURRENT_INFERENCE: Final[int] = int(
    os.getenv(
        "BG_MAX_CONCURRENT",
        "1",
    )
)

ALLOWED_CONTENT_TYPES: Final[set[str]] = {
    "image/png",
    "image/jpeg",
    "image/webp",
}


# ============================================================
# PROCESS
# ============================================================

_process = psutil.Process(os.getpid())


# ============================================================
# MODEL STATE
# ============================================================

_session = None

_session_lock = None

_inference_semaphore = None


# ============================================================
# LAZY ASYNC PRIMITIVES
# ============================================================

def _get_session_lock():
    global _session_lock

    if _session_lock is None:
        import asyncio

        _session_lock = asyncio.Lock()

    return _session_lock


def _get_inference_semaphore():
    global _inference_semaphore

    if _inference_semaphore is None:
        import asyncio

        _inference_semaphore = asyncio.Semaphore(
            MAX_CONCURRENT_INFERENCE
        )

    return _inference_semaphore


# ============================================================
# MEMORY
# ============================================================

def log_memory(
    label: str,
    request_id: str,
) -> None:

    try:

        memory = _process.memory_info()

        rss_mb = (
            memory.rss /
            1024 /
            1024
        )

        logger.info(
            "[%s] MEMORY | %s | RSS=%.2f MB",
            request_id,
            label,
            rss_mb,
        )

    except Exception:

        logger.debug(
            "[%s] Memory information unavailable.",
            request_id,
            exc_info=True,
        )


# ============================================================
# MODEL
# ============================================================

async def get_model_session():

    global _session

    if _session is not None:
        return _session

    lock = _get_session_lock()

    async with lock:

        if _session is not None:
            return _session

        logger.info(
            "Loading background-removal model: %s",
            MODEL_NAME,
        )

        started = time.perf_counter()

        try:

            _session = await run_in_threadpool(
                new_session,
                MODEL_NAME,
            )

        except Exception:

            logger.exception(
                "Failed to initialize model: %s",
                MODEL_NAME,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Background removal model "
                    "is temporarily unavailable."
                ),
            )

        elapsed = (
            time.perf_counter() -
            started
        )

        logger.info(
            "Model loaded successfully | model=%s | time=%.2fs",
            MODEL_NAME,
            elapsed,
        )

        log_memory(
            "AFTER MODEL LOAD",
            "SYSTEM",
        )

    return _session


# ============================================================
# IMAGE VALIDATION
# ============================================================

def inspect_image(
    image_bytes: bytes,
    request_id: str,
) -> tuple[int, int]:

    try:

        with Image.open(
            io.BytesIO(image_bytes)
        ) as image:

            image.verify()

        with Image.open(
            io.BytesIO(image_bytes)
        ) as image:

            width = image.width
            height = image.height

            logger.info(
                "[%s] IMAGE | width=%d | height=%d | format=%s | mode=%s",
                request_id,
                width,
                height,
                image.format,
                image.mode,
            )

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception:

        logger.exception(
            "[%s] Image validation failed.",
            request_id,
        )

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image.",
        )

    pixels = width * height

    if pixels > MAX_IMAGE_PIXELS:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image dimensions are too large. "
                "Please upload a smaller image."
            ),
        )

    return width, height


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(
    image_bytes: bytes,
    request_id: str,
) -> tuple[Image.Image, tuple[int, int]]:

    try:

        with Image.open(
            io.BytesIO(image_bytes)
        ) as original:

            image =
                ImageOps.exif_transpose(
                    original
                )

            original_size = (
                image.width,
                image.height,
            )

            # Work on a copy so the source image
            # remains untouched.
            image = image.copy()

            # RGB is required by the rembg model.
            if image.mode != "RGB":

                image = image.convert(
                    "RGB"
                )

            if (
                image.width >
                MAX_INFERENCE_DIMENSION
                or
                image.height >
                MAX_INFERENCE_DIMENSION
            ):

                image.thumbnail(
                    (
                        MAX_INFERENCE_DIMENSION,
                        MAX_INFERENCE_DIMENSION,
                    ),
                    Image.Resampling.LANCZOS,
                )

            logger.info(
                "[%s] PREPROCESS | original=%sx%s | inference=%sx%s",
                request_id,
                original_size[0],
                original_size[1],
                image.width,
                image.height,
            )

            return image, original_size

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            "[%s] Image preprocessing failed.",
            request_id,
        )

        raise HTTPException(
            status_code=400,
            detail="Unable to process image.",
        )


# ============================================================
# BACKGROUND REMOVAL
# ============================================================

async def remove_background(
    file: UploadFile,
) -> bytes:

    request_id = uuid.uuid4().hex[:10]

    started = time.perf_counter()

    logger.info(
        "[%s] START | filename=%s | content_type=%s",
        request_id,
        file.filename,
        file.content_type,
    )

    log_memory(
        "BEFORE REQUEST",
        request_id,
    )

    # --------------------------------------------------------
    # MIME validation
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PNG, JPEG and WEBP "
                "images are supported."
            ),
        )

    # --------------------------------------------------------
    # Read upload
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

    finally:

        await file.close()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    upload_size = len(image_bytes)

    logger.info(
        "[%s] UPLOAD | size=%.2f MB",
        request_id,
        upload_size / 1024 / 1024,
    )

    if upload_size > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                f"Maximum upload size is "
                f"{MAX_FILE_SIZE // 1024 // 1024} MB."
            ),
        )

    # --------------------------------------------------------
    # Validate actual image
    # --------------------------------------------------------

    inspect_image(
        image_bytes,
        request_id,
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image, original_size = preprocess_image(
        image_bytes,
        request_id,
    )

    del image_bytes

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    semaphore = (
        _get_inference_semaphore()
    )

    async with semaphore:

        logger.info(
            "[%s] INFERENCE SLOT ACQUIRED",
            request_id,
        )

        log_memory(
            "BEFORE INFERENCE",
            request_id,
        )

        try:

            model_session =
                await get_model_session()

            inference_started =
                time.perf_counter()

            logger.info(
                "[%s] INFERENCE START | model=%s",
                request_id,
                MODEL_NAME,
            )

            output = await run_in_threadpool(
                remove,
                image,
                session=model_session,
            )

            inference_time = (
                time.perf_counter()
                -
                inference_started
            )

            logger.info(
                "[%s] INFERENCE SUCCESS | time=%.2fs | output=%.2f MB",
                request_id,
                inference_time,
                len(output) / 1024 / 1024,
            )

            # ------------------------------------------------
            # Restore output to original dimensions
            # ------------------------------------------------

            final_output = await run_in_threadpool(
                resize_output,
                output,
                original_size,
                request_id,
            )

            total_time = (
                time.perf_counter()
                -
                started
            )

            logger.info(
                "[%s] COMPLETE | total=%.2fs | final=%.2f MB",
                request_id,
                total_time,
                len(final_output) / 1024 / 1024,
            )

            log_memory(
                "AFTER COMPLETE",
                request_id,
            )

            return final_output

        except HTTPException:

            raise

        except Exception:

            logger.exception(
                "[%s] INFERENCE FAILED",
                request_id,
            )

            log_memory(
                "INFERENCE FAILURE",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Background removal failed "
                    "during AI processing."
                ),
            )

        finally:

            try:
                image.close()
            except Exception:
                pass

            gc.collect()

            logger.info(
                "[%s] INFERENCE SLOT RELEASED",
                request_id,
            )


# ============================================================
# OUTPUT PROCESSING
# ============================================================

def resize_output(
    output: bytes,
    original_size: tuple[int, int],
    request_id: str,
) -> bytes:

    try:

        with Image.open(
            io.BytesIO(output)
        ) as result:

            result = result.convert(
                "RGBA"
            )

            if result.size != original_size:

                logger.info(
                    "[%s] OUTPUT RESIZE | %sx%s -> %sx%s",
                    request_id,
                    result.width,
                    result.height,
                    original_size[0],
                    original_size[1],
                )

                result = result.resize(
                    original_size,
                    Image.Resampling.LANCZOS,
                )

            buffer = io.BytesIO()

            result.save(
                buffer,
                format="PNG",
                optimize=False,
                compress_level=6,
            )

            return buffer.getvalue()

    except Exception:

        logger.exception(
            "[%s] Failed to finalize output.",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to create the "
                "transparent output image."
            ),
        )