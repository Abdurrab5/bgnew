import os
import io
import gc
import time
import uuid
import logging
import asyncio

import psutil

from PIL import Image, ImageOps, UnidentifiedImageError

from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from rembg import remove, new_session


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

MAX_CONCURRENT = int(
    os.getenv("MAX_CONCURRENT", "1")
)

MAX_FILE_SIZE = 5 * 1024 * 1024

MAX_DIMENSION = 2048

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}


# =========================================================
# PROCESS
# =========================================================

process = psutil.Process(
    os.getpid()
)


# =========================================================
# MODEL
# =========================================================

session = None

session_lock = asyncio.Lock()

inference_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT
)


# =========================================================
# MEMORY LOGGING
# =========================================================

def log_memory(label: str, request_id: str = "-"):

    try:

        memory = process.memory_info()

        rss_mb = memory.rss / 1024 / 1024
        vms_mb = memory.vms / 1024 / 1024

        logger.info(
            "[%s] MEMORY | %s | RSS=%.2f MB | VMS=%.2f MB",
            request_id,
            label,
            rss_mb,
            vms_mb,
        )

    except Exception:

        logger.exception(
            "[%s] Failed to read process memory.",
            request_id,
        )


# =========================================================
# MODEL SESSION
# =========================================================

async def get_session():

    global session

    if session is not None:
        return session

    async with session_lock:

        if session is not None:
            return session

        logger.info(
            "Loading U2NetP model..."
        )

        start = time.perf_counter()

        session = await run_in_threadpool(
            new_session,
            "u2netp"
        )

        elapsed = time.perf_counter() - start

        logger.info(
            "U2NetP model loaded in %.2fs",
            elapsed,
        )

        log_memory(
            "AFTER MODEL LOAD"
        )

    return session


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(
    image_bytes: bytes,
    request_id: str,
) -> bytes:

    try:

        with Image.open(
            io.BytesIO(image_bytes)
        ) as original:

            logger.info(
                "[%s] IMAGE | original=%sx%s | mode=%s | format=%s",
                request_id,
                original.width,
                original.height,
                original.mode,
                original.format,
            )

            # Correct EXIF rotation
            img = ImageOps.exif_transpose(
                original
            )

            # Convert to RGB
            if img.mode != "RGB":

                img = img.convert("RGB")

            # Limit dimensions
            if (
                img.width > MAX_DIMENSION
                or img.height > MAX_DIMENSION
            ):

                img.thumbnail(
                    (
                        MAX_DIMENSION,
                        MAX_DIMENSION,
                    ),
                    Image.Resampling.LANCZOS,
                )

            logger.info(
                "[%s] IMAGE | processed=%sx%s",
                request_id,
                img.width,
                img.height,
            )

            output = io.BytesIO()

            img.save(
                output,
                format="PNG",
                optimize=False,
            )

            return output.getvalue()

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    except Exception:

        logger.exception(
            "[%s] Image preprocessing failed.",
            request_id,
        )

        raise HTTPException(
            status_code=400,
            detail="Unable to process image.",
        )


# =========================================================
# BACKGROUND REMOVAL
# =========================================================

async def remove_bg(
    file: UploadFile,
):

    request_id = uuid.uuid4().hex[:8]

    start = time.perf_counter()

    logger.info(
        "[%s] START | filename=%s | type=%s",
        request_id,
        file.filename,
        file.content_type,
    )

    log_memory(
        "BEFORE REQUEST",
        request_id,
    )

    # -----------------------------------------------------
    # Validate MIME type
    # -----------------------------------------------------

    if file.content_type not in ALLOWED_TYPES:

        logger.warning(
            "[%s] Rejected file type: %s",
            request_id,
            file.content_type,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PNG, JPEG and WEBP "
                "images are supported."
            ),
        )

    # -----------------------------------------------------
    # Read file
    # -----------------------------------------------------

    try:

        image = await file.read()

    finally:

        await file.close()

    # -----------------------------------------------------
    # Validate empty
    # -----------------------------------------------------

    if not image:

        logger.warning(
            "[%s] Empty image.",
            request_id,
        )

        raise HTTPException(
            status_code=400,
            detail="Empty image.",
        )

    # -----------------------------------------------------
    # Validate size
    # -----------------------------------------------------

    image_size = len(image)

    logger.info(
        "[%s] UPLOAD | size=%.2f MB",
        request_id,
        image_size / 1024 / 1024,
    )

    if image_size > MAX_FILE_SIZE:

        logger.warning(
            "[%s] File too large.",
            request_id,
        )

        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 5 MB.",
        )

    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    normalized = preprocess_image(
        image,
        request_id,
    )

    # Release original image bytes
    del image

    gc.collect()

    log_memory(
        "AFTER PREPROCESS",
        request_id,
    )

    # -----------------------------------------------------
    # Get model
    # -----------------------------------------------------

    async with inference_semaphore:

        logger.info(
            "[%s] INFERENCE SLOT ACQUIRED",
            request_id,
        )

        log_memory(
            "BEFORE INFERENCE",
            request_id,
        )

        try:

            current_session = await get_session()

            logger.info(
                "[%s] INFERENCE START",
                request_id,
            )

            inference_start = time.perf_counter()

            output = await run_in_threadpool(
                remove,
                normalized,
                session=current_session,
            )

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            logger.info(
                "[%s] INFERENCE SUCCESS | time=%.2fs | output=%.2f MB",
                request_id,
                inference_time,
                len(output) / 1024 / 1024,
            )

            log_memory(
                "AFTER INFERENCE",
                request_id,
            )

            # Release normalized input
            del normalized

            gc.collect()

            log_memory(
                "AFTER CLEANUP",
                request_id,
            )

            total_time = (
                time.perf_counter()
                - start
            )

            logger.info(
                "[%s] COMPLETE | total=%.2fs",
                request_id,
                total_time,
            )

            return output

        except HTTPException:

            raise

        except Exception as exc:

            logger.exception(
                "[%s] INFERENCE FAILED | %s",
                request_id,
                str(exc),
            )

            log_memory(
                "AFTER INFERENCE FAILURE",
                request_id,
            )

            # IMPORTANT:
            #
            # Do NOT automatically create another
            # ONNX session here.
            #
            # On a 1 GB cPanel server that can
            # increase memory pressure.

            raise HTTPException(
                status_code=500,
                detail="Background removal failed.",
            )

        finally:

            logger.info(
                "[%s] INFERENCE SLOT RELEASED",
                request_id,
            )