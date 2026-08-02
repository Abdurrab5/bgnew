import asyncio
import gc
import io
import logging
import os
import time
from typing import Optional

from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError
from rembg import new_session, remove

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

def positive_int(value: Optional[str], default: int) -> int:
    try:
        value = int(value)
        return value if value > 0 else default
    except Exception:
        return default

MAX_CONCURRENT = positive_int(
    os.getenv("MAX_CONCURRENT"),
    1,
)

# Maximum upload size: 3 MB
MAX_FILE_SIZE = positive_int(
    os.getenv("MAX_FILE_SIZE"),
    3 * 1024 * 1024,
)

# Resize images before inference
MAX_DIMENSION = positive_int(
    os.getenv("MAX_DIMENSION"),
    512,
)

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

logger.info(
    "Configuration: file=%d dimension=%d concurrent=%d",
    MAX_FILE_SIZE,
    MAX_DIMENSION,
    MAX_CONCURRENT,
)

# ==========================================================
# Globals
# ==========================================================

session = None
session_lock = asyncio.Lock()
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# ==========================================================
# Session
# ==========================================================


async def get_session():
    global session

    if session is None:
        async with session_lock:

            if session is None:
                logger.info("Loading U2Net model...")

                session = await run_in_threadpool(
                    new_session,
                    "u2netp",
                )

                logger.info("Model loaded.")

    return session


async def recreate_session():
    global session

    async with session_lock:

        logger.warning("Recreating ONNX session...")

        session = await run_in_threadpool(
            new_session,
            "u2netp",
        )

        logger.info("New session ready.")

# ==========================================================
# Image preprocessing
# ==========================================================


def preprocess_image(image_bytes: bytes) -> bytes:

    try:
        img = Image.open(io.BytesIO(image_bytes))

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA") or (
        img.mode == "P"
        and "transparency" in img.info
    ):
        img = img.convert("RGBA")

    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail(
        (MAX_DIMENSION, MAX_DIMENSION),
        Image.Resampling.LANCZOS,
    )

    output = io.BytesIO()

    img.save(
        output,
        format="PNG",
        optimize=True,
    )

    return output.getvalue()

# ==========================================================
# Upload reader
# ==========================================================


async def read_upload(file: UploadFile) -> bytes:

    buffer = io.BytesIO()

    while True:

        chunk = await file.read(65536)

        if not chunk:
            break

        buffer.write(chunk)

        if buffer.tell() > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image is too large.",
            )

    return buffer.getvalue()

# ==========================================================
# Background removal
# ==========================================================


async def remove_bg(file: UploadFile) -> bytes:
    started = time.perf_counter()
    filename = file.filename or "unknown"

    logger.info(
        "Request started | filename=%s | content_type=%s",
        filename,
        file.content_type,
    )

    try:
        # --------------------------------------------------
        # Validate content type
        # --------------------------------------------------

        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported image type.",
            )

        # --------------------------------------------------
        # Read upload
        # --------------------------------------------------

        image = await read_upload(file)

        logger.info(
            "Upload size: %d bytes",
            len(image),
        )

        # --------------------------------------------------
        # Preprocess
        # --------------------------------------------------

        preprocess_started = time.perf_counter()

        normalized = preprocess_image(image)

        logger.info(
            "Preprocessing completed in %.2fs",
            time.perf_counter() - preprocess_started,
        )

        logger.info(
            "Processed image size: %d bytes",
            len(normalized),
        )

        # --------------------------------------------------
        # Run inference
        # --------------------------------------------------

        async with semaphore:

            for attempt in (1, 2):

                try:

                    if attempt == 1:
                        current_session = await get_session()
                    else:
                        logger.warning(
                            "Retrying with a fresh ONNX session..."
                        )
                        await recreate_session()
                        current_session = session

                    inference_started = time.perf_counter()

                    output = await run_in_threadpool(
                        remove,
                        normalized,
                        session=current_session,
                    )

                    logger.info(
                        "Inference completed in %.2fs (attempt %d)",
                        time.perf_counter() - inference_started,
                        attempt,
                    )

                    if not output:
                        raise RuntimeError(
                            "rembg returned empty output."
                        )

                    if not isinstance(output, bytes):
                        raise RuntimeError(
                            f"Expected bytes but got {type(output)}"
                        )

                    logger.info(
                        "Background removal succeeded."
                    )

                    return output

                except HTTPException:
                    raise

                except Exception:

                    logger.exception(
                        "Attempt %d failed.",
                        attempt,
                    )

                    if attempt == 2:
                        raise HTTPException(
                            status_code=500,
                            detail="Background removal failed.",
                        )

    finally:

        gc.collect()

        logger.info(
            "Request finished in %.2fs",
            time.perf_counter() - started,
        )
    async with semaphore:

        # --------------------------------------------------
        # First attempt
        # --------------------------------------------------

        try:

            current_session = await get_session()

            logger.info("Calling rembg.remove()")

            inference_start = time.perf_counter()

            output = await run_in_threadpool(
                remove,
                normalized,
                session=current_session,
            )

            logger.info(
                "Inference completed in %.2fs",
                time.perf_counter() - inference_start,
            )

            if not output:
                raise RuntimeError(
                    "rembg returned empty output"
                )

            if not isinstance(output, bytes):
                raise RuntimeError(
                    f"Expected bytes but got {type(output)}"
                )

            return output

        except Exception:

            logger.exception(
                "First attempt failed."
            )

        # --------------------------------------------------
        # Retry
        # --------------------------------------------------

        try:

            await recreate_session()

            logger.info(
                "Retrying with fresh session..."
            )

            inference_start = time.perf_counter()

            logger.info("Starting rembg inference")

            output = await run_in_threadpool(
            remove,
            normalized,
            session=current_session,
            )

            logger.info("Finished rembg inference")

            logger.info(
                "Retry completed in %.2fs",
                time.perf_counter() - inference_start,
            )

            if not output:
                raise RuntimeError(
                    "rembg returned empty output"
                )

            if not isinstance(output, bytes):
                raise RuntimeError(
                    f"Expected bytes but got {type(output)}"
                )

            return output

        except Exception:

            logger.exception(
                "Retry failed."
            )

            raise HTTPException(
                status_code=500,
                detail="Background removal failed.",
            )

        finally:

            gc.collect()

            logger.info(
                "Request finished in %.2fs",
                time.perf_counter() - started,
            )