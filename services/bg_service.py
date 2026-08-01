import os
import io
import gc
import time
import asyncio
import logging

from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from rembg import remove, new_session

logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_DIMENSION = 2048

# -----------------------------
# Global objects
# -----------------------------

session = None
session_lock = asyncio.Lock()
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# -----------------------------
# Session Management
# -----------------------------


async def get_session():
    global session

    if session is None:
        async with session_lock:
            if session is None:
                logger.info("Loading U2Net model...")
                session = await run_in_threadpool(new_session, "u2netp")
                logger.info("Model loaded.")

    return session


async def recreate_session():
    global session

    async with session_lock:
        logger.warning("Recreating ONNX session...")

        session = await run_in_threadpool(new_session, "u2netp")

        logger.info("New ONNX session created.")

# -----------------------------
# Image Preprocessing
# -----------------------------


def preprocess_image(image_bytes: bytes) -> bytes:

    try:
        img = Image.open(io.BytesIO(image_bytes))

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail(
        (MAX_DIMENSION, MAX_DIMENSION),
        Image.Resampling.LANCZOS
    )

    output = io.BytesIO()

    img.save(
        output,
        format="PNG",
        optimize=True
    )

    return output.getvalue()

# -----------------------------
# Background Removal
# -----------------------------


async def remove_bg(file: UploadFile):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPEG and WEBP images are supported."
        )

    image = await file.read()

    if not image:
        raise HTTPException(
            status_code=400,
            detail="Empty image."
        )

    if len(image) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 5 MB."
        )

    try:
        normalized = preprocess_image(image)

    except HTTPException:
        raise

    except Exception:
        logger.exception("Image preprocessing failed")

        raise HTTPException(
            status_code=400,
            detail="Unable to process image."
        )

    start = time.perf_counter()

    async with semaphore:

        # -----------------------------
        # First attempt
        # -----------------------------

        try:

            current_session = await get_session()

            output = await run_in_threadpool(
                remove,
                normalized,
                session=current_session
            )

            logger.info(
                "Completed in %.2fs",
                time.perf_counter() - start
            )

            gc.collect()

            return output

        except Exception:

            logger.exception(
                "Background removal failed. Retrying with a new session..."
            )

        # -----------------------------
        # Recreate session and retry
        # -----------------------------

        try:

            await recreate_session()

            output = await run_in_threadpool(
                remove,
                normalized,
                session=session
            )

            logger.info(
                "Retry succeeded in %.2fs",
                time.perf_counter() - start
            )

            gc.collect()

            return output

        except Exception:

            logger.exception(
                "Retry also failed."
            )

            raise HTTPException(
                status_code=500,
                detail="Background removal failed."
            )