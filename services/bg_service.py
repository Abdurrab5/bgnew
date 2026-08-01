import os
import io
import gc
import time
import asyncio
import logging
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from rembg import remove, new_session

logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------

def _parse_positive_int(value: Optional[str], default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
        return parsed if parsed > 0 else default
    except ValueError:
        return default

MAX_CONCURRENT = _parse_positive_int(os.getenv("MAX_CONCURRENT"), 2)

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

MAX_FILE_SIZE = _parse_positive_int(os.getenv("MAX_FILE_SIZE"), 5 * 1024 * 1024)
MAX_DIMENSION = _parse_positive_int(os.getenv("MAX_DIMENSION"), 2048)

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

    if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
    elif img.mode != "RGB":
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


async def read_upload_bytes(file: UploadFile, max_size: int) -> bytes:
    output = io.BytesIO()
    chunk_size = 64 * 1024

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        output.write(chunk)
        if output.tell() > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Maximum upload size is {max_size / (1024 * 1024):.1f} MB."
            )

    return output.getvalue()


async def remove_bg(file: UploadFile):

    content_type = file.content_type.lower() if file.content_type else None

    if content_type not in ALLOWED_TYPES:
        if content_type and content_type != "application/octet-stream":
            raise HTTPException(
                status_code=400,
                detail="Only PNG, JPEG and WEBP images are supported."
            )

    image = await read_upload_bytes(file, MAX_FILE_SIZE)

    if not image:
        raise HTTPException(
            status_code=400,
            detail="Empty image."
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