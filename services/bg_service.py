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

# Conservative defaults for a free cloud tier
MAX_CONCURRENT = _parse_positive_int(os.getenv("MAX_CONCURRENT"), 1)

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

MAX_FILE_SIZE = _parse_positive_int(os.getenv("MAX_FILE_SIZE"), 5 * 1024 * 1024)
# Reduce default processing dimension to lower memory and CPU on free tiers
MAX_DIMENSION = _parse_positive_int(os.getenv("MAX_DIMENSION"), 1024)

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

    # Keep alpha if present; otherwise use RGB
    if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to a conservative maximum to reduce memory & CPU usage on free tiers
    img.thumbnail(
        (MAX_DIMENSION, MAX_DIMENSION),
        Image.Resampling.LANCZOS
    )

    output = io.BytesIO()

    # Try to reduce output size while preserving quality. Use PNG for alpha, otherwise save optimized JPEG
    if img.mode == "RGBA":
        img.save(output, format="PNG", optimize=True)
    else:
        # JPEG saves are much smaller; use quality 85 for good balance
        img.save(output, format="JPEG", quality=85, optimize=True)

    return output.getvalue()

# -----------------------------
# Background Removal
# -----------------------------


async def read_upload_bytes(file: UploadFile, max_size: int) -> bytes:
    output = io.BytesIO()
    chunk_size = 64 * 1024

    logger.info("Reading uploaded file in chunks (chunk_size=%d)", chunk_size)

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        output.write(chunk)
        if output.tell() > max_size:
            logger.warning("Upload exceeded max size: %d bytes", output.tell())
            raise HTTPException(
                status_code=413,
                detail=f"Maximum upload size is {max_size / (1024 * 1024):.1f} MB."
            )

    return output.getvalue()


async def remove_bg(file: UploadFile):
    """Read an UploadFile, preprocess it and remove the background.

    Returns PNG bytes on success. Raises HTTPException with appropriate status
    codes and logs full stack traces for debugging.
    """

    filename = getattr(file, "filename", "<unknown>")
    content_type = file.content_type.lower() if file.content_type else None

    logger.info("POST /bg-remove received: filename=%s, content_type=%s", filename, content_type)

    if content_type not in ALLOWED_TYPES:
        if content_type and content_type != "application/octet-stream":
            logger.warning("Unsupported content type: %s", content_type)
            raise HTTPException(
                status_code=400,
                detail="Only PNG, JPEG and WEBP images are supported."
            )

    # Read upload in chunks (protect memory)
    logger.info("Reading uploaded file...")
    image = await read_upload_bytes(file, MAX_FILE_SIZE)
    logger.info("Upload size: %d bytes", len(image))

    if not image:
        raise HTTPException(status_code=400, detail="Empty image.")

    # Preprocess (resize/convert) with robust error handling
    try:
        logger.info("Preprocessing image %s...", filename)
        normalized = preprocess_image(image)
        logger.info("Preprocessed size: %d bytes", len(normalized))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Image preprocessing failed for %s", filename)
        raise HTTPException(status_code=400, detail="Unable to process image.")

    start = time.perf_counter()

    # Limit concurrency with semaphore
    async with semaphore:
        # First attempt using existing session
        try:
            logger.info("Acquiring ONNX session for %s...", filename)
            current_session = await get_session()

            logger.info("Calling rembg.remove for %s (first attempt)", filename)
            output = await run_in_threadpool(remove, normalized, session=current_session)

            logger.info("Background removal completed in %.2fs", time.perf_counter() - start)
            # Ensure memory is freed promptly
            gc.collect()

            # If rembg returned bytes, ensure we return PNG bytes and proper media type from the route
            return output

        except Exception as exc:
            # log full exception and attempt to recover
            logger.exception("Background removal failed on first attempt for %s: %s", filename, exc)

        # Retry with recreated session
        try:
            logger.warning("Recreating ONNX session and retrying for %s", filename)
            await recreate_session()
            logger.info("Calling rembg.remove for %s (retry)", filename)
            output = await run_in_threadpool(remove, normalized, session=session)

            logger.info("Retry succeeded in %.2fs", time.perf_counter() - start)
            gc.collect()
            return output

        except Exception as exc:
            logger.exception("Retry also failed for %s: %s", filename, exc)
            # Final failure for this request
            raise HTTPException(
                status_code=500,
                detail="Background removal failed due to an internal error."
            )
