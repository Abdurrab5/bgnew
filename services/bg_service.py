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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ==========================================================
# Model session
# ==========================================================

_bg_session = None
_session_lock = None


async def initialize_session() -> None:
    global _bg_session, _session_lock

    if _bg_session is not None:
        return

    os.environ.setdefault("U2NET_HOME", MODELS_DIR)
    model_file = os.path.join(MODELS_DIR, "u2netp.onnx")
    if not os.path.exists(model_file):
        raise RuntimeError(f"Missing model file: {model_file}")

    logger.info("Loading rembg session from %s", model_file)
    _bg_session = new_session("u2netp")
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    logger.info("Background removal model loaded")


async def ensure_session() -> None:
    global _session_lock

    if _bg_session is not None:
        return

    if _session_lock is None:
        _session_lock = asyncio.Lock()

    async with _session_lock:
        if _bg_session is not None:
            return
        await initialize_session()


def _run_removal(image_bytes: bytes) -> bytes:
    if _bg_session is None:
        raise RuntimeError("Background removal session is not initialized.")

    output = remove(image_bytes, session=_bg_session, force_return_bytes=True)
    if not output or not isinstance(output, (bytes, bytearray)):
        raise RuntimeError("Background removal returned invalid output.")
    return bytes(output)

# ==========================================================
# Configuration
# ==========================================================

def positive_int(value: Optional[str], default: int) -> int:
    try:
        value = int(value)
        return value if value > 0 else default
    except Exception:
        return default


MAX_CONCURRENT = positive_int(os.getenv("MAX_CONCURRENT"), 1)
MAX_FILE_SIZE = positive_int(os.getenv("MAX_FILE_SIZE"), 3 * 1024 * 1024)
MAX_DIMENSION = positive_int(os.getenv("MAX_DIMENSION"), 512)

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/gif",
    "image/tiff",
}

logger.info(
    "Configuration: file=%d dimension=%d concurrent=%d",
    MAX_FILE_SIZE,
    MAX_DIMENSION,
    MAX_CONCURRENT,
)

try:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
except Exception:
    semaphore = None

# ==========================================================
# Image preprocessing
# ==========================================================

def preprocess_image(image_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image.")

    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

    output = io.BytesIO()
    img.save(output, format="PNG", optimize=True)
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
            raise HTTPException(status_code=413, detail="Image is too large.")
    return buffer.getvalue()

# ==========================================================
# Background removal
# ==========================================================


def _run_removal(image_bytes: bytes) -> bytes:
    if _bg_session is None:
        raise RuntimeError("Background removal session is not initialized.")

    output = remove(image_bytes, session=_bg_session, force_return_bytes=True)
    if not output or not isinstance(output, (bytes, bytearray)):
        raise RuntimeError("Background removal returned invalid output.")
    return bytes(output)


async def remove_bg(file: UploadFile) -> bytes:
    started = time.perf_counter()
    filename = file.filename or "unknown"

    logger.info("Request started | filename=%s | content_type=%s", filename, file.content_type)

    try:
        if file.content_type and file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported image type.")

        image_bytes = await read_upload(file)
        logger.info("Upload size: %d bytes", len(image_bytes))

        preprocess_started = time.perf_counter()
        normalized = preprocess_image(image_bytes)
        logger.info("Preprocessing completed in %.2fs", time.perf_counter() - preprocess_started)
        logger.info("Processed image size: %d bytes", len(normalized))

        await ensure_session()
        worker_started = time.perf_counter()
        if semaphore is not None:
            async with semaphore:
                output_bytes = await run_in_threadpool(_run_removal, normalized)
        else:
            output_bytes = await run_in_threadpool(_run_removal, normalized)
        logger.info("Background removal completed in %.2fs", time.perf_counter() - worker_started)

        if not output_bytes:
            raise HTTPException(status_code=500, detail="Background removal returned no output.")

        logger.info("Background removal succeeded.")
        return output_bytes

    except HTTPException:
        raise
    except Exception:
        logger.exception("Background removal failed.")
        raise HTTPException(status_code=500, detail="Background removal failed.")
    finally:
        gc.collect()
        logger.info("Request finished in %.2fs", time.perf_counter() - started)
