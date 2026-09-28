from __future__ import annotations

import asyncio
import fcntl
import hashlib
import io
import logging
import os
import tempfile
import threading
import time
import urllib.request
import uuid
import warnings

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

from config import settings

logger = logging.getLogger(__name__)
_session: ort.InferenceSession | None = None
_session_lock = threading.Lock()
_inference_slots = asyncio.Semaphore(settings.max_concurrent_jobs)

_MIME_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
}
_INFERENCE_SIZE = (320, 320)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_MODEL_FILENAME = "u2netp.onnx"
_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
_MODEL_SIZE_BYTES = 4_574_861
_MODEL_MD5 = "8e83ca70e441ab06c318d82300c84806"


def _rss_mb() -> float | None:
    """Read Linux resident memory without adding a runtime dependency."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _log_stage(
    request_id: str,
    stage: str,
    started: float,
    *,
    original_size: tuple[int, int] | None = None,
) -> None:
    rss = _rss_mb()
    fields = [f"request_id={request_id}", f"stage={stage}"]
    if rss is not None:
        fields.append(f"rss_mb={rss:.1f}")
    fields.append(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    if original_size is not None:
        fields.append(f"original={original_size[0]}x{original_size[1]}")
    if stage in {"inference_prepared", "before_inference", "after_inference"}:
        fields.append(f"inference={_INFERENCE_SIZE[0]}x{_INFERENCE_SIZE[1]}")
    fields.append(f"pid={os.getpid()}")
    logger.info("MEM_STAGE %s", " ".join(fields))


def model_ready() -> bool:
    return _session is not None


def _model_md5(path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_file(path) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError("u2netp model file is unavailable") from exc
    if size != _MODEL_SIZE_BYTES:
        raise RuntimeError("u2netp model file has an unexpected size")
    if _model_md5(path) != _MODEL_MD5:
        raise RuntimeError("u2netp model checksum verification failed")


def _ensure_model_file():
    """Verify the cached model or download its pinned official artifact atomically."""
    cache_dir = settings.model_cache_dir
    model_path = cache_dir / _MODEL_FILENAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f"{_MODEL_FILENAME}.lock"

    # The process session lock already prevents duplicate threads; flock also avoids
    # parallel downloads if a host starts more than one worker against this cache.
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if model_path.exists():
            _validate_model_file(model_path)
            return model_path

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{_MODEL_FILENAME}.",
            suffix=".download",
            dir=cache_dir,
        )
        temporary_path = type(model_path)(temporary_name)
        try:
            digest = hashlib.md5(usedforsecurity=False)
            downloaded = 0
            with os.fdopen(descriptor, "wb") as target:
                with urllib.request.urlopen(_MODEL_URL, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > _MODEL_SIZE_BYTES:
                            raise RuntimeError("u2netp download exceeded its expected size")
                        digest.update(chunk)
                        target.write(chunk)
                target.flush()
                os.fsync(target.fileno())

            if downloaded != _MODEL_SIZE_BYTES:
                raise RuntimeError("u2netp download has an unexpected size")
            if digest.hexdigest() != _MODEL_MD5:
                raise RuntimeError("u2netp download checksum verification failed")
            os.replace(temporary_path, model_path)
            _validate_model_file(model_path)
            return model_path
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _get_model_session(request_id: str | None = None) -> ort.InferenceSession:
    """Download and verify once, then cache one constrained ORT session."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                started = time.perf_counter()
                rss = _rss_mb()
                logger.info(
                    "MEM_STAGE stage=before_model_init request_id=%s rss_mb=%s pid=%s model=%s",
                    request_id or "startup",
                    f"{rss:.1f}" if rss is not None else "unavailable",
                    os.getpid(),
                    settings.model_name,
                )
                try:
                    if settings.model_name != "u2netp":
                        raise RuntimeError("Direct ONNX inference is configured for u2netp only")
                    model_path = _ensure_model_file()
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    # Avoid retaining a large CPU allocator arena after inference on small hosts.
                    options.enable_cpu_mem_arena = False
                    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
                    candidate = ort.InferenceSession(
                        model_path,
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    inputs = candidate.get_inputs()
                    outputs = candidate.get_outputs()
                    if not inputs or not outputs:
                        raise RuntimeError("u2netp ONNX model has no usable inputs or outputs")
                    input_shape = inputs[0].shape
                    if len(input_shape) != 4 or input_shape[1:] != [3, 320, 320]:
                        raise RuntimeError("u2netp ONNX model does not accept the expected 320x320 RGB tensor")
                    # Publish only a fully initialized session.
                    _session = candidate
                except Exception:
                    logger.exception(
                        "MEM_STAGE stage=model_init_failed request_id=%s duration_seconds=%.3f pid=%s",
                        request_id or "startup",
                        time.perf_counter() - started,
                        os.getpid(),
                    )
                    raise
                rss = _rss_mb()
                logger.info(
                    "MEM_STAGE stage=after_model_init request_id=%s rss_mb=%s duration_seconds=%.3f pid=%s provider=%s",
                    request_id or "startup",
                    f"{rss:.1f}" if rss is not None else "unavailable",
                    time.perf_counter() - started,
                    os.getpid(),
                    ",".join(candidate.get_providers()),
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


def _decode_image(data: bytes, declared_type: str | None) -> tuple[Image.Image, tuple[int, int]]:
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
                try:
                    original_size = oriented.size
                    rgb_image = oriented.convert("RGB")
                finally:
                    if oriented is not source:
                        oriented.close()
                return rgb_image, original_size
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc
    except Exception as exc:
        logger.warning("Image decoding failed: %s", type(exc).__name__)
        raise HTTPException(status_code=400, detail="Unable to read the uploaded image.") from exc


def _make_input_tensor(image: Image.Image) -> np.ndarray:
    """Reproduce rembg U2netpSession.normalize at its exact fixed 320px input."""
    resized = image.resize(_INFERENCE_SIZE, Image.Resampling.LANCZOS)
    try:
        pixels = np.asarray(resized)
        divisor = max(int(pixels.max()), 1e-6)
        tensor = np.empty((1, 3, _INFERENCE_SIZE[1], _INFERENCE_SIZE[0]), dtype=np.float32)
        for channel in range(3):
            plane = tensor[0, channel]
            plane[:] = pixels[:, :, channel]
            plane /= divisor
            plane -= _MEAN[channel]
            plane /= _STD[channel]
        return tensor
    finally:
        resized.close()


def _infer_and_encode(
    image: Image.Image,
    original_size: tuple[int, int],
    session,
    request_id: str,
    started: float,
) -> bytes:
    tensor = _make_input_tensor(image)
    _log_stage(request_id, "inference_prepared", started, original_size=original_size)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    _log_stage(request_id, "before_inference", started, original_size=original_size)
    outputs = session.run([output_name], {input_name: tensor})
    _log_stage(request_id, "after_inference", started, original_size=original_size)
    del tensor

    prediction = outputs[0][0, 0]
    minimum = float(prediction.min())
    maximum = float(prediction.max())
    if maximum > minimum:
        prediction -= minimum
        prediction *= 255.0 / (maximum - minimum)
    else:
        prediction.fill(0)
    mask_pixels = np.clip(prediction, 0, 255).astype(np.uint8)
    del outputs, prediction

    mask = Image.fromarray(mask_pixels)
    del mask_pixels
    try:
        if mask.size != original_size:
            resized_mask = mask.resize(original_size, Image.Resampling.LANCZOS)
            mask.close()
            mask = resized_mask
        _log_stage(request_id, "mask_resized", started, original_size=original_size)
        # RGB->RGBA is performed by Pillow here; do not create a full-resolution float array.
        image.putalpha(mask)
        _log_stage(request_id, "rgba_composed", started, original_size=original_size)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=False, compress_level=6)
        _log_stage(request_id, "png_encoded", started, original_size=original_size)
        return output.getvalue()
    finally:
        mask.close()


async def remove_background(file: UploadFile, request_id: str | None = None) -> bytes:
    started = time.perf_counter()
    request_id = request_id or uuid.uuid4().hex[:10]
    image = None
    acquired = False
    original_size = None
    try:
        _log_stage(request_id, "after_multipart_parse", started)
        data = await _read_limited(file)
        _log_stage(request_id, "upload_read", started)
        image, original_size = await run_in_threadpool(_decode_image, data, file.content_type)
        del data
        _log_stage(request_id, "image_decoded", started, original_size=original_size)

        if _inference_slots.locked():
            logger.info("Request rejected as capacity is full request_id=%s", request_id)
            raise HTTPException(status_code=503, detail="Background removal is busy. Please retry shortly.")
        await _inference_slots.acquire()
        acquired = True

        try:
            session = await run_in_threadpool(_get_model_session, request_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Background removal is temporarily unavailable.") from exc

        inference_started = time.perf_counter()
        result = await run_in_threadpool(
            _infer_and_encode,
            image,
            original_size,
            session,
            request_id,
            started,
        )
        logger.info(
            "Image processed request_id=%s model=%s inference_seconds=%.3f total_seconds=%.3f output_bytes=%d",
            request_id,
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
        _log_stage(request_id, "request_complete", started, original_size=original_size)
