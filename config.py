"""Environment-backed settings for the background-removal service."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_setting(name: str, default: int, *, legacy: str | None = None) -> int:
    raw = os.getenv(name)
    if raw is None and legacy:
        raw = os.getenv(legacy)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    model_name: str
    max_upload_bytes: int
    max_image_pixels: int
    max_inference_dimension: int
    max_concurrent_jobs: int
    log_level: str


_max_upload_mb = os.getenv("MAX_UPLOAD_MB")
_legacy_max_bytes = os.getenv("BG_MAX_FILE_SIZE")
if _max_upload_mb is not None:
    _max_upload_bytes = _int_setting("MAX_UPLOAD_MB", 5) * 1024 * 1024
elif _legacy_max_bytes is not None:
    _max_upload_bytes = _int_setting("BG_MAX_FILE_SIZE", 5 * 1024 * 1024)
else:
    _max_upload_bytes = 5 * 1024 * 1024

settings = Settings(
    model_name=os.getenv("MODEL_NAME", os.getenv("BG_MODEL", "u2netp")),
    max_upload_bytes=_max_upload_bytes,
    max_image_pixels=_int_setting("MAX_IMAGE_PIXELS", 25_000_000, legacy="BG_MAX_IMAGE_PIXELS"),
    max_inference_dimension=_int_setting("MAX_INFERENCE_DIMENSION", 2048, legacy="BG_MAX_INFERENCE_DIMENSION"),
    max_concurrent_jobs=_int_setting("MAX_CONCURRENT_JOBS", 1, legacy="BG_MAX_CONCURRENT"),
    log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
