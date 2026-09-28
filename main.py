from __future__ import annotations


def _read_proc_rss_mb() -> float | None:
    """Read resident memory without importing any application dependencies."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


_BARE_START_RSS_MB = _read_proc_rss_mb()

# Import only the standard library before recording the first two boundaries.
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("xhunta.background_remover")


def _log_mem_stage(stage: str, rss_mb: float | None = None) -> None:
    value = _read_proc_rss_mb() if rss_mb is None else rss_mb
    logger.info(
        "MEM_STAGE stage=%s rss_mb=%s pid=%s",
        stage,
        f"{value:.1f}" if value is not None else "unavailable",
        os.getpid(),
    )


_log_mem_stage("bare_python_start", _BARE_START_RSS_MB)
_log_mem_stage("stdlib_imported")

# Set numerical-library thread counts before importing ONNX Runtime or NumPy.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_DYNAMIC"] = "FALSE"

from fastapi import FastAPI

_log_mem_stage("fastapi_imported")

PROCESS_START = time.monotonic()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _log_mem_stage("application_startup")
    logger.info("Starting Xhunta background remover model=%s", settings.model_name)
    yield
    logger.info("Stopping Xhunta background remover")


app = FastAPI(
    title="Xhunta AI Image Background Remover",
    description="CPU-based AI image background removal API.",
    version="2.1.0",
    lifespan=lifespan,
)
_log_mem_stage("fastapi_app_created")

from PIL import Image  # noqa: E402

_log_mem_stage("pillow_imported")

import numpy  # noqa: E402,F401

_log_mem_stage("numpy_imported")

import onnxruntime  # noqa: E402,F401

_log_mem_stage("onnxruntime_imported")

from routes.bg_remove import router  # noqa: E402

_log_mem_stage("routes_imported")

from services.bg_service import model_ready  # noqa: E402

_log_mem_stage("services_bg_service_imported")

from config import settings  # noqa: E402
from middleware import UploadBodyLimitMiddleware  # noqa: E402

app.include_router(router)
app.add_middleware(UploadBodyLimitMiddleware)


def _safe_worker_env() -> str:
    names = ("WEB_CONCURRENCY", "UVICORN_WORKERS", "FASTAPI_WORKERS", "GUNICORN_WORKERS")
    configured = [f"{name}={os.environ[name]}" for name in names if os.getenv(name, "").isdigit()]
    return ",".join(configured) if configured else "unset"


def _runtime_flags() -> tuple[str, str, str]:
    """Report only safe process flags relevant to worker/reload topology."""
    args = sys.argv[1:]
    workers = "unspecified"
    for index, arg in enumerate(args):
        if arg in {"--workers", "-w"} and index + 1 < len(args) and args[index + 1].isdigit():
            workers = args[index + 1]
        elif arg.startswith("--workers=") and arg.partition("=")[2].isdigit():
            workers = arg.partition("=")[2]
    reload_mode = "enabled" if "--reload" in args or "--reload-dir" in args else "disabled"
    preload = "enabled" if "--preload" in args or "--preload-app" in args else "disabled"
    return workers, reload_mode, preload


def _instrumentation_modules() -> str:
    roots = ("ddtrace", "opentelemetry", "newrelic", "sentry_sdk", "elasticapm", "scout_apm")
    loaded = [name for name in roots if name in sys.modules]
    return ",".join(loaded) if loaded else "none-detected"


_log_mem_stage("application_import_complete")
_cli_workers, _reload_flag, _preload_flag = _runtime_flags()
logger.info(
    "MEM_STAGE stage=runtime_topology pid=%s ppid=%s cli_workers=%s worker_env=%s reload_flag=%s reload_env=%s preload_flag=%s instrumentation=%s",
    os.getpid(),
    os.getppid(),
    _cli_workers,
    _safe_worker_env(),
    _reload_flag,
    (
        "unset"
        if "UVICORN_RELOAD" not in os.environ
        else "enabled"
        if os.getenv("UVICORN_RELOAD", "").lower() in {"1", "true", "yes"}
        else "disabled"
    ),
    _preload_flag,
    _instrumentation_modules(),
)


@app.get("/")
async def root() -> dict:
    return {
        "service": "xhunta-background-remover",
        "status": "running",
        "version": app.version,
    }


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "xhunta-background-remover",
        "model": settings.model_name,
        "ready": model_ready(),
        "uptime_seconds": round(time.monotonic() - PROCESS_START, 2),
    }


@app.get("/capabilities")
async def capabilities() -> dict:
    return {
        "service": "Xhunta AI Background Remover",
        "supported_formats": ["png", "jpg", "jpeg", "webp"],
        "output_format": "png",
        "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
    }
