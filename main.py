from __future__ import annotations

# Keep native numerical libraries conservative on small CPU instances. These
# values must be set before importing rembg/ONNX Runtime through the router.
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_DYNAMIC"] = "FALSE"

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from middleware import UploadBodyLimitMiddleware
from routes.bg_remove import router
from services.bg_service import model_ready

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("xhunta.background_remover")
PROCESS_START = time.monotonic()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting Xhunta background remover model=%s", settings.model_name)
    yield
    logger.info("Stopping Xhunta background remover")


app = FastAPI(
    title="Xhunta AI Image Background Remover",
    description="CPU-based AI image background removal API.",
    version="2.1.0",
    lifespan=lifespan,
)
app.include_router(router)
app.add_middleware(UploadBodyLimitMiddleware)


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
