from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI

from routes.bg_remove import router


# ============================================================
# CPU / ONNX RUNTIME CONFIGURATION
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("bgnew")


# ============================================================
# PROCESS INFORMATION
# ============================================================

PROCESS_START = time.monotonic()


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Xhunta AI Image Background Remover",
    description=(
        "CPU-based AI image background removal API "
        "with browser-compatible FastAPI fallback."
    ),
    version="2.0.0",
)


app.include_router(router)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root() -> dict:

    return {
        "service": "xhunta-background-remover",
        "status": "running",
        "version": app.version,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health() -> dict:

    return {
        "status": "healthy",
        "uptime_seconds": round(
            time.monotonic() - PROCESS_START,
            2,
        ),
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup() -> None:

    logger.info("=" * 64)
    logger.info("XHUNTA BACKGROUND REMOVER STARTED")
    logger.info("PID=%s", os.getpid())
    logger.info("VERSION=%s", app.version)
    logger.info("=" * 64)