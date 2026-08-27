
import os
import time
import logging

from fastapi import FastAPI

from routes.bg_remove import router


# =========================================================
# CPU / THREAD LIMITS
# =========================================================

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# PROCESS INFORMATION
# =========================================================

PROCESS_START = time.time()
PROCESS_ID = os.getpid()


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="BG Remove API",
    version="1.1",
)


app.include_router(router)


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup_event():

    logger.info("=" * 60)
    logger.info("BG REMOVE API STARTED")
    logger.info("PID: %s", PROCESS_ID)
    logger.info("Python PID process initialized")
    logger.info("=" * 60)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "status": "BG API running",
        "pid": os.getpid(),
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():

    uptime = time.time() - PROCESS_START

    return {
        "status": "healthy",
        "pid": os.getpid(),
        "uptime_seconds": round(uptime, 2),
    }