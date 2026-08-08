import logging
import os

# ==========================================================
# Environment
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
os.environ.setdefault("U2NET_HOME", MODELS_DIR)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.bg_remove import router
from services.bg_service import initialize_session

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(
    title="Background Removal API",
    description="High-performance background removal service powered by rembg and U²Net.",
    version="1.0.0",
    debug=False,
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting BG Remove API...")
    logger.info("U2NET_HOME=%s", os.environ.get("U2NET_HOME"))
    await initialize_session()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("BG Remove API stopped.")

# ==========================================================
# Middleware
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Routes
# ==========================================================

app.include_router(router)

# ==========================================================
# Health
# ==========================================================


@app.get("/", tags=["System"])
async def root():
    return {
        "status": "running",
        "service": "Background Removal API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
    }

# ==========================================================
# Local Development
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        access_log=True,
        reload=False,
    )
