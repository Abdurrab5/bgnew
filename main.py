import os
import logging
import asyncio

# Project directory and model cache path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ.setdefault("U2NET_HOME", MODELS_DIR)

# Limit CPU threads used by BLAS/OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_DYNAMIC"] = "FALSE"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.bg_remove import router

app = FastAPI(
    title="BG Remove API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def preload_model():
    from services.bg_service import get_session

    try:
        asyncio.create_task(get_session())
    except Exception:
        logging.exception("Failed to create background model preload task")


@app.get("/")
async def root():
    return {"status": "BG API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )