import logging

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response

from services.bg_service import remove_bg


logger = logging.getLogger(__name__)

router = APIRouter()


# =========================================================
# BACKGROUND REMOVAL
# =========================================================

@router.post("/bg-remove")
async def bg_remove(file: UploadFile = File(...)):

    logger.info(
        "BG REMOVE REQUEST | filename=%s | content_type=%s",
        file.filename,
        file.content_type,
    )

    output = await remove_bg(file)

    logger.info(
        "BG REMOVE RESPONSE | filename=%s | output_size=%.2f MB",
        file.filename,
        len(output) / 1024 / 1024,
    )

    return Response(
        content=output,
        media_type="image/png",
    )