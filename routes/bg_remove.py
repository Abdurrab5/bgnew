from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response
import logging
from services.bg_service import remove_bg
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/bg-remove")
async def bg_remove(file: UploadFile = File(...)):
    logger.info("POST /bg-remove received")

    output = await remove_bg(file)

    logger.info("Background removed successfully")

    return Response(
        content=output,
        media_type="image/png"
    )

 