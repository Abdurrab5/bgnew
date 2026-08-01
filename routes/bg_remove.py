from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response

from services.bg_service import remove_bg

router = APIRouter()


@router.post("/bg-remove")
async def bg_remove(file: UploadFile = File(...)):

    output = await remove_bg(file)

    return Response(
        content=output,
        media_type="image/png"
    )