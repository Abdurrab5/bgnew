from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response

from services.bg_service import remove_background

router = APIRouter(tags=["Background Removal"])


@router.post("/bg-remove", summary="Remove image background", response_class=Response)
async def bg_remove(file: UploadFile = File(...)) -> Response:
    png_bytes = await remove_background(file)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="no-bg.png"',
            "Cache-Control": "no-store, no-cache, must-revalidate",
        },
    )
