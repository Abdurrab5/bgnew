from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response

from services.bg_service import remove_bg

router = APIRouter()


@router.post(
    "/bg-remove",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        400: {"description": "Invalid image or request."},
        413: {"description": "Uploaded image is too large."},
        500: {"description": "Internal background removal error."},
    },
)
async def bg_remove(file: UploadFile = File(..., description="PNG, JPEG or WEBP image for background removal")):

    output = await remove_bg(file)

    return Response(
        content=output,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )