from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import Response

router = APIRouter(tags=["Background Removal"])


@router.post("/bg-remove", summary="Remove image background", response_class=Response)
async def bg_remove(request: Request, file: UploadFile = File(...)) -> Response:
    # Keep route registration light so import diagnostics can isolate the service.
    from services.bg_service import remove_background

    png_bytes = await remove_background(file, request_id=request.state.request_id)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="no-bg.png"',
            "Cache-Control": "no-store, no-cache, must-revalidate",
        },
    )
