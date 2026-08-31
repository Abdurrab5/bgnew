from __future__ import annotations

import logging

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response

from services.bg_service import remove_background


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Background Removal"],
)


# ============================================================
# BACKGROUND REMOVAL
# ============================================================

@router.post(
    "/bg-remove",
    summary="Remove image background",
    response_class=Response,
)
async def bg_remove(
    file: UploadFile = File(...),
) -> Response:

    logger.info(
        "BG REQUEST | filename=%s | content_type=%s",
        file.filename,
        file.content_type,
    )

    result = await remove_background(file)

    logger.info(
        "BG RESPONSE | filename=%s | output_bytes=%d",
        file.filename,
        len(result),
    )

    return Response(
        content=result,
        media_type="image/png",
        headers={
            "Content-Disposition": (
                'inline; filename="no-bg.png"'
            ),
            "Cache-Control": "no-store",
        },
    )