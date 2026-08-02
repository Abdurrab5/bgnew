import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from services.bg_service import remove_bg

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Background Removal"],
)


@router.post(
    "/bg-remove",
    summary="Remove image background",
)
async def bg_remove(
    file: UploadFile = File(...),
):
    started = time.perf_counter()

    logger.info(
        "Request received filename=%s content_type=%s",
        file.filename,
        file.content_type,
    )

    try:
        output = await remove_bg(file)

        if not output:
            logger.error("Background removal returned empty output.")

            raise HTTPException(
                status_code=500,
                detail="Failed to generate output image.",
            )

        elapsed = time.perf_counter() - started

        logger.info(
            "Request completed in %.2fs",
            elapsed,
        )

        return Response(
            content=output,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'inline; filename="output.png"',
            },
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Unhandled exception during background removal."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )