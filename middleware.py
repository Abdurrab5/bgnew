from __future__ import annotations

import logging
import os
import threading
import uuid

from starlette.responses import JSONResponse

from config import settings

_MULTIPART_OVERHEAD_BYTES = 64 * 1024
logger = logging.getLogger(__name__)


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


# This is a rough estimate of the overhead that multipart/form-data adds to the request body.
class UploadBodyLimitMiddleware:
    """Bound the multipart body before Starlette spools its uploaded file."""

    def __init__(self, app):
        self.app = app
        self.body_limit = settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES
        self._first_request_logged = False
        self._first_request_lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/bg-remove":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:10]
        scope.setdefault("state", {})["request_id"] = request_id
        with self._first_request_lock:
            if not self._first_request_logged:
                rss = _rss_mb()
                logger.info(
                    "MEM_STAGE request_id=%s stage=before_first_request rss_mb=%s pid=%s",
                    request_id,
                    f"{rss:.1f}" if rss is not None else "unavailable",
                    os.getpid(),
                )
                self._first_request_logged = True

        content_length = next(
            (value for key, value in scope.get("headers", []) if key.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.body_limit:
                    await self._too_large(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.body_limit:
                    raise _UploadBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _UploadBodyTooLarge:
            await self._too_large(scope, receive, send)

    @staticmethod
    async def _too_large(scope, receive, send):
        response = JSONResponse(
            status_code=413,
            content={"detail": f"Maximum upload size is {settings.max_upload_bytes // (1024 * 1024)} MB."},
        )
        await response(scope, receive, send)


class _UploadBodyTooLarge(Exception):
    pass
