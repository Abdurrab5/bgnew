from __future__ import annotations

from starlette.responses import JSONResponse

from config import settings

_MULTIPART_OVERHEAD_BYTES = 64 * 1024


class UploadBodyLimitMiddleware:
    """Bound the multipart body before Starlette spools its uploaded file."""

    def __init__(self, app):
        self.app = app
        self.body_limit = settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/bg-remove":
            await self.app(scope, receive, send)
            return

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
