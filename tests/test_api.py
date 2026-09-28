from __future__ import annotations

import io
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import services.bg_service as bg_service
from main import app


def image_bytes(image_format: str) -> bytes:
    stream = io.BytesIO()
    image = Image.new("RGB", (3, 2), "blue")
    image.save(stream, format=image_format)
    image.close()
    return stream.getvalue()


def fake_rembg(image, session):
    return Image.new("RGBA", image.size, (30, 60, 90, 128))


class BackgroundRemovalAPITests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(bg_service, "_get_model_session", return_value=object()),
            patch.object(bg_service, "remove", side_effect=fake_rembg),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def assert_png_response(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertTrue(response.content.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(io.BytesIO(response.content)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getpixel((0, 0))[3], 128)

    def upload(self, content, content_type):
        return self.client.post(
            "/bg-remove",
            files={"file": ("input", content, content_type)},
        )

    def test_health_is_inexpensive_and_reports_lazy_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "xhunta-background-remover")
        self.assertEqual(response.json()["model"], "u2netp")
        self.assertFalse(response.json()["ready"])

    def test_png_jpeg_and_webp_return_raw_transparent_png(self):
        for image_format, content_type in (
            ("PNG", "image/png"),
            ("JPEG", "image/jpeg"),
            ("WEBP", "image/webp"),
        ):
            with self.subTest(format=image_format):
                self.assert_png_response(self.upload(image_bytes(image_format), content_type))

    def test_missing_file_returns_validation_error(self):
        self.assertEqual(self.client.post("/bg-remove").status_code, 422)

    def test_empty_file_is_rejected(self):
        response = self.upload(b"", "image/png")
        self.assertEqual(response.status_code, 400)

    def test_unsupported_mime_is_rejected(self):
        response = self.upload(b"not relevant", "image/gif")
        self.assertEqual(response.status_code, 415)

    def test_fake_or_corrupt_image_is_rejected(self):
        for content in (b"not an image", b"\x89PNG\r\n\x1a\ntruncated"):
            with self.subTest(content=content):
                self.assertEqual(self.upload(content, "image/png").status_code, 400)

    def test_mime_must_match_image_content(self):
        self.assertEqual(self.upload(image_bytes("JPEG"), "image/png").status_code, 415)

    def test_upload_size_limit_is_enforced(self):
        for size in (5 * 1024 * 1024 + 1, 5 * 1024 * 1024 + 64 * 1024 + 1):
            with self.subTest(size=size):
                response = self.upload(b"x" * size, "image/png")
                self.assertEqual(response.status_code, 413)

    def test_pixel_limit_is_enforced(self):
        updated = replace(bg_service.settings, max_image_pixels=5)
        with patch.object(bg_service, "settings", updated):
            self.assertEqual(self.upload(image_bytes("PNG"), "image/png").status_code, 413)

    def test_inference_failure_returns_safe_non_200(self):
        with patch.object(bg_service, "remove", side_effect=RuntimeError("private internals")):
            response = self.upload(image_bytes("PNG"), "image/png")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Background removal failed during image processing."},
        )
        self.assertNotIn("private internals", response.text)

    def test_model_initialization_failure_returns_503(self):
        with patch.object(bg_service, "_get_model_session", side_effect=RuntimeError("model unavailable")):
            response = self.upload(image_bytes("PNG"), "image/png")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": "Background removal is temporarily unavailable."},
        )

    def test_capacity_limit_rejects_a_second_active_job(self):
        inference_started = threading.Event()
        release_inference = threading.Event()

        def slow_rembg(*_args, **_kwargs):
            inference_started.set()
            if not release_inference.wait(timeout=5):
                raise TimeoutError("test inference release timed out")
            return fake_rembg(_args[0], _kwargs.get("session"))

        response_holder = []

        def first_request():
            with TestClient(app) as other_client:
                response_holder.append(
                    other_client.post(
                        "/bg-remove",
                        files={"file": ("input.png", image_bytes("PNG"), "image/png")},
                    )
                )

        worker = threading.Thread(target=first_request)
        with patch.object(bg_service, "remove", side_effect=slow_rembg):
            worker.start()
            self.assertTrue(inference_started.wait(timeout=5))
            second_response = self.upload(image_bytes("PNG"), "image/png")
            release_inference.set()
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(second_response.status_code, 503)
        self.assertEqual(response_holder[0].status_code, 200)


if __name__ == "__main__":
    unittest.main()
