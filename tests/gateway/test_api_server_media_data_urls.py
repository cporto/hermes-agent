"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.
"""

import base64
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import _MediaStreamBuffer, _resolve_media_to_data_urls

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


class TestResolveMediaToDataUrls(unittest.TestCase):
    def _write_png(self, tmpdir_name="hermes_media_test"):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix=tmpdir_name))
        p = d / "shot.png"
        p.write_bytes(_PNG_BYTES)
        return p

    def test_media_tag_inlined(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"Here you go: MEDIA:{p}")
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)

    def test_backtick_wrapped_tag(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"See `MEDIA:{p}` above")
        self.assertIn("data:image/png;base64,", out)

    def test_terminal_eos_sentinel_does_not_block_inlining(self):
        """A leaked terminal ``<|eos|>`` glued to the tag (#111046) inlines exactly like the clean
        response; the control token is dropped rather than returned to the HTTP client."""
        p = self._write_png()
        clean = f"Here you go: MEDIA:{p}"
        self.assertEqual(_resolve_media_to_data_urls(clean + "<|eos|>"), _resolve_media_to_data_urls(clean))

    def test_missing_file_left_untouched(self):
        text = "MEDIA:/nonexistent/path/shot.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_non_image_left_untouched(self):
        text = "MEDIA:/tmp/archive.zip"
        self.assertEqual(_resolve_media_to_data_urls(text), text)


class TestMediaStreamHoldback(unittest.TestCase):
    def _write_png(self):
        import tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp(prefix="hermes_media_stream"))
        p = d / "shot.png"
        p.write_bytes(_PNG_BYTES)
        return p

    def test_tag_split_across_deltas_reassembled(self):
        """A MEDIA: tag split mid-path across two deltas is held back and only emitted as a
        resolved data URL once complete, mirroring the non-streaming path."""
        p = self._write_png()
        stem = p.with_suffix("").name
        ext = p.suffix
        buf = _MediaStreamBuffer()
        first = buf.push(f"result: MEDIA:{p.parent}/")
        self.assertNotIn("MEDIA:", first)
        second = buf.push(f"{stem}{ext}")
        self.assertIn("data:image/png;base64,", first + second)
        self.assertNotIn("MEDIA:", first + second)
        self.assertEqual(buf.flush(), "")

    def test_tag_fully_in_one_delta(self):
        p = self._write_png()
        buf = _MediaStreamBuffer()
        out = buf.push(f"Here: MEDIA:{p}")
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)
        self.assertEqual(buf.flush(), "")

    def test_plain_text_unchanged(self):
        buf = _MediaStreamBuffer()
        self.assertEqual(buf.push("just plain text"), "just plain text")
        self.assertEqual(buf.flush(), "")


if __name__ == "__main__":
    unittest.main()
