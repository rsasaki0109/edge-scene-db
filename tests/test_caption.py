"""Tests for scene_db.caption."""

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from scene_db.caption import _encode_image, _pick_representative_image, generate_vlm_caption


class TestEncodeImage:
    def test_encodes_file_to_base64(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake image data")
        result = _encode_image(img)
        decoded = base64.b64decode(result)
        assert decoded == b"\x89PNG\r\n\x1a\nfake image data"

    def test_roundtrip(self, tmp_path):
        img = tmp_path / "binary.bin"
        original = bytes(range(256))
        img.write_bytes(original)
        encoded = _encode_image(img)
        assert base64.b64decode(encoded) == original


class TestPickRepresentativeImage:
    def test_empty_list_returns_none(self):
        assert _pick_representative_image([]) is None

    def test_single_image(self):
        p = Path("/img/0.png")
        assert _pick_representative_image([p]) == p

    def test_picks_middle(self):
        paths = [Path(f"/img/{i}.png") for i in range(5)]
        # middle of 5 items -> index 2
        assert _pick_representative_image(paths) == paths[2]

    def test_even_count_picks_lower_middle(self):
        paths = [Path(f"/img/{i}.png") for i in range(4)]
        # len=4, index 2
        assert _pick_representative_image(paths) == paths[2]


class TestGenerateVlmCaption:
    def test_fallback_when_no_api_key(self):
        """Without OPENAI_API_KEY, should fall back to rule-based caption."""
        with patch.dict("os.environ", {}, clear=True):
            caption = generate_vlm_caption([], 30.0, 100.0)
        assert "moving forward" in caption
        assert "30 km/h" in caption

    def test_fallback_when_no_images(self):
        """Even if API key existed, no images -> rule-based fallback."""
        with patch.dict("os.environ", {}, clear=True):
            caption = generate_vlm_caption([], 0.0, 0.0)
        assert "stationary" in caption

    def test_fallback_when_image_missing(self, tmp_path):
        """Image path that doesn't exist -> rule-based fallback."""
        missing = tmp_path / "nonexistent.png"
        with patch.dict("os.environ", {}, clear=True):
            caption = generate_vlm_caption([missing], 80.0, 500.0)
        assert "high speed" in caption

    def test_fallback_on_api_error(self, tmp_path):
        """If OpenAI client raises, should fall back to rule-based."""
        img = tmp_path / "test.png"
        img.write_bytes(b"fake")

        mock_client = type("MockClient", (), {})()
        mock_chat = type("MockChat", (), {})()
        mock_completions = type("MockCompletions", (), {})()

        def create_raise(**kwargs):
            raise RuntimeError("API error")

        mock_completions.create = create_raise
        mock_chat.completions = mock_completions
        mock_client.chat = mock_chat

        with patch("scene_db.caption._get_openai_client", return_value=mock_client):
            caption = generate_vlm_caption([img], 10.0, 5.0)
        assert "slowly" in caption
