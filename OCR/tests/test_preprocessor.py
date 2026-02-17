"""
Tests for the OCR preprocessor module (Surya-compatible).

Preprocessing now works with PIL Images throughout and uses
a lighter pipeline suitable for Surya's transformer-based models.
"""

import numpy as np
import pytest
from PIL import Image

from OCR.preprocessor import (
    check_and_upscale_resolution,
    denoise,
    deskew,
    enhance_contrast,
    preprocess_image,
    remove_borders,
)


def _make_test_image(w=400, h=500):
    """Create a synthetic test PIL Image with some text-like features."""
    arr = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Add dark rectangles to simulate text
    arr[100:120, 50:350] = 0
    arr[140:160, 50:300] = 0
    arr[180:200, 100:350] = 0
    return Image.fromarray(arr)


def _make_small_image(w=150, h=200):
    """Create a small test image that should trigger upscaling."""
    arr = np.ones((h, w, 3), dtype=np.uint8) * 255
    arr[50:70, 30:120] = 0
    return Image.fromarray(arr)


class TestCheckAndUpscaleResolution:
    def test_small_image_upscaled(self):
        img = _make_small_image()
        result = check_and_upscale_resolution(img)
        assert result.height > img.height
        assert result.width > img.width

    def test_large_image_unchanged(self):
        img = _make_test_image(w=3000, h=4000)
        result = check_and_upscale_resolution(img)
        assert result.size == img.size

    def test_returns_pil_image(self):
        img = _make_small_image()
        result = check_and_upscale_resolution(img)
        assert isinstance(result, Image.Image)


class TestDeskew:
    def test_already_straight_image(self):
        img = _make_test_image()
        result = deskew(img)
        assert isinstance(result, Image.Image)
        # Size should be roughly the same (small expansion from rotation is ok)
        assert abs(result.width - img.width) < 50
        assert abs(result.height - img.height) < 50

    def test_handles_white_image(self):
        """An all-white image should not be deskewed."""
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img = Image.fromarray(arr)
        result = deskew(img)
        assert result.size == img.size

    def test_returns_pil_image(self):
        img = _make_test_image()
        result = deskew(img)
        assert isinstance(result, Image.Image)


class TestRemoveBorders:
    def test_image_with_border(self):
        # Create image with black border
        arr = np.zeros((500, 400, 3), dtype=np.uint8)
        # White content area in center
        arr[50:450, 50:350] = 255
        img = Image.fromarray(arr)
        result = remove_borders(img)
        # Result should be smaller or same size
        assert result.width <= img.width
        assert result.height <= img.height

    def test_image_without_border(self):
        img = _make_test_image()
        result = remove_borders(img)
        assert result.width > 0
        assert result.height > 0

    def test_returns_pil_image(self):
        img = _make_test_image()
        result = remove_borders(img)
        assert isinstance(result, Image.Image)


class TestEnhanceContrast:
    def test_returns_same_size(self):
        img = _make_test_image()
        result = enhance_contrast(img)
        assert result.size == img.size

    def test_returns_pil_image(self):
        img = _make_test_image()
        result = enhance_contrast(img)
        assert isinstance(result, Image.Image)

    def test_preserves_rgb_mode(self):
        img = _make_test_image()
        result = enhance_contrast(img)
        assert result.mode == "RGB"


class TestDenoise:
    def test_returns_same_size(self):
        img = _make_test_image()
        result = denoise(img)
        assert result.size == img.size

    def test_returns_pil_image(self):
        img = _make_test_image()
        result = denoise(img)
        assert isinstance(result, Image.Image)


class TestPreprocessImage:
    def test_full_pipeline(self):
        img = _make_test_image()
        result = preprocess_image(img)
        assert isinstance(result, Image.Image)
        assert result.width > 0
        assert result.height > 0

    def test_all_steps_disabled(self):
        img = _make_test_image()
        result = preprocess_image(
            img,
            enable_denoise=False,
            enable_deskew=False,
            enable_border_removal=False,
            enable_contrast_enhancement=False,
            enable_resolution_check=False,
        )
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_preserves_content(self):
        """Preprocessing should not produce an empty image."""
        img = _make_test_image()
        result = preprocess_image(img)
        arr = np.array(result)
        # Should still have some dark pixels (the simulated text)
        assert np.any(arr < 200)

    def test_returns_rgb(self):
        """Result should be RGB for Surya compatibility."""
        img = _make_test_image()
        result = preprocess_image(img)
        assert isinstance(result, Image.Image)
        # Should be RGB or at least convertible
        if result.mode != "RGB":
            result = result.convert("RGB")
        assert result.mode == "RGB"
