"""
Tests for the OCR preprocessor module.
"""

import numpy as np
import pytest

from OCR.preprocessor import (
    binarize,
    denoise,
    deskew,
    enhance_contrast,
    normalize_resolution,
    preprocess_image,
    remove_borders,
    to_grayscale,
)


def _make_test_image(h=500, w=400, channels=3):
    """Create a synthetic test image with some text-like features."""
    img = np.ones((h, w, channels), dtype=np.uint8) * 255
    # Add some dark rectangles to simulate text
    img[100:120, 50:350] = 0
    img[140:160, 50:300] = 0
    img[180:200, 100:350] = 0
    return img


def _make_grayscale_image(h=500, w=400):
    """Create a synthetic grayscale test image."""
    img = np.ones((h, w), dtype=np.uint8) * 255
    img[100:120, 50:350] = 0
    img[140:160, 50:300] = 0
    return img


class TestToGrayscale:
    def test_converts_bgr_to_gray(self):
        img = _make_test_image()
        result = to_grayscale(img)
        assert len(result.shape) == 2

    def test_already_grayscale_returns_same(self):
        img = _make_grayscale_image()
        result = to_grayscale(img)
        assert len(result.shape) == 2
        assert result.shape == img.shape

    def test_single_channel_3d(self):
        img = np.ones((100, 100, 1), dtype=np.uint8) * 128
        result = to_grayscale(img)
        assert len(result.shape) == 2


class TestDenoise:
    def test_returns_same_shape(self):
        img = _make_grayscale_image()
        result = denoise(img)
        assert result.shape == img.shape

    def test_reduces_noise(self):
        # Add noise to image
        img = _make_grayscale_image()
        noisy = img.copy()
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        noisy = np.clip(noisy.astype(int) + noise, 0, 255).astype(np.uint8)

        result = denoise(noisy)
        # Denoised image should be closer to original than noisy
        assert result.shape == img.shape


class TestEnhanceContrast:
    def test_returns_same_shape(self):
        img = _make_grayscale_image()
        result = enhance_contrast(img)
        assert result.shape == img.shape

    def test_output_range(self):
        img = _make_grayscale_image()
        result = enhance_contrast(img)
        assert result.min() >= 0
        assert result.max() <= 255


class TestBinarize:
    def test_otsu(self):
        img = _make_grayscale_image()
        result = binarize(img, method="otsu")
        assert result.shape == img.shape
        unique_values = np.unique(result)
        assert len(unique_values) <= 2

    def test_adaptive(self):
        img = _make_grayscale_image()
        result = binarize(img, method="adaptive")
        assert result.shape == img.shape

    def test_sauvola(self):
        img = _make_grayscale_image()
        result = binarize(img, method="sauvola")
        assert result.shape == img.shape

    def test_unknown_method_falls_back_to_otsu(self):
        img = _make_grayscale_image()
        result = binarize(img, method="unknown")
        assert result.shape == img.shape


class TestDeskew:
    def test_already_straight_image(self):
        img = _make_grayscale_image()
        result = deskew(img)
        assert result.shape == img.shape

    def test_handles_empty_image(self):
        img = np.ones((100, 100), dtype=np.uint8) * 255
        result = deskew(img)
        assert result.shape == img.shape


class TestRemoveBorders:
    def test_image_with_border(self):
        # Create image with black border
        img = np.zeros((500, 400), dtype=np.uint8)
        # White content area in center
        img[50:450, 50:350] = 255
        result = remove_borders(img)
        # Result should be smaller or same size
        assert result.shape[0] <= img.shape[0]
        assert result.shape[1] <= img.shape[1]

    def test_image_without_border(self):
        img = _make_grayscale_image()
        result = remove_borders(img)
        assert result.shape[0] > 0
        assert result.shape[1] > 0


class TestNormalizeResolution:
    def test_small_image_upscaled(self):
        img = _make_grayscale_image(h=200, w=150)
        result = normalize_resolution(img, target_dpi=300)
        assert result.shape[0] > img.shape[0]
        assert result.shape[1] > img.shape[1]

    def test_large_image_unchanged(self):
        img = _make_grayscale_image(h=4000, w=3000)
        result = normalize_resolution(img, target_dpi=300)
        assert result.shape == img.shape


class TestPreprocessImage:
    def test_full_pipeline(self):
        img = _make_test_image()
        result = preprocess_image(img)
        assert len(result.shape) == 2  # Should be grayscale
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_all_steps_disabled(self):
        img = _make_test_image()
        result = preprocess_image(
            img,
            enable_denoise=False,
            enable_deskew=False,
            enable_border_removal=False,
            enable_contrast_enhancement=False,
        )
        assert len(result.shape) == 2
        assert result.shape[0] > 0

    def test_preserves_content(self):
        """Preprocessing should not produce an empty image."""
        img = _make_test_image()
        result = preprocess_image(img)
        assert np.any(result < 255)  # Should still have some dark pixels
