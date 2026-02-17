"""
preprocessor.py

OpenCV-based image preprocessing pipeline to maximize OCR accuracy
on Arabic legal documents.

Each preprocessing step is independently toggleable via config.
The pipeline returns processed images as numpy arrays ready for
the EasyOCR engine.
"""

import logging
import math
from typing import Optional

import cv2
import numpy as np

from . import config

logger = logging.getLogger(__name__)


def preprocess_image(
    image: np.ndarray,
    enable_denoise: Optional[bool] = None,
    enable_deskew: Optional[bool] = None,
    enable_border_removal: Optional[bool] = None,
    enable_contrast_enhancement: Optional[bool] = None,
    binarization_method: Optional[str] = None,
    target_dpi: Optional[int] = None,
) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a single image.

    Args:
        image: Input image as BGR numpy array.
        enable_denoise: Override config ENABLE_DENOISE.
        enable_deskew: Override config ENABLE_DESKEW.
        enable_border_removal: Override config ENABLE_BORDER_REMOVAL.
        enable_contrast_enhancement: Override config ENABLE_CONTRAST_ENHANCEMENT.
        binarization_method: Override config BINARIZATION_METHOD.
        target_dpi: Override config TARGET_DPI.

    Returns:
        Preprocessed image as numpy array.
    """
    if enable_denoise is None:
        enable_denoise = config.ENABLE_DENOISE
    if enable_deskew is None:
        enable_deskew = config.ENABLE_DESKEW
    if enable_border_removal is None:
        enable_border_removal = config.ENABLE_BORDER_REMOVAL
    if enable_contrast_enhancement is None:
        enable_contrast_enhancement = config.ENABLE_CONTRAST_ENHANCEMENT
    if binarization_method is None:
        binarization_method = config.BINARIZATION_METHOD
    if target_dpi is None:
        target_dpi = config.TARGET_DPI

    result = image.copy()

    # 1. Grayscale conversion
    result = to_grayscale(result)
    logger.debug("Grayscale conversion done")

    # 2. Noise removal
    if enable_denoise:
        result = denoise(result)
        logger.debug("Denoising done")

    # 3. Contrast enhancement (CLAHE) - applied before binarization
    if enable_contrast_enhancement:
        result = enhance_contrast(result)
        logger.debug("Contrast enhancement done")

    # 4. Binarization
    result = binarize(result, method=binarization_method)
    logger.debug("Binarization done (method: %s)", binarization_method)

    # 5. Deskewing
    if enable_deskew:
        result = deskew(result)
        logger.debug("Deskewing done")

    # 6. Border removal
    if enable_border_removal:
        result = remove_borders(result)
        logger.debug("Border removal done")

    # 7. Resolution normalization
    result = normalize_resolution(result, target_dpi=target_dpi)
    logger.debug("Resolution normalization done")

    return result


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale. If already grayscale, return as-is."""
    if len(image.shape) == 2:
        return image
    if len(image.shape) == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Remove noise using Non-Local Means denoising.

    This works well for scanned documents where noise comes from
    the scanning process.
    """
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve contrast on faded documents.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def binarize(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Convert to binary image using the specified thresholding method.

    Args:
        image: Grayscale image.
        method: One of 'otsu', 'sauvola', 'adaptive'.

    Returns:
        Binary image (black text on white background).
    """
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == "adaptive":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "sauvola":
        # Sauvola thresholding approximation using adaptive method
        # with a larger block size suitable for document images
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8
        )
    else:
        logger.warning("Unknown binarization method '%s', falling back to otsu", method)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct document rotation using minAreaRect on contours.

    Handles small rotations typical of scanned documents (up to ~15 degrees).
    """
    # Find contours in a dilated version to get text block boundaries
    coords = np.column_stack(np.where(image < 128))

    if len(coords) < 50:
        # Not enough dark pixels to determine skew
        return image

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angles in [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct if angle is small (large angles likely mean
    # the document is intentionally rotated, not skewed)
    if abs(angle) > 15 or abs(angle) < 0.1:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    logger.info("Deskewed by %.2f degrees", angle)
    return rotated


def remove_borders(image: np.ndarray) -> np.ndarray:
    """
    Remove black scan borders by finding the largest contour
    that represents the document area.
    """
    # Invert so document area is white
    inverted = cv2.bitwise_not(image)

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Find the largest contour (should be the document)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Only crop if the contour is at least 50% of the image area
    img_area = image.shape[0] * image.shape[1]
    contour_area = w * h
    if contour_area < img_area * 0.5:
        return image

    # Add small padding
    pad = 5
    y_start = max(0, y - pad)
    y_end = min(image.shape[0], y + h + pad)
    x_start = max(0, x - pad)
    x_end = min(image.shape[1], x + w + pad)

    return image[y_start:y_end, x_start:x_end]


def normalize_resolution(image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
    """
    Upscale low-resolution images to meet target DPI.

    EasyOCR performs best at 300+ DPI. We estimate current DPI
    from image dimensions and upscale if needed.

    For images without embedded DPI info, we use a heuristic:
    if the image height is less than what a standard A4 page
    would be at target DPI, we upscale proportionally.
    """
    # A4 page at target DPI would be approximately:
    # 297mm * (target_dpi / 25.4) ~= target_dpi * 11.69 inches
    expected_height_at_target = int(target_dpi * 11.69)

    h, w = image.shape[:2]

    if h >= expected_height_at_target:
        # Already at or above target resolution
        return image

    # Calculate scale factor
    scale = expected_height_at_target / h

    # Cap upscaling at 3x to avoid excessive memory use
    scale = min(scale, 3.0)

    if scale <= 1.05:
        # Not worth upscaling for tiny improvements
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    logger.info("Upscaling image from %dx%d to %dx%d (%.1fx)", w, h, new_w, new_h, scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
