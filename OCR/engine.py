"""
engine.py

Surya OCR engine wrapper with confidence scoring.

Uses Surya's two-stage pipeline:
1. Text Detection — finds text line bounding boxes with reading order
2. Text Recognition — recognizes characters with per-line confidence

Provides lazy model loading, GPU auto-detection, and batched inference
for efficient processing of multi-page documents.
"""

import logging
from typing import List, Optional

from PIL import Image

from . import config
from .schemas import OCRLine, OCRPageResult, OCRWord

logger = logging.getLogger(__name__)


class SuryaOCREngine:
    """
    Wrapper around Surya's detection and recognition models.

    Models are loaded lazily on first use and cached for reuse.
    GPU is auto-detected with fallback to CPU.
    """

    def __init__(self):
        self._det_model = None
        self._rec_model = None
        self._det_processor = None
        self._rec_processor = None
        self._models_loaded = False

    def _load_models(self) -> None:
        """
        Lazy-load Surya detection and recognition models.

        Called automatically on first inference. Models are cached
        for all subsequent calls.
        """
        if self._models_loaded:
            return

        try:
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.detection.processor import (
                load_processor as load_det_processor,
            )
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import (
                load_processor as load_rec_processor,
            )
        except ImportError:
            raise ImportError(
                "surya-ocr is required. Install it with: pip install surya-ocr"
            )

        use_gpu = config.USE_GPU
        try:
            import torch

            if use_gpu and not torch.cuda.is_available():
                logger.info("CUDA not available, falling back to CPU")
                use_gpu = False
        except ImportError:
            logger.info("PyTorch not found, using CPU mode")
            use_gpu = False

        logger.info("Loading Surya models (gpu=%s)...", use_gpu)

        self._det_processor = load_det_processor()
        self._det_model = load_det_model()
        self._rec_processor = load_rec_processor()
        self._rec_model = load_rec_model()

        if not use_gpu:
            self._det_model = self._det_model.cpu()
            self._rec_model = self._rec_model.cpu()

        self._models_loaded = True
        logger.info("Surya models loaded successfully")

    def process(self, images: List[Image.Image]) -> List[OCRPageResult]:
        """
        Run the full Surya pipeline (detect + recognize) on a list of images.

        Args:
            images: List of PIL Images in RGB mode.

        Returns:
            List of OCRPageResult, one per input image.
        """
        self._load_models()

        try:
            from surya.detection import batch_text_detection
            from surya.recognition import batch_recognition
        except ImportError:
            raise ImportError(
                "surya-ocr is required. Install it with: pip install surya-ocr"
            )

        results = []
        batch_size = config.SURYA_BATCH_SIZE

        # Process in batches to manage memory
        for batch_start in range(0, len(images), batch_size):
            batch_images = images[batch_start : batch_start + batch_size]
            batch_results = self._process_batch(
                batch_images,
                batch_text_detection,
                batch_recognition,
                page_offset=batch_start,
            )
            results.extend(batch_results)

        return results

    def _process_batch(
        self,
        images: List[Image.Image],
        batch_text_detection,
        batch_recognition,
        page_offset: int = 0,
    ) -> List[OCRPageResult]:
        """Process a single batch of images through detection and recognition."""
        page_results = []

        for i, image in enumerate(images):
            page_num = page_offset + i + 1

            try:
                page_result = self._process_single_image(
                    image, batch_text_detection, batch_recognition, page_num
                )
            except Exception as e:
                logger.error("Surya engine error on page %d: %s", page_num, e)
                page_result = OCRPageResult(
                    page_number=page_num,
                    lines=[],
                    raw_text="",
                    confidence=0.0,
                    warnings=[f"OCR engine error: {str(e)}"],
                    has_errors=True,
                )

            page_results.append(page_result)

        return page_results

    def _process_single_image(
        self,
        image: Image.Image,
        batch_text_detection,
        batch_recognition,
        page_number: int,
    ) -> OCRPageResult:
        """Process a single image through Surya detection and recognition."""
        warnings: List[str] = []

        # Stage 1: Text Detection
        det_results = batch_text_detection(
            [image], self._det_model, self._det_processor
        )

        if not det_results or not det_results[0].bboxes:
            return OCRPageResult(
                page_number=page_number,
                lines=[],
                raw_text="",
                confidence=0.0,
                warnings=["No text detected on page"],
                has_errors=False,
            )

        # Stage 2: Text Recognition
        langs = [[config.OCR_LANGUAGE]] * 1  # One language list per image
        rec_results = batch_recognition(
            [image], langs, self._rec_model, self._rec_processor, det_results
        )

        if not rec_results or not rec_results[0].text_lines:
            return OCRPageResult(
                page_number=page_number,
                lines=[],
                raw_text="",
                confidence=0.0,
                warnings=["Recognition produced no text"],
                has_errors=False,
            )

        # Convert Surya results to our schema
        rec_result = rec_results[0]
        lines = []

        for text_line in rec_result.text_lines:
            text = text_line.text.strip()
            if not text:
                continue

            confidence = getattr(text_line, "confidence", 0.0)
            bbox = getattr(text_line, "bbox", [0, 0, 0, 0])

            # Convert bbox [x1, y1, x2, y2] to corner points format
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                corner_points = [
                    (float(x1), float(y1)),
                    (float(x2), float(y1)),
                    (float(x2), float(y2)),
                    (float(x1), float(y2)),
                ]
            else:
                corner_points = [(0.0, 0.0)] * 4

            # Create a single word per line (Surya returns line-level text)
            word = OCRWord(
                text=text,
                bbox=corner_points,
                confidence=float(confidence),
            )

            line = OCRLine(
                words=[word],
                text=text,
                confidence=float(confidence),
            )

            # Flag low-confidence lines
            if confidence < config.MEDIUM_CONFIDENCE_THRESHOLD:
                warnings.append(
                    f"Low confidence ({confidence:.2f}) for: '{text[:50]}'"
                )

            lines.append(line)

        # Compute page-level confidence
        page_confidence = _compute_page_confidence(lines)

        # Build raw text
        raw_text = "\n".join(line.text for line in lines)

        return OCRPageResult(
            page_number=page_number,
            lines=lines,
            raw_text=raw_text,
            confidence=page_confidence,
            warnings=warnings,
            has_errors=False,
        )

    def reset(self) -> None:
        """Release models and free memory."""
        self._det_model = None
        self._rec_model = None
        self._det_processor = None
        self._rec_processor = None
        self._models_loaded = False
        logger.info("Surya models released")


# Module-level singleton engine
_engine: Optional[SuryaOCREngine] = None


def get_engine() -> SuryaOCREngine:
    """Get or create the singleton Surya OCR engine."""
    global _engine
    if _engine is None:
        _engine = SuryaOCREngine()
    return _engine


def reset_engine() -> None:
    """Reset the singleton engine (useful for testing)."""
    global _engine
    if _engine is not None:
        _engine.reset()
    _engine = None


def run_ocr(images: List[Image.Image]) -> List[OCRPageResult]:
    """
    Run Surya OCR on a list of PIL Images.

    This is the main entry point for the engine module.
    Uses the singleton engine with lazy model loading.

    Args:
        images: List of PIL Images in RGB mode.

    Returns:
        List of OCRPageResult, one per input image.
    """
    engine = get_engine()
    return engine.process(images)


def _compute_page_confidence(lines: List[OCRLine]) -> float:
    """
    Compute page-level confidence as a weighted average of line confidences.
    Weight is proportional to the number of characters in each line.
    """
    if not lines:
        return 0.0

    total_chars = sum(len(line.text) for line in lines)
    if total_chars == 0:
        return 0.0

    weighted_sum = sum(line.confidence * len(line.text) for line in lines)
    return weighted_sum / total_chars
