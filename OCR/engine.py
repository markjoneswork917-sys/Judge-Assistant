"""
engine.py

EasyOCR wrapper with confidence scoring.

Provides a singleton reader instance for efficient reuse across
multiple calls, maps EasyOCR output into structured schema objects,
and computes word, line, and page-level confidence scores.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from . import config
from .schemas import OCRLine, OCRPageResult, OCRWord

logger = logging.getLogger(__name__)

# Singleton reader instance
_reader = None


def _get_reader():
    """
    Lazy-initialize the EasyOCR reader singleton.

    Creates the reader on first call and reuses it for all subsequent calls.
    Auto-detects GPU availability, falling back to CPU.
    """
    global _reader
    if _reader is not None:
        return _reader

    try:
        import easyocr
    except ImportError:
        raise ImportError(
            "easyocr is required. Install it with: pip install easyocr"
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

    logger.info(
        "Initializing EasyOCR reader (language=%s, gpu=%s)",
        config.OCR_LANGUAGE,
        use_gpu,
    )
    _reader = easyocr.Reader(
        lang_list=[config.OCR_LANGUAGE],
        gpu=use_gpu,
    )
    return _reader


def reset_reader():
    """Reset the singleton reader (useful for testing)."""
    global _reader
    _reader = None


def run_ocr(image: np.ndarray, page_number: int = 1) -> OCRPageResult:
    """
    Run EasyOCR on a single preprocessed image and return structured results.

    Args:
        image: Preprocessed image as numpy array (grayscale or BGR).
        page_number: 1-based page number for the result.

    Returns:
        OCRPageResult with lines, confidence scores, and any warnings.
    """
    warnings: List[str] = []

    try:
        reader = _get_reader()
        raw_results = reader.readtext(image, detail=1)
    except Exception as e:
        logger.error("EasyOCR engine error: %s", e)
        return OCRPageResult(
            page_number=page_number,
            lines=[],
            raw_text="",
            confidence=0.0,
            warnings=[f"OCR engine error: {str(e)}"],
            has_errors=True,
        )

    if not raw_results:
        return OCRPageResult(
            page_number=page_number,
            lines=[],
            raw_text="",
            confidence=0.0,
            warnings=["No text detected on page"],
            has_errors=False,
        )

    # Convert raw results to OCRWord objects
    words = _parse_raw_results(raw_results)

    # Group words into lines based on vertical position
    lines = _group_words_into_lines(words)

    # Flag low-confidence words
    for line in lines:
        for word in line.words:
            if word.confidence < config.MEDIUM_CONFIDENCE_THRESHOLD:
                warnings.append(
                    f"Low confidence ({word.confidence:.2f}) for: '{word.text}'"
                )

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


def _parse_raw_results(
    raw_results: List[Tuple],
) -> List[OCRWord]:
    """
    Parse EasyOCR raw output into OCRWord objects.

    EasyOCR returns: [(bbox, text, confidence), ...]
    where bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    words = []
    for bbox, text, conf in raw_results:
        if not text.strip():
            continue
        words.append(
            OCRWord(
                text=text.strip(),
                bbox=[(float(p[0]), float(p[1])) for p in bbox],
                confidence=float(conf),
            )
        )
    return words


def _group_words_into_lines(words: List[OCRWord]) -> List[OCRLine]:
    """
    Group words into lines based on vertical proximity of bounding boxes.

    Words whose vertical centers are within a threshold are considered
    to be on the same line. Lines are sorted top-to-bottom, and words
    within each line are sorted right-to-left (for Arabic RTL text).
    """
    if not words:
        return []

    # Compute vertical center for each word
    def v_center(word: OCRWord) -> float:
        ys = [p[1] for p in word.bbox]
        return sum(ys) / len(ys)

    def h_center(word: OCRWord) -> float:
        xs = [p[0] for p in word.bbox]
        return sum(xs) / len(xs)

    def word_height(word: OCRWord) -> float:
        ys = [p[1] for p in word.bbox]
        return max(ys) - min(ys)

    # Sort by vertical position first
    sorted_words = sorted(words, key=v_center)

    # Group into lines using vertical proximity
    lines_groups: List[List[OCRWord]] = []
    current_group: List[OCRWord] = [sorted_words[0]]

    for word in sorted_words[1:]:
        prev_center = v_center(current_group[-1])
        curr_center = v_center(word)
        threshold = max(word_height(word), word_height(current_group[-1])) * 0.5

        if abs(curr_center - prev_center) <= threshold:
            current_group.append(word)
        else:
            lines_groups.append(current_group)
            current_group = [word]

    lines_groups.append(current_group)

    # Build OCRLine objects
    lines = []
    for group in lines_groups:
        # Sort words right-to-left for Arabic
        group.sort(key=h_center, reverse=True)

        line_text = " ".join(w.text for w in group)
        line_confidence = _compute_weighted_confidence(group)

        lines.append(
            OCRLine(
                words=group,
                text=line_text,
                confidence=line_confidence,
            )
        )

    return lines


def _compute_weighted_confidence(words: List[OCRWord]) -> float:
    """
    Compute weighted average confidence for a list of words.
    Weight is proportional to word length (longer words contribute more).
    """
    if not words:
        return 0.0

    total_weight = sum(len(w.text) for w in words)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(w.confidence * len(w.text) for w in words)
    return weighted_sum / total_weight


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
