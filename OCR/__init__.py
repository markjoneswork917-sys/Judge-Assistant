"""
OCR Module for Judge Assistant

Extracts Arabic text from document page images using Surya,
a state-of-the-art open-source ML-based document OCR system
with strong Arabic support.

Includes image preprocessing, confidence scoring, and
post-processing error correction with legal dictionary support.

Public API:
    process_document  - Process a single document (image or PDF)
    process_batch     - Process multiple documents
    OCRDocumentResult - Structured result model
    OCRPageResult     - Per-page result model
"""

from .ocr_pipeline import process_batch, process_document
from .schemas import OCRDocumentResult, OCRLine, OCRPageResult, OCRWord

__all__ = [
    "process_document",
    "process_batch",
    "OCRDocumentResult",
    "OCRPageResult",
    "OCRLine",
    "OCRWord",
]
