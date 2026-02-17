"""
Tests for the OCR pipeline orchestrator (Surya-based).

Uses mocking to avoid dependency on actual Surya models and image files.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from OCR.ocr_pipeline import _compute_document_confidence, process_batch, process_document
from OCR.schemas import OCRDocumentResult, OCRLine, OCRPageResult, OCRWord


@pytest.fixture
def sample_image_path():
    """Create a temporary test image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (400, 500), color=(255, 255, 255))
        # Draw some dark regions to simulate text
        pixels = img.load()
        for x in range(50, 350):
            for y in range(100, 120):
                pixels[x, y] = (0, 0, 0)
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_ocr_page_result():
    """Create a mock OCR page result."""
    words = [
        OCRWord(
            text="محكمة",
            bbox=[(0, 0), (100, 0), (100, 30), (0, 30)],
            confidence=0.95,
        ),
        OCRWord(
            text="ابتدائية",
            bbox=[(110, 0), (250, 0), (250, 30), (110, 30)],
            confidence=0.88,
        ),
    ]
    line = OCRLine(words=words, text="محكمة ابتدائية", confidence=0.92)
    return OCRPageResult(
        page_number=1,
        lines=[line],
        raw_text="محكمة ابتدائية",
        confidence=0.92,
    )


class TestProcessDocument:
    @patch("OCR.ocr_pipeline.run_ocr")
    @patch("OCR.ocr_pipeline.postprocess_page")
    @patch("OCR.ocr_pipeline.postprocess_document_pages")
    @patch("OCR.ocr_pipeline.preprocess_image")
    def test_returns_document_result(
        self, mock_preprocess, mock_doc_postprocess, mock_postprocess,
        mock_run_ocr, sample_image_path, mock_ocr_page_result
    ):
        mock_preprocess.return_value = Image.new("RGB", (100, 100))
        mock_run_ocr.return_value = [mock_ocr_page_result]
        mock_postprocess.return_value = mock_ocr_page_result
        mock_doc_postprocess.return_value = [mock_ocr_page_result]

        result = process_document(sample_image_path)

        assert isinstance(result, OCRDocumentResult)
        assert result.total_pages == 1
        assert "محكمة" in result.raw_text
        assert result.overall_confidence > 0

    @patch("OCR.ocr_pipeline.run_ocr")
    @patch("OCR.ocr_pipeline.postprocess_page")
    @patch("OCR.ocr_pipeline.postprocess_document_pages")
    @patch("OCR.ocr_pipeline.preprocess_image")
    def test_return_for_node0(
        self, mock_preprocess, mock_doc_postprocess, mock_postprocess,
        mock_run_ocr, sample_image_path, mock_ocr_page_result
    ):
        mock_preprocess.return_value = Image.new("RGB", (100, 100))
        mock_run_ocr.return_value = [mock_ocr_page_result]
        mock_postprocess.return_value = mock_ocr_page_result
        mock_doc_postprocess.return_value = [mock_ocr_page_result]

        result = process_document(sample_image_path, return_for_node0=True)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "raw_text" in result[0]
        assert "doc_id" in result[0]
        assert "محكمة" in result[0]["raw_text"]

    @patch("OCR.ocr_pipeline.run_ocr")
    @patch("OCR.ocr_pipeline.postprocess_page")
    @patch("OCR.ocr_pipeline.postprocess_document_pages")
    @patch("OCR.ocr_pipeline.preprocess_image")
    def test_custom_doc_id(
        self, mock_preprocess, mock_doc_postprocess, mock_postprocess,
        mock_run_ocr, sample_image_path, mock_ocr_page_result
    ):
        mock_preprocess.return_value = Image.new("RGB", (100, 100))
        mock_run_ocr.return_value = [mock_ocr_page_result]
        mock_postprocess.return_value = mock_ocr_page_result
        mock_doc_postprocess.return_value = [mock_ocr_page_result]

        result = process_document(sample_image_path, doc_id="test-123")

        assert result.doc_id == "test-123"

    @patch("OCR.ocr_pipeline.run_ocr")
    @patch("OCR.ocr_pipeline.postprocess_page")
    @patch("OCR.ocr_pipeline.postprocess_document_pages")
    @patch("OCR.ocr_pipeline.preprocess_image")
    def test_accepts_list_of_paths(
        self, mock_preprocess, mock_doc_postprocess, mock_postprocess,
        mock_run_ocr, sample_image_path, mock_ocr_page_result
    ):
        mock_preprocess.return_value = Image.new("RGB", (100, 100))
        mock_run_ocr.return_value = [mock_ocr_page_result]
        mock_postprocess.return_value = mock_ocr_page_result
        mock_doc_postprocess.return_value = [mock_ocr_page_result]

        result = process_document([sample_image_path])

        assert isinstance(result, OCRDocumentResult)
        assert result.total_pages == 1


class TestProcessBatch:
    @patch("OCR.ocr_pipeline.run_ocr")
    @patch("OCR.ocr_pipeline.postprocess_page")
    @patch("OCR.ocr_pipeline.postprocess_document_pages")
    @patch("OCR.ocr_pipeline.preprocess_image")
    def test_batch_processing(
        self, mock_preprocess, mock_doc_postprocess, mock_postprocess,
        mock_run_ocr, sample_image_path, mock_ocr_page_result
    ):
        mock_preprocess.return_value = Image.new("RGB", (100, 100))
        mock_run_ocr.return_value = [mock_ocr_page_result]
        mock_postprocess.return_value = mock_ocr_page_result
        mock_doc_postprocess.return_value = [mock_ocr_page_result]

        results = process_batch([sample_image_path])

        assert len(results) == 1
        assert isinstance(results[0], OCRDocumentResult)

    @patch("OCR.ocr_pipeline.run_ocr")
    @patch("OCR.ocr_pipeline.postprocess_page")
    @patch("OCR.ocr_pipeline.postprocess_document_pages")
    @patch("OCR.ocr_pipeline.preprocess_image")
    def test_batch_for_node0(
        self, mock_preprocess, mock_doc_postprocess, mock_postprocess,
        mock_run_ocr, sample_image_path, mock_ocr_page_result
    ):
        mock_preprocess.return_value = Image.new("RGB", (100, 100))
        mock_run_ocr.return_value = [mock_ocr_page_result]
        mock_postprocess.return_value = mock_ocr_page_result
        mock_doc_postprocess.return_value = [mock_ocr_page_result]

        results = process_batch(
            [sample_image_path], return_for_node0=True
        )

        assert isinstance(results, list)
        assert "raw_text" in results[0]


class TestComputeDocumentConfidence:
    def test_single_page(self):
        pages = [
            OCRPageResult(page_number=1, lines=[], raw_text="hello", confidence=0.9)
        ]
        assert _compute_document_confidence(pages) == pytest.approx(0.9)

    def test_multiple_pages_weighted(self):
        pages = [
            OCRPageResult(page_number=1, lines=[], raw_text="ab", confidence=1.0),
            OCRPageResult(page_number=2, lines=[], raw_text="abcdef", confidence=0.5),
        ]
        # 2*1.0 + 6*0.5 = 5.0 / 8 = 0.625
        assert _compute_document_confidence(pages) == pytest.approx(0.625)

    def test_empty_pages(self):
        assert _compute_document_confidence([]) == 0.0

    def test_pages_with_no_text(self):
        pages = [
            OCRPageResult(page_number=1, lines=[], raw_text="", confidence=0.0)
        ]
        assert _compute_document_confidence(pages) == 0.0
