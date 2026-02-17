"""
Tests for the OCR engine module (Surya-based).

Note: Tests that require the actual Surya model are marked with
pytest.mark.integration and skipped by default. Unit tests use mocking.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from OCR.engine import (
    SuryaOCREngine,
    _compute_page_confidence,
    get_engine,
    reset_engine,
    run_ocr,
)
from OCR.schemas import OCRLine, OCRPageResult, OCRWord


@pytest.fixture(autouse=True)
def _reset():
    """Reset singleton engine before each test."""
    reset_engine()
    yield
    reset_engine()


def _make_pil_image(w=400, h=500):
    """Create a synthetic test PIL Image."""
    arr = np.ones((h, w, 3), dtype=np.uint8) * 255
    arr[100:120, 50:350] = 0  # Dark region simulating text
    return Image.fromarray(arr)


def _make_mock_det_result():
    """Create a mock detection result."""
    mock = MagicMock()
    mock_bbox = MagicMock()
    mock_bbox.bbox = [10, 10, 200, 40]
    mock.bboxes = [mock_bbox]
    return mock


def _make_mock_rec_result(text="نص تجريبي", confidence=0.92):
    """Create a mock recognition result."""
    mock = MagicMock()
    mock_text_line = MagicMock()
    mock_text_line.text = text
    mock_text_line.confidence = confidence
    mock_text_line.bbox = [10, 10, 200, 40]
    mock.text_lines = [mock_text_line]
    return mock


class TestSuryaOCREngine:
    def test_engine_initializes_unloaded(self):
        engine = SuryaOCREngine()
        assert engine._models_loaded is False
        assert engine._det_model is None
        assert engine._rec_model is None

    def test_reset_clears_models(self):
        engine = SuryaOCREngine()
        engine._models_loaded = True
        engine._det_model = "fake"
        engine.reset()
        assert engine._models_loaded is False
        assert engine._det_model is None

    @patch("OCR.engine.SuryaOCREngine._load_models")
    def test_process_calls_batch(self, mock_load):
        engine = SuryaOCREngine()
        engine._models_loaded = True

        expected_result = OCRPageResult(
            page_number=1,
            lines=[],
            raw_text="test",
            confidence=0.9,
        )

        with patch.object(engine, "_process_batch", return_value=[expected_result]) as mock_batch:
            # Mock the surya imports inside process()
            mock_det = MagicMock()
            mock_rec = MagicMock()
            import sys
            sys.modules["surya"] = MagicMock()
            sys.modules["surya.detection"] = MagicMock(batch_text_detection=mock_det)
            sys.modules["surya.recognition"] = MagicMock(batch_recognition=mock_rec)

            try:
                images = [_make_pil_image()]
                results = engine.process(images)

                mock_load.assert_called_once()
                mock_batch.assert_called_once()
                assert len(results) == 1
            finally:
                sys.modules.pop("surya", None)
                sys.modules.pop("surya.detection", None)
                sys.modules.pop("surya.recognition", None)


class TestRunOCR:
    @patch("OCR.engine.get_engine")
    def test_returns_page_results(self, mock_get_engine):
        mock_engine = MagicMock()
        page_result = OCRPageResult(
            page_number=1,
            lines=[
                OCRLine(
                    words=[
                        OCRWord(
                            text="نص تجريبي",
                            bbox=[(0, 0), (100, 0), (100, 30), (0, 30)],
                            confidence=0.92,
                        )
                    ],
                    text="نص تجريبي",
                    confidence=0.92,
                )
            ],
            raw_text="نص تجريبي",
            confidence=0.92,
        )
        mock_engine.process.return_value = [page_result]
        mock_get_engine.return_value = mock_engine

        images = [_make_pil_image()]
        results = run_ocr(images)

        assert len(results) == 1
        assert results[0].page_number == 1
        assert "نص تجريبي" in results[0].raw_text
        assert results[0].confidence > 0

    @patch("OCR.engine.get_engine")
    def test_handles_empty_results(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_engine.process.return_value = [
            OCRPageResult(
                page_number=1,
                lines=[],
                raw_text="",
                confidence=0.0,
                warnings=["No text detected on page"],
            )
        ]
        mock_get_engine.return_value = mock_engine

        images = [_make_pil_image()]
        results = run_ocr(images)

        assert len(results) == 1
        assert results[0].raw_text == ""
        assert results[0].confidence == 0.0

    @patch("OCR.engine.get_engine")
    def test_handles_engine_error(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_engine.process.return_value = [
            OCRPageResult(
                page_number=1,
                lines=[],
                raw_text="",
                confidence=0.0,
                warnings=["OCR engine error: test error"],
                has_errors=True,
            )
        ]
        mock_get_engine.return_value = mock_engine

        images = [_make_pil_image()]
        results = run_ocr(images)

        assert results[0].has_errors is True
        assert results[0].confidence == 0.0
        assert len(results[0].warnings) > 0

    @patch("OCR.engine.get_engine")
    def test_multiple_pages(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_engine.process.return_value = [
            OCRPageResult(page_number=1, lines=[], raw_text="page1", confidence=0.9),
            OCRPageResult(page_number=2, lines=[], raw_text="page2", confidence=0.85),
        ]
        mock_get_engine.return_value = mock_engine

        images = [_make_pil_image(), _make_pil_image()]
        results = run_ocr(images)

        assert len(results) == 2
        assert results[0].page_number == 1
        assert results[1].page_number == 2


class TestComputePageConfidence:
    def test_single_line(self):
        lines = [OCRLine(words=[], text="hello", confidence=0.8)]
        assert _compute_page_confidence(lines) == pytest.approx(0.8)

    def test_weighted_by_length(self):
        lines = [
            OCRLine(words=[], text="ab", confidence=1.0),
            OCRLine(words=[], text="abcdef", confidence=0.5),
        ]
        # 2*1.0 + 6*0.5 = 5.0 / 8 = 0.625
        assert _compute_page_confidence(lines) == pytest.approx(0.625)

    def test_empty(self):
        assert _compute_page_confidence([]) == 0.0

    def test_empty_text_lines(self):
        lines = [OCRLine(words=[], text="", confidence=0.9)]
        assert _compute_page_confidence(lines) == 0.0


class TestGetEngine:
    def test_returns_singleton(self):
        engine1 = get_engine()
        engine2 = get_engine()
        assert engine1 is engine2

    def test_reset_creates_new_instance(self):
        engine1 = get_engine()
        reset_engine()
        engine2 = get_engine()
        assert engine1 is not engine2
