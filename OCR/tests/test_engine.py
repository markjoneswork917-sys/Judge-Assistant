"""
Tests for the OCR engine module.

Note: Tests that require the actual EasyOCR model are marked with
pytest.mark.integration and skipped by default. Unit tests use mocking.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from OCR.engine import (
    _compute_page_confidence,
    _compute_weighted_confidence,
    _group_words_into_lines,
    _parse_raw_results,
    reset_reader,
    run_ocr,
)
from OCR.schemas import OCRLine, OCRWord


@pytest.fixture(autouse=True)
def _reset():
    """Reset singleton reader before each test."""
    reset_reader()
    yield
    reset_reader()


class TestParseRawResults:
    def test_parses_easyocr_output(self):
        raw = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "مرحبا", 0.95),
            ([[0, 40], [80, 40], [80, 70], [0, 70]], "عالم", 0.88),
        ]
        words = _parse_raw_results(raw)
        assert len(words) == 2
        assert words[0].text == "مرحبا"
        assert words[0].confidence == 0.95
        assert words[1].text == "عالم"

    def test_skips_empty_text(self):
        raw = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "   ", 0.5),
            ([[0, 40], [80, 40], [80, 70], [0, 70]], "نص", 0.9),
        ]
        words = _parse_raw_results(raw)
        assert len(words) == 1
        assert words[0].text == "نص"

    def test_empty_input(self):
        assert _parse_raw_results([]) == []


class TestGroupWordsIntoLines:
    def test_groups_nearby_words(self):
        words = [
            OCRWord(text="كلمة", bbox=[(200, 10), (250, 10), (250, 30), (200, 30)], confidence=0.9),
            OCRWord(text="أخرى", bbox=[(100, 12), (150, 12), (150, 32), (100, 32)], confidence=0.85),
        ]
        lines = _group_words_into_lines(words)
        assert len(lines) == 1
        assert "كلمة" in lines[0].text
        assert "أخرى" in lines[0].text

    def test_separates_distant_words(self):
        words = [
            OCRWord(text="سطر1", bbox=[(100, 10), (200, 10), (200, 30), (100, 30)], confidence=0.9),
            OCRWord(text="سطر2", bbox=[(100, 100), (200, 100), (200, 120), (100, 120)], confidence=0.8),
        ]
        lines = _group_words_into_lines(words)
        assert len(lines) == 2

    def test_empty_words(self):
        assert _group_words_into_lines([]) == []

    def test_rtl_ordering(self):
        """Words should be sorted right-to-left within a line."""
        words = [
            OCRWord(text="يسار", bbox=[(50, 10), (100, 10), (100, 30), (50, 30)], confidence=0.9),
            OCRWord(text="يمين", bbox=[(200, 10), (250, 10), (250, 30), (200, 30)], confidence=0.9),
        ]
        lines = _group_words_into_lines(words)
        assert len(lines) == 1
        # "يمين" (right) should come first in RTL
        assert lines[0].text.startswith("يمين")


class TestComputeWeightedConfidence:
    def test_single_word(self):
        words = [OCRWord(text="test", bbox=[(0, 0), (1, 0), (1, 1), (0, 1)], confidence=0.9)]
        assert _compute_weighted_confidence(words) == pytest.approx(0.9)

    def test_weighted_by_length(self):
        words = [
            OCRWord(text="ab", bbox=[(0, 0), (1, 0), (1, 1), (0, 1)], confidence=1.0),
            OCRWord(text="abcdef", bbox=[(0, 0), (1, 0), (1, 1), (0, 1)], confidence=0.5),
        ]
        # 2*1.0 + 6*0.5 = 5.0 / 8 = 0.625
        assert _compute_weighted_confidence(words) == pytest.approx(0.625)

    def test_empty(self):
        assert _compute_weighted_confidence([]) == 0.0


class TestComputePageConfidence:
    def test_single_line(self):
        lines = [OCRLine(words=[], text="hello", confidence=0.8)]
        assert _compute_page_confidence(lines) == pytest.approx(0.8)

    def test_empty(self):
        assert _compute_page_confidence([]) == 0.0


class TestRunOCR:
    @patch("OCR.engine._get_reader")
    def test_returns_page_result(self, mock_get_reader):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "نص تجريبي", 0.92),
        ]
        mock_get_reader.return_value = mock_reader

        img = np.ones((100, 100), dtype=np.uint8) * 255
        result = run_ocr(img, page_number=1)

        assert result.page_number == 1
        assert len(result.lines) == 1
        assert "نص تجريبي" in result.raw_text
        assert result.confidence > 0

    @patch("OCR.engine._get_reader")
    def test_handles_empty_results(self, mock_get_reader):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_get_reader.return_value = mock_reader

        img = np.ones((100, 100), dtype=np.uint8) * 255
        result = run_ocr(img, page_number=1)

        assert result.page_number == 1
        assert len(result.lines) == 0
        assert result.raw_text == ""
        assert result.confidence == 0.0

    @patch("OCR.engine._get_reader")
    def test_handles_engine_error(self, mock_get_reader):
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError("Engine failure")
        mock_get_reader.return_value = mock_reader

        img = np.ones((100, 100), dtype=np.uint8) * 255
        result = run_ocr(img, page_number=1)

        assert result.has_errors is True
        assert result.confidence == 0.0
        assert len(result.warnings) > 0
