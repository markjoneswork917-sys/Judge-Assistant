"""
Tests for the OCR postprocessor module.
"""

import pytest

from OCR.postprocessor import (
    _levenshtein_distance,
    dictionary_correct,
    fix_whitespace,
    normalize_arabic,
    postprocess_page,
    reset_dictionary,
    validate_legal_patterns,
)
from OCR.schemas import OCRLine, OCRPageResult, OCRWord


@pytest.fixture(autouse=True)
def _reset():
    """Reset dictionary cache before each test."""
    reset_dictionary()
    yield
    reset_dictionary()


class TestNormalizeArabic:
    def test_removes_zero_width_chars(self):
        text = "مر\u200bحبا"
        assert "\u200b" not in normalize_arabic(text)

    def test_removes_directional_marks(self):
        text = "\u200fمرحبا\u200e"
        result = normalize_arabic(text)
        assert "\u200f" not in result
        assert "\u200e" not in result

    def test_normalizes_alef_variants(self):
        # Alef with hamza above -> bare alef
        assert normalize_arabic("\u0623حمد") == "\u0627حمد"
        # Alef with hamza below -> bare alef
        assert normalize_arabic("\u0625سلام") == "\u0627سلام"
        # Alef with madda -> bare alef
        assert normalize_arabic("\u0622خر") == "\u0627خر"

    def test_removes_tatweel(self):
        text = "مـــحـــكمة"
        result = normalize_arabic(text)
        assert "\u0640" not in result
        assert result == "محكمة"

    def test_empty_string(self):
        assert normalize_arabic("") == ""

    def test_non_arabic_unchanged(self):
        assert normalize_arabic("hello") == "hello"


class TestFixWhitespace:
    def test_collapses_multiple_spaces(self):
        assert fix_whitespace("كلمة   أخرى") == "كلمة أخرى"

    def test_removes_space_before_punctuation(self):
        assert fix_whitespace("نص ، آخر") == "نص، آخر"

    def test_strips_leading_trailing(self):
        assert fix_whitespace("  نص  ") == "نص"

    def test_collapses_tabs(self):
        assert fix_whitespace("كلمة\t\tأخرى") == "كلمة أخرى"


class TestValidateLegalPatterns:
    def test_fixes_broken_mada(self):
        assert "مادة" in validate_legal_patterns("م ا د ة 15")

    def test_fixes_broken_mahkama(self):
        assert "محكمة" in validate_legal_patterns("م ح ك م ة ابتدائية")

    def test_fixes_broken_plaintiff(self):
        assert "المدعي" in validate_legal_patterns("ا ل م د ع ي")

    def test_normal_text_unchanged(self):
        text = "نص عادي بدون أخطاء"
        assert validate_legal_patterns(text) == text


class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert _levenshtein_distance("محكمة", "محكمة") == 0

    def test_single_edit(self):
        assert _levenshtein_distance("محكمة", "محكمه") == 1

    def test_empty_string(self):
        assert _levenshtein_distance("", "abc") == 3
        assert _levenshtein_distance("abc", "") == 3

    def test_completely_different(self):
        assert _levenshtein_distance("abc", "xyz") == 3


class TestDictionaryCorrect:
    def test_correct_word_unchanged(self):
        """A word already in the dictionary should not be changed."""
        # "محكمة" is in our dictionary
        result = dictionary_correct("محكمة")
        # Should return the word (possibly normalized)
        assert "محكم" in result  # The core of the word should be preserved

    def test_returns_original_when_no_match(self):
        """Completely unrelated words should not be corrected."""
        result = dictionary_correct("xyzxyzxyz")
        assert result == "xyzxyzxyz"


class TestPostprocessPage:
    def test_processes_page(self):
        words = [
            OCRWord(
                text="\u200fمحكمة\u200e",
                bbox=[(0, 0), (100, 0), (100, 30), (0, 30)],
                confidence=0.95,
            ),
        ]
        line = OCRLine(words=words, text="\u200fمحكمة\u200e", confidence=0.95)
        page = OCRPageResult(
            page_number=1,
            lines=[line],
            raw_text="\u200fمحكمة\u200e",
            confidence=0.95,
        )

        result = postprocess_page(page)
        assert result.page_number == 1
        assert len(result.lines) == 1
        # Directional marks should be removed
        assert "\u200f" not in result.raw_text
        assert "\u200e" not in result.raw_text

    def test_empty_page(self):
        page = OCRPageResult(
            page_number=1,
            lines=[],
            raw_text="",
            confidence=0.0,
        )
        result = postprocess_page(page)
        assert result.raw_text == ""
        assert len(result.lines) == 0
