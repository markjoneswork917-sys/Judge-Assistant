"""
Tests for the OCR postprocessor module.
"""

import pytest

from OCR.postprocessor import (
    _levenshtein_distance,
    dictionary_correct,
    fix_intra_word_spaces,
    fix_whitespace,
    merge_split_lines,
    normalize_arabic,
    normalize_digits,
    postprocess_document_pages,
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


class TestNormalizeDigits:
    def test_western_to_arabic_indic(self):
        from OCR import config
        original = config.NORMALIZE_DIGITS
        try:
            config.NORMALIZE_DIGITS = "arabic_indic"
            assert normalize_digits("123") == "\u0661\u0662\u0663"
        finally:
            config.NORMALIZE_DIGITS = original

    def test_arabic_indic_to_western(self):
        from OCR import config
        original = config.NORMALIZE_DIGITS
        try:
            config.NORMALIZE_DIGITS = "western"
            assert normalize_digits("\u0661\u0662\u0663") == "123"
        finally:
            config.NORMALIZE_DIGITS = original

    def test_preserve_mode(self):
        from OCR import config
        original = config.NORMALIZE_DIGITS
        try:
            config.NORMALIZE_DIGITS = "preserve"
            assert normalize_digits("123") == "123"
            assert normalize_digits("\u0661\u0662\u0663") == "\u0661\u0662\u0663"
        finally:
            config.NORMALIZE_DIGITS = original


class TestFixWhitespace:
    def test_collapses_multiple_spaces(self):
        assert fix_whitespace("كلمة   أخرى") == "كلمة أخرى"

    def test_removes_space_before_punctuation(self):
        assert fix_whitespace("نص ، آخر") == "نص، آخر"

    def test_strips_leading_trailing(self):
        assert fix_whitespace("  نص  ") == "نص"

    def test_collapses_tabs(self):
        assert fix_whitespace("كلمة\t\tأخرى") == "كلمة أخرى"


class TestFixIntraWordSpaces:
    def test_merges_spaced_letters(self):
        """Should merge single Arabic letters separated by spaces."""
        # Pattern of 3+ single chars with spaces
        result = fix_intra_word_spaces("م ح ك م ة")
        assert result == "محكمة"

    def test_leaves_normal_text(self):
        text = "كلمة أخرى"
        assert fix_intra_word_spaces(text) == text


class TestMergeSplitLines:
    def test_single_line_unchanged(self):
        lines = [OCRLine(words=[], text="test", confidence=0.9)]
        result = merge_split_lines(lines)
        assert len(result) == 1

    def test_empty_list(self):
        assert merge_split_lines([]) == []

    def test_non_connecting_lines_not_merged(self):
        lines = [
            OCRLine(words=[], text="سطر أول.", confidence=0.9),
            OCRLine(words=[], text="سطر ثاني", confidence=0.85),
        ]
        result = merge_split_lines(lines)
        assert len(result) == 2


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
        result = dictionary_correct("محكمة")
        assert "محكم" in result

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


class TestPostprocessDocumentPages:
    def test_removes_repeated_headers(self):
        """Lines repeated on >50% of pages should be removed."""
        pages = []
        for i in range(6):
            lines = [
                OCRLine(words=[], text="رقم الصفحة", confidence=0.9),
                OCRLine(words=[], text=f"محتوى الصفحة {i}", confidence=0.9),
                OCRLine(words=[], text="تذييل الصفحة", confidence=0.9),
            ]
            pages.append(
                OCRPageResult(
                    page_number=i + 1,
                    lines=lines,
                    raw_text="\n".join(l.text for l in lines),
                    confidence=0.9,
                )
            )

        result = postprocess_document_pages(pages)

        # The repeated header/footer should be removed
        for page in result:
            assert page.lines[0].text != "رقم الصفحة"
            assert page.lines[-1].text != "تذييل الصفحة"

    def test_few_pages_unchanged(self):
        """With fewer than 3 pages, no header/footer removal."""
        pages = [
            OCRPageResult(page_number=1, lines=[], raw_text="test", confidence=0.9),
            OCRPageResult(page_number=2, lines=[], raw_text="test", confidence=0.9),
        ]
        result = postprocess_document_pages(pages)
        assert len(result) == 2
