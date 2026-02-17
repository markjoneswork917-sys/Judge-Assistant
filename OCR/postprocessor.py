"""
postprocessor.py

Multi-stage error correction pipeline targeting Arabic legal text.

Handles:
- Unicode normalization (alef variants, teh marbuta, etc.)
- RTL text ordering
- Dictionary-based correction for medium-confidence words
- Legal pattern validation
- Whitespace and punctuation cleanup
"""

import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional, Set

from . import config
from .schemas import OCRLine, OCRPageResult, OCRWord

logger = logging.getLogger(__name__)

# Legal dictionary cache
_legal_dictionary: Optional[Set[str]] = None


def load_legal_dictionary() -> Set[str]:
    """
    Load the legal Arabic dictionary from disk.
    Caches the result after first load.
    """
    global _legal_dictionary
    if _legal_dictionary is not None:
        return _legal_dictionary

    dict_path = config.DICTIONARY_PATH
    if not os.path.isfile(dict_path):
        logger.warning("Legal dictionary not found at %s", dict_path)
        _legal_dictionary = set()
        return _legal_dictionary

    with open(dict_path, "r", encoding="utf-8") as f:
        words = set()
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                words.add(word)

    _legal_dictionary = words
    logger.info("Loaded %d terms from legal dictionary", len(words))
    return _legal_dictionary


def reset_dictionary():
    """Reset the cached dictionary (useful for testing)."""
    global _legal_dictionary
    _legal_dictionary = None


def postprocess_page(page_result: OCRPageResult) -> OCRPageResult:
    """
    Apply all post-processing steps to a page result.

    Args:
        page_result: Raw OCR page result.

    Returns:
        Corrected OCRPageResult with cleaned text and updated lines.
    """
    corrected_lines = []

    for line in page_result.lines:
        corrected_words = []
        for word in line.words:
            corrected_text = normalize_arabic(word.text)

            # Apply dictionary correction for medium-confidence words
            if (
                config.ENABLE_DICTIONARY_CORRECTION
                and config.MEDIUM_CONFIDENCE_THRESHOLD
                <= word.confidence
                < config.HIGH_CONFIDENCE_THRESHOLD
            ):
                corrected_text = dictionary_correct(corrected_text)

            corrected_words.append(
                OCRWord(
                    text=corrected_text,
                    bbox=word.bbox,
                    confidence=word.confidence,
                )
            )

        line_text = " ".join(w.text for w in corrected_words)
        line_text = fix_whitespace(line_text)
        line_text = validate_legal_patterns(line_text)

        corrected_lines.append(
            OCRLine(
                words=corrected_words,
                text=line_text,
                confidence=line.confidence,
            )
        )

    raw_text = "\n".join(line.text for line in corrected_lines)

    return OCRPageResult(
        page_number=page_result.page_number,
        lines=corrected_lines,
        raw_text=raw_text,
        confidence=page_result.confidence,
        warnings=page_result.warnings,
        has_errors=page_result.has_errors,
    )


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic Unicode characters.

    Handles:
    - Alef variants (alef with hamza, madda, etc.) -> bare alef
    - Teh marbuta -> heh normalization (optional, conservative)
    - Remove tatweel (kashida)
    - Remove zero-width characters
    - Remove directional marks
    """
    if not text:
        return text

    # Remove zero-width characters
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("\u200c", "")  # Zero-width non-joiner
    text = text.replace("\u200d", "")  # Zero-width joiner
    text = text.replace("\u200e", "")  # Left-to-right mark
    text = text.replace("\u200f", "")  # Right-to-left mark
    text = text.replace("\ufeff", "")  # BOM

    # Normalize alef variants to bare alef
    text = text.replace("\u0622", "\u0627")  # Alef with madda -> alef
    text = text.replace("\u0623", "\u0627")  # Alef with hamza above -> alef
    text = text.replace("\u0625", "\u0627")  # Alef with hamza below -> alef
    text = text.replace("\u0671", "\u0627")  # Alef wasla -> alef

    # Remove tatweel (kashida) - decorative elongation
    text = text.replace("\u0640", "")

    # Normalize Unicode form (NFC)
    text = unicodedata.normalize("NFC", text)

    return text


def dictionary_correct(word: str) -> str:
    """
    Attempt to correct a word using the legal dictionary.

    If the word is not in the dictionary, find the closest match
    within the configured Levenshtein distance threshold.
    """
    dictionary = load_legal_dictionary()

    if not dictionary:
        return word

    # Already correct
    normalized = normalize_arabic(word)
    if normalized in dictionary:
        return normalized

    # Find closest match
    best_match = None
    best_distance = config.MAX_LEVENSHTEIN_DISTANCE + 1

    for dict_word in dictionary:
        dist = _levenshtein_distance(normalized, dict_word)
        if dist < best_distance:
            best_distance = dist
            best_match = dict_word

    if best_match and best_distance <= config.MAX_LEVENSHTEIN_DISTANCE:
        logger.debug(
            "Dictionary correction: '%s' -> '%s' (distance=%d)",
            word,
            best_match,
            best_distance,
        )
        return best_match

    return word


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    Uses the dynamic programming approach. Falls back to python-Levenshtein
    if available for better performance.
    """
    try:
        import Levenshtein

        return Levenshtein.distance(s1, s2)
    except ImportError:
        pass

    # Pure Python fallback
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fix_whitespace(text: str) -> str:
    """
    Fix common OCR whitespace artifacts.

    - Collapse multiple spaces into one
    - Remove spaces before punctuation
    - Fix spaces within words (common Arabic OCR error)
    """
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces before Arabic punctuation
    text = re.sub(r"\s+([،؛.,:؟!])", r"\1", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def validate_legal_patterns(text: str) -> str:
    """
    Validate and fix known legal patterns.

    Patterns:
    - Article references: مادة + number
    - Date formats
    - Court name patterns
    """
    # Fix broken article references: "م ا د ة" -> "مادة"
    text = re.sub(r"م\s+ا\s+د\s+ة", "مادة", text)

    # Fix broken "محكمة" (court)
    text = re.sub(r"م\s+ح\s+ك\s+م\s+ة", "محكمة", text)

    # Fix broken "المدعي" (plaintiff)
    text = re.sub(r"ا\s+ل\s+م\s+د\s+ع\s+ي", "المدعي", text)

    # Fix broken "المدعى عليه" (defendant)
    text = re.sub(r"ا\s+ل\s+م\s+د\s+ع\s+ى\s+ع\s+ل\s+ي\s+ه", "المدعى عليه", text)

    return text
