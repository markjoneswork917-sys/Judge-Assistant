"""
schemas.py

Pydantic models for OCR results.

Follows the project's existing pattern from Summerize/schemas.py,
providing structured, validated output from the OCR pipeline.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class OCRWord(BaseModel):
    """Single recognized word with bounding box and confidence."""

    text: str = Field(..., description="Recognized text for this word")
    bbox: List[Tuple[float, float]] = Field(
        ..., description="Bounding box as list of (x, y) corner points"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Recognition confidence (0.0 to 1.0)"
    )


class OCRLine(BaseModel):
    """A line of text composed of one or more words."""

    words: List[OCRWord] = Field(default_factory=list, description="Words in the line")
    text: str = Field(..., description="Full line text")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregated line confidence"
    )


class OCRPageResult(BaseModel):
    """Full page OCR result containing lines, confidence, and warnings."""

    page_number: int = Field(..., ge=1, description="1-based page number")
    lines: List[OCRLine] = Field(default_factory=list, description="Recognized lines")
    raw_text: str = Field(default="", description="Full page text joined from lines")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall page confidence"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings for low-confidence regions"
    )
    has_errors: bool = Field(
        default=False, description="Whether OCR engine encountered errors"
    )


class OCRDocumentResult(BaseModel):
    """Multi-page document OCR result."""

    file_path: str = Field(..., description="Source file path")
    doc_id: str = Field(..., description="Unique document identifier")
    pages: List[OCRPageResult] = Field(
        default_factory=list, description="Per-page results"
    )
    raw_text: str = Field(
        default="", description="Combined text from all pages, ready for Node 0"
    )
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Document-level confidence"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Document-level warnings"
    )
