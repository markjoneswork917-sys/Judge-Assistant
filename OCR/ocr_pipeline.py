"""
ocr_pipeline.py

Main orchestrator for the OCR module.

Coordinates the full pipeline: loading -> preprocessing -> OCR -> post-processing.
Supports single-file and batch processing, with an option to return results
formatted for the Summarization pipeline's Node 0.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

import numpy as np

from . import config
from .engine import run_ocr
from .postprocessor import postprocess_page
from .preprocessor import preprocess_image
from .schemas import OCRDocumentResult, OCRPageResult
from .utils import load_images

logger = logging.getLogger(__name__)


def process_document(
    file_path: str,
    return_for_node0: bool = False,
    doc_id: Optional[str] = None,
) -> Union[OCRDocumentResult, List[Dict]]:
    """
    Process a document through the full OCR pipeline.

    Args:
        file_path: Path to an image or PDF file.
        return_for_node0: If True, return in the format expected by
            SummarizationState.documents: [{"raw_text": "...", "doc_id": "..."}]
        doc_id: Optional document identifier. Auto-generated if not provided.

    Returns:
        OCRDocumentResult with structured results, or a list of dicts
        for Node 0 integration if return_for_node0 is True.
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())[:8]

    logger.info("Processing document: %s (doc_id=%s)", file_path, doc_id)

    # 1. Load images
    images = load_images(file_path)
    logger.info("Loaded %d page(s)", len(images))

    # 2. Process each page
    page_results: List[OCRPageResult] = []
    all_warnings: List[str] = []

    for i, image in enumerate(images):
        page_num = i + 1
        logger.info("Processing page %d/%d", page_num, len(images))

        # Preprocess
        preprocessed = preprocess_image(image)

        # Run OCR
        page_result = run_ocr(preprocessed, page_number=page_num)

        # Post-process
        page_result = postprocess_page(page_result)

        page_results.append(page_result)
        all_warnings.extend(page_result.warnings)

        # Release image memory
        del image
        del preprocessed

    # 3. Combine results
    combined_text = "\n\n".join(
        page.raw_text for page in page_results if page.raw_text
    )

    overall_confidence = _compute_document_confidence(page_results)

    result = OCRDocumentResult(
        file_path=str(file_path),
        doc_id=doc_id,
        pages=page_results,
        raw_text=combined_text,
        total_pages=len(page_results),
        overall_confidence=overall_confidence,
        warnings=all_warnings,
    )

    logger.info(
        "Document processed: %d pages, confidence=%.2f, warnings=%d",
        result.total_pages,
        result.overall_confidence,
        len(result.warnings),
    )

    if return_for_node0:
        return [{"raw_text": result.raw_text, "doc_id": result.doc_id}]

    return result


def process_batch(
    file_paths: List[str],
    return_for_node0: bool = False,
    max_workers: Optional[int] = None,
) -> Union[List[OCRDocumentResult], List[Dict]]:
    """
    Process multiple documents with optional concurrency.

    Args:
        file_paths: List of file paths to process.
        return_for_node0: If True, return in Node 0 format.
        max_workers: Number of concurrent workers. Defaults to config.BATCH_WORKERS.

    Returns:
        List of OCRDocumentResult objects, or list of dicts for Node 0.
    """
    if max_workers is None:
        max_workers = config.BATCH_WORKERS

    results: List[OCRDocumentResult] = []
    node0_results: List[Dict] = []

    # For single file or small batches, process sequentially
    if len(file_paths) <= 1 or max_workers <= 1:
        for fp in file_paths:
            result = process_document(fp, return_for_node0=False)
            results.append(result)
    else:
        # Use thread pool for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(process_document, fp, False): fp
                for fp in file_paths
            }

            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error("Failed to process %s: %s", fp, e)
                    # Create an error result
                    results.append(
                        OCRDocumentResult(
                            file_path=str(fp),
                            doc_id=str(uuid.uuid4())[:8],
                            pages=[],
                            raw_text="",
                            total_pages=0,
                            overall_confidence=0.0,
                            warnings=[f"Processing failed: {str(e)}"],
                        )
                    )

    if return_for_node0:
        return [
            {"raw_text": r.raw_text, "doc_id": r.doc_id}
            for r in results
        ]

    return results


def _compute_document_confidence(pages: List[OCRPageResult]) -> float:
    """
    Compute document-level confidence as a weighted average of page confidences.
    Weight is proportional to the text length on each page.
    """
    if not pages:
        return 0.0

    total_chars = sum(len(p.raw_text) for p in pages)
    if total_chars == 0:
        return 0.0

    weighted_sum = sum(p.confidence * len(p.raw_text) for p in pages)
    return weighted_sum / total_chars
