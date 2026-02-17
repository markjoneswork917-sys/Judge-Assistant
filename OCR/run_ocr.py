"""
run_ocr.py

Simple CLI script to run OCR on an image and print the extracted text.

Usage:
    python -m OCR.run_ocr <image_path>
    python -m OCR.run_ocr <image_path> --json
    python -m OCR.run_ocr <image_path> --node0
"""

import argparse
import json
import sys
import os

# Add project root to path so OCR module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OCR.ocr_pipeline import process_document


def main():
    parser = argparse.ArgumentParser(
        description="Extract Arabic text from document images using OCR"
    )
    parser.add_argument(
        "image_path",
        help="Path to the image or PDF file to process",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full structured result as JSON",
    )
    parser.add_argument(
        "--node0",
        action="store_true",
        help="Output in Node 0 integration format",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Custom document ID (auto-generated if not provided)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    if args.node0:
        result = process_document(
            args.image_path, return_for_node0=True, doc_id=args.doc_id
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.json:
        result = process_document(args.image_path, doc_id=args.doc_id)
        print(result.json(ensure_ascii=False, indent=2))
    else:
        result = process_document(args.image_path, doc_id=args.doc_id)
        print(f"File: {result.file_path}")
        print(f"Pages: {result.total_pages}")
        print(f"Confidence: {result.overall_confidence:.2%}")
        if result.warnings:
            print(f"Warnings: {len(result.warnings)}")
        print("---")
        print(result.raw_text)


if __name__ == "__main__":
    main()
