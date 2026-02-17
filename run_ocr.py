"""
Run OCR on an image file and print the extracted text.

Usage:
    python run_ocr.py path/to/image.png
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from OCR.ocr_pipeline import process_document


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_ocr.py <image_path>")
        print("Example: python run_ocr.py document.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Processing: {image_path}")
    print("Loading OCR engine (first run downloads the model)...")

    result = process_document(image_path)

    print(f"\nPages: {result.total_pages}")
    print(f"Confidence: {result.overall_confidence:.0%}")
    print("=" * 50)
    print(result.raw_text)
    print("=" * 50)


if __name__ == "__main__":
    main()
