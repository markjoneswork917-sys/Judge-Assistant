"""
config.py

Configuration module for the OCR pipeline.

Purpose:
--------
Contains all OCR-specific constants and settings used across
the module, including engine parameters, preprocessing toggles,
confidence thresholds, and security limits.

Design Principle:
-----------------
Configuration is isolated from business logic.
Changing thresholds or toggling preprocessing steps should
not require editing core OCR code.
"""

import os

# -----------------------------
# Engine
# -----------------------------
OCR_LANGUAGE = "ar"
USE_GPU = True  # Auto-detect CUDA, fallback to CPU

# -----------------------------
# Preprocessing
# -----------------------------
TARGET_DPI = 300
ENABLE_DESKEW = True
ENABLE_DENOISE = True
ENABLE_BORDER_REMOVAL = True
ENABLE_CONTRAST_ENHANCEMENT = True
BINARIZATION_METHOD = "otsu"  # otsu | sauvola | adaptive

# -----------------------------
# Confidence Thresholds
# -----------------------------
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.60

# -----------------------------
# Post-processing
# -----------------------------
ENABLE_DICTIONARY_CORRECTION = True
MAX_LEVENSHTEIN_DISTANCE = 2

# -----------------------------
# Security
# -----------------------------
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"]

# -----------------------------
# Performance
# -----------------------------
BATCH_WORKERS = 4

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICTIONARY_PATH = os.path.join(BASE_DIR, "dictionaries", "legal_arabic.txt")
