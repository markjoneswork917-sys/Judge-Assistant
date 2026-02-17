"""
config.py

Configuration module for the OCR pipeline.

Purpose:
--------
Contains all OCR-specific constants and settings used across
the module, including Surya engine parameters, preprocessing toggles,
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
ENABLE_RESOLUTION_CHECK = True
MIN_DPI = 150
ENABLE_DESKEW = True
ENABLE_DENOISE = False  # Off by default; Surya is robust
ENABLE_BORDER_REMOVAL = True
ENABLE_CONTRAST_ENHANCEMENT = True

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
NORMALIZE_DIGITS = "arabic_indic"  # arabic_indic | western | preserve

# -----------------------------
# Security
# -----------------------------
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"]

# -----------------------------
# Performance
# -----------------------------
SURYA_BATCH_SIZE = 4  # Pages per batch for GPU inference
BATCH_WORKERS = 4

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICTIONARY_PATH = os.path.join(BASE_DIR, "dictionaries", "legal_arabic.txt")
