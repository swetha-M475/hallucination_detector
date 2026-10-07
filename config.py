"""
config.py — Central configuration for the hallucination detection system.

Change MODEL_NAME here to swap the underlying LLM (any causal LM on HuggingFace).
"""

import os

# ─── LLM Settings ────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B"          # Swap to any Hugging Face causal LM
MAX_NEW_TOKENS = 150

# ─── Self-Consistency Settings ───────────────────────────────────────────────
NUM_SAMPLES = 2                           # Number of stochastic samples (keep low for speed)
TEMPERATURE_MAIN = 0.1                    # Low temp for main greedy answer
TEMPERATURE_SAMPLES = 1.0                 # High temp for stochastic samples
SAMPLE_MAX_TOKENS = 60                    # Samples can be shorter than the main answer

# ─── White-Box Signal Settings ───────────────────────────────────────────────
SIVR_LAYER_PAIRS = [(-4, -3), (-3, -2), (-2, -1)]  # Late transformer layer pairs for SIVR
SENTENCE_DELIMITERS = [".", "!", "?"]

# ─── Risk Thresholds ─────────────────────────────────────────────────────────
RISK_PROB_LOW  = 0.33
RISK_PROB_HIGH = 0.66

# ─── Classifier Settings ─────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_MODEL_PATH = os.path.join(_BASE_DIR, "models", "classifier.pkl")
FALLBACK_CSV          = os.path.join(_BASE_DIR, "data", "sample_data.csv")
