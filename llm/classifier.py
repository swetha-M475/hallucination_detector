"""
llm/classifier.py — Logistic Regression classifier for hallucination risk.

Training:
    Reads questions + binary labels from config.FALLBACK_CSV,
    runs generate_full() to get white-box features (Entropy, Confidence, SIVR, HARP),
    fits a StandardScaler + LogisticRegression, and saves to config.CLASSIFIER_MODEL_PATH.

Inference:
    predict_risk(features_dict) loads the saved model and returns a probability in [0, 1].
    Falls back to 0.5 if no model file exists yet.

Run standalone to train:
    python -m llm.classifier
"""

import os
import sys
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.generator import get_generator
from llm.features import extract_features, aggregate_response_features
import config


# ── Training ──────────────────────────────────────────────────────────────────

def train_classifier() -> bool:
    """
    Train the classifier from config.FALLBACK_CSV and save to config.CLASSIFIER_MODEL_PATH.
    Returns True on success, False if the dataset is missing.
    """
    print("Loading dataset from:", config.FALLBACK_CSV)
    if not os.path.exists(config.FALLBACK_CSV):
        print("Dataset not found — skipping training.")
        return False

    df = pd.read_csv(config.FALLBACK_CSV)
    if "question" not in df.columns or "label" not in df.columns:
        print("CSV must have 'question' and 'label' columns.")
        return False

    print(f"Loaded {len(df)} samples.")

    generator = get_generator()
    unembedding = generator.get_unembedding_matrix()

    X = []
    y = df["label"].values

    print("Extracting features (this may take a moment)…")
    for idx, row in df.iterrows():
        question = str(row["question"])
        print(f"  [{idx + 1}/{len(df)}] {question}")

        # generate_full() returns GenerationOutput with logits + hidden states
        # (generate() only returns a plain str — do NOT use it here)
        output = generator.generate_full(question, max_new_tokens=20)
        token_features = extract_features(output, unembedding)
        agg = aggregate_response_features(token_features)

        X.append([
            agg["entropy"],
            agg["confidence"],
            agg["sivr"],
            agg["harp"],
        ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Logistic Regression classifier…")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_scaled, y)

    # Ensure output directory exists
    model_dir = os.path.dirname(config.CLASSIFIER_MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Saving model to {config.CLASSIFIER_MODEL_PATH} …")
    with open(config.CLASSIFIER_MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler}, f)

    print("Training complete.")
    return True


def retrain_if_missing() -> None:
    """Convenience helper: train only when the model file does not exist yet."""
    if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
        print("[Classifier] No trained model found — running training now…")
        train_classifier()


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_risk(features_dict: dict) -> Optional[float]:
    """
    Predict hallucination risk probability from a feature dict.

    Parameters
    ----------
    features_dict : dict with keys 'entropy', 'confidence', 'sivr', 'harp'

    Returns
    -------
    float in [0, 1]  — probability of class 1 (hallucination)
    None             — if the model file does not exist (caller should fall back)
    """
    if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
        return None          # caller decides what to do (fallback to manual formula)

    with open(config.CLASSIFIER_MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    clf: LogisticRegression = data["model"]
    scaler: StandardScaler  = data["scaler"]

    x = [[
        features_dict["entropy"],
        features_dict["confidence"],
        features_dict["sivr"],
        features_dict["harp"],
    ]]
    x_scaled = scaler.transform(x)
    prob = clf.predict_proba(x_scaled)[0][1]   # probability of class 1 (hallucination)
    return float(prob)


# ── Standalone entry-point ────────────────────────────────────────────────────

if __name__ == "__main__":
    train_classifier()
