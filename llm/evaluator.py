"""
llm/evaluator.py — Fast Self-Consistency Scorer.

Instead of an LLM-as-a-judge (which needs extra generation calls),
we use word-overlap (Jaccard similarity) between the main answer and each
stochastic sample. Low overlap = the model contradicted itself = hallucination risk.
"""

from typing import List
import re


def tokenize(text: str) -> set:
    """Lowercase, remove punctuation, return set of words."""
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between word sets of two strings. Returns 0..1."""
    set_a = tokenize(a)
    set_b = tokenize(b)
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def evaluate_consistency(question: str, main_answer: str, samples: List[str]) -> float:
    """
    Computes a hallucination risk score via self-consistency.

    For each stochastic sample, we measure word-overlap with the main answer.
    High overlap → consistent → low risk.
    Low overlap  → contradiction → high risk.

    Returns a risk score between 0.0 (no risk) and 1.0 (high risk).
    """
    if not samples or not main_answer.strip():
        return 0.5  # Uncertain when no data

    similarities = [jaccard_similarity(main_answer, s) for s in samples]
    avg_similarity = sum(similarities) / len(similarities)

    # Convert similarity to risk: low similarity = high risk
    risk = 1.0 - avg_similarity
    return round(risk, 4)

