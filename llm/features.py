import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from llm.generator import GenerationOutput
import config

_CACHED_V = None

def compute_entropy(probs: torch.Tensor) -> float:
    """Calculates Shannon entropy."""
    epsilon = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + epsilon)).item()
    return entropy

def compute_confidence(probs: torch.Tensor) -> float:
    """Returns max probability."""
    return torch.max(probs).item()

def compute_sivr(hidden_states_step: Tuple[torch.Tensor, ...], layer_pairs: List[Tuple[int, int]]) -> float:
    """Measures variance of hidden representations across late-stage Transformer layers."""
    total_variation = 0.0
    valid_pairs = 0
    for l1, l2 in layer_pairs:
        if abs(l1) <= len(hidden_states_step) and abs(l2) <= len(hidden_states_step):
            h1 = hidden_states_step[l1].float()
            h2 = hidden_states_step[l2].float()
            
            cos_sim = torch.nn.functional.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
            variation = 1.0 - cos_sim
            total_variation += variation
            valid_pairs += 1
            
    if valid_pairs == 0:
        return 0.0
        
    return total_variation / valid_pairs

def compute_harp(hidden_states_step: Tuple[torch.Tensor, ...], unembedding_matrix: torch.Tensor, k: int = 5) -> float:
    """Applies SVD to the unembedding matrix and projects final hidden state."""
    global _CACHED_V
    if _CACHED_V is None:
        W = unembedding_matrix.float()
        # SVD might fail or be slow on CPU for large vocabularies, but we only do it once.
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        _CACHED_V = Vh[:k, :] # Top k directions
        
    final_hidden = hidden_states_step[-1].float()
    projection = torch.matmul(_CACHED_V, final_hidden)
    return torch.norm(projection).item()

def extract_features(outputs: GenerationOutput, unembedding_matrix: torch.Tensor) -> List[Dict[str, Any]]:
    """Extracts features per token."""
    token_features = []
    
    for i in range(len(outputs.tokens)):
        token = outputs.tokens[i]
        prob_tensor = outputs.probabilities[i]
        hs_step = outputs.hidden_states[i]
        
        entropy = compute_entropy(prob_tensor)
        confidence = compute_confidence(prob_tensor)
        sivr = compute_sivr(hs_step, config.SIVR_LAYER_PAIRS)
        harp = compute_harp(hs_step, unembedding_matrix)
        
        token_features.append({
            "token": token,
            "entropy": entropy,
            "confidence": confidence,
            "sivr": sivr,
            "harp": harp
        })
        
    return token_features

def aggregate_sentence_features(token_features: List[Dict[str, Any]], delimiters: List[str] = config.SENTENCE_DELIMITERS) -> List[Dict[str, Any]]:
    sentences = []
    current_sentence_tokens = []
    current_sentence_features = []
    
    for tf in token_features:
        current_sentence_tokens.append(tf["token"])
        current_sentence_features.append(tf)
        
        if any(delim in tf["token"] for delim in delimiters):
            if current_sentence_tokens:
                text = "".join(current_sentence_tokens)
                avg_entropy = float(np.mean([f["entropy"] for f in current_sentence_features]))
                avg_confidence = float(np.mean([f["confidence"] for f in current_sentence_features]))
                avg_sivr = float(np.mean([f["sivr"] for f in current_sentence_features]))
                avg_harp = float(np.mean([f["harp"] for f in current_sentence_features]))
                
                sentences.append({
                    "text": text,
                    "entropy": avg_entropy,
                    "confidence": avg_confidence,
                    "sivr": avg_sivr,
                    "harp": avg_harp,
                    "tokens": current_sentence_features
                })
                current_sentence_tokens = []
                current_sentence_features = []
                
    if current_sentence_tokens:
        text = "".join(current_sentence_tokens)
        avg_entropy = float(np.mean([f["entropy"] for f in current_sentence_features]))
        avg_confidence = float(np.mean([f["confidence"] for f in current_sentence_features]))
        avg_sivr = float(np.mean([f["sivr"] for f in current_sentence_features]))
        avg_harp = float(np.mean([f["harp"] for f in current_sentence_features]))
        
        sentences.append({
            "text": text,
            "entropy": avg_entropy,
            "confidence": avg_confidence,
            "sivr": avg_sivr,
            "harp": avg_harp,
            "tokens": current_sentence_features
        })
        
    return sentences

def aggregate_response_features(token_features: List[Dict[str, Any]]) -> Dict[str, float]:
    if not token_features:
        return {"entropy": 0.0, "confidence": 0.0, "sivr": 0.0, "harp": 0.0}
        
    return {
        "entropy": float(np.mean([f["entropy"] for f in token_features])),
        "confidence": float(np.mean([f["confidence"] for f in token_features])),
        "sivr": float(np.mean([f["sivr"] for f in token_features])),
        "harp": float(np.mean([f["harp"] for f in token_features]))
    }
