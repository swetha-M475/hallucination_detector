"""
llm/generator.py — Load a causal LM and generate text.

Supports both:
  - generate_full()         : returns GenerationOutput with hidden states for Entropy/SIVR/HARP
  - generate_with_samples() : returns main answer + stochastic samples for Self-Consistency
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils.device import get_device
import config


@dataclass
class GenerationOutput:
    """
    All signals produced during a full generation pass (used for Entropy, SIVR, HARP).

    Fields
    ------
    answer        : The decoded generated answer (string).
    tokens        : List of token strings (one per generated step).
    token_ids     : Corresponding integer IDs.
    logits        : Per-step logits tensor — each shape (vocab_size,).
    probabilities : Softmax of logits — each shape (vocab_size,).
    hidden_states : Per-step tuple of per-layer hidden states.
                    hidden_states[step][layer] has shape (hidden_dim,).
    prompt_len    : Number of tokens in the original prompt.
    """
    answer: str
    tokens: List[str]
    token_ids: List[int]
    logits: List[torch.Tensor]
    probabilities: List[torch.Tensor]
    hidden_states: List[Tuple[torch.Tensor, ...]]
    prompt_len: int = 0


class LLMGenerator:
    """Wraps a Hugging Face causal LM for text generation."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or get_device()
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    def load(self) -> "LLMGenerator":
        print(f"[Generator] Loading '{self.model_name}' on {self.device} …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self._model.eval()
        print(f"[Generator] Model loaded. Parameters: {sum(p.numel() for p in self._model.parameters()):,}")
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    @torch.no_grad()
    def generate(
        self,
        question: str,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = 0.1,
        raw_prompt: bool = False,
    ) -> str:
        """Generates a single response."""
        self.ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model

        if raw_prompt:
            prompt = question
        else:
            prompt = f"Question: {question}\nAnswer:"
            
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        prompt_len = inputs["input_ids"].shape[1]

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0.1 else False,
            temperature=temperature if temperature > 0.1 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        outputs = model.generate(
            **inputs,
            generation_config=gen_cfg,
        )

        generated_ids = outputs[0, prompt_len:].tolist()
        if tokenizer.eos_token_id in generated_ids:
            eos_pos = generated_ids.index(tokenizer.eos_token_id)
            generated_ids = generated_ids[:eos_pos]
            
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Simple stop criterion for few-shot bleed
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
            
        return answer

    @torch.no_grad()
    def generate_full(
        self,
        question: str,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> "GenerationOutput":
        """
        Full generation pass that captures logits + hidden states.
        Used to compute Entropy, Confidence, SIVR, and HARP signals.
        """
        self.ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        outputs = model.generate(
            **inputs,
            generation_config=gen_cfg,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        generated_ids = outputs.sequences[0, prompt_len:].tolist()
        if tokenizer.eos_token_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(tokenizer.eos_token_id)]

        tokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in generated_ids]

        # Stop at few-shot bleed
        stop_idx = len(generated_ids)
        accumulated = ""
        for i, t in enumerate(tokens):
            accumulated += t
            if "\n" in t or "Question:" in accumulated:
                stop_idx = i
                break
        generated_ids = generated_ids[:stop_idx]
        tokens = tokens[:stop_idx]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Parse logits/probs
        logits_list, probs_list = [], []
        for step_logits in outputs.scores[:len(generated_ids)]:
            logit = step_logits[0]
            probs_list.append(F.softmax(logit, dim=-1).cpu())
            logits_list.append(logit.cpu())

        # Parse hidden states
        hidden_list: List[Tuple[torch.Tensor, ...]] = []
        if outputs.hidden_states is not None:
            for step_hs in outputs.hidden_states[:len(generated_ids)]:
                hidden_list.append(tuple(layer_hs[0, -1, :].cpu() for layer_hs in step_hs))
        if len(hidden_list) < len(generated_ids):
            pad = generated_ids[len(hidden_list):]
            hidden_list.extend([hidden_list[-1]] * len(pad) if hidden_list else [()] * len(pad))

        return GenerationOutput(
            answer=answer, tokens=tokens, token_ids=generated_ids,
            logits=logits_list, probabilities=probs_list,
            hidden_states=hidden_list, prompt_len=prompt_len,
        )

    def get_unembedding_matrix(self) -> torch.Tensor:
        """Return the model's lm_head weight matrix (vocab_size, hidden_dim) for HARP."""
        self.ensure_loaded()
        model = self._model
        if hasattr(model, "lm_head"):
            return model.lm_head.weight.detach().cpu().float()
        elif hasattr(model, "embed_out"):
            return model.embed_out.weight.detach().cpu().float()
        raise AttributeError(f"Cannot find unembedding matrix in {type(model).__name__}.")

    def generate_with_samples(
        self, 
        question: str, 
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        num_samples: int = config.NUM_SAMPLES,
        sample_max_tokens: int = config.SAMPLE_MAX_TOKENS,
    ) -> Tuple[str, List[str]]:
        """Generates a main confident answer and multiple stochastic samples for consistency checking."""
        
        # 1. Generate main answer (low temp, full token budget)
        main_answer = self.generate(question, max_new_tokens, temperature=config.TEMPERATURE_MAIN)
        
        # 2. Generate samples (high temp, shorter = faster)
        samples = []
        for _ in range(num_samples):
            sample_answer = self.generate(question, sample_max_tokens, temperature=config.TEMPERATURE_SAMPLES)
            samples.append(sample_answer)
            
        return main_answer, samples



_generator_instance: Optional[LLMGenerator] = None

def get_generator(model_name: str = config.MODEL_NAME) -> LLMGenerator:
    """Return a cached LLMGenerator instance."""
    global _generator_instance
    if _generator_instance is None or _generator_instance.model_name != model_name:
        _generator_instance = LLMGenerator(model_name=model_name).load()
    return _generator_instance

