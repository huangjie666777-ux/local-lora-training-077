"""Probability helpers for temperature scaling, sampling and rejection."""
from __future__ import annotations

import torch


def temperature_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-normalized categorical distribution over the vocabulary."""
    if temperature <= 0:
        raise ValueError("temperature_probs requires temperature > 0")
    return torch.softmax(logits.float() / temperature, dim=-1)


def greedy_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits.float(), dim=-1).item())


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def acceptance_probability(p: torch.Tensor, q: torch.Tensor, token: int) -> float:
    """min(1, p(x)/q(x)) for a draft token x."""
    qx = float(q[token])
    if qx <= 0.0:
        return 1.0
    return min(1.0, float(p[token]) / qx)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Normalized positive part of (p - q), used after a rejection."""
    diff = (p - q).clamp_min(0.0)
    total = diff.sum()
    if float(total) <= 0.0:
        return p
    return diff / total

