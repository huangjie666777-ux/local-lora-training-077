import torch

from specserve.probability import (
    acceptance_probability,
    greedy_token,
    residual_distribution,
    sample_from_probs,
    temperature_probs,
)


def test_temperature_probs_normalized():
    logits = torch.randn(100)
    probs = temperature_probs(logits, 0.7)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs >= 0).all()


def test_greedy_token_is_argmax():
    logits = torch.zeros(10)
    logits[4] = 3.0
    assert greedy_token(logits) == 4


def test_sampling_seeded_reproducible():
    probs = temperature_probs(torch.randn(50), 1.0)
    a = sample_from_probs(probs, torch.Generator().manual_seed(7))
    b = sample_from_probs(probs, torch.Generator().manual_seed(7))
    assert a == b


def test_acceptance_probability_bounds():
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.25, 0.75])
    assert acceptance_probability(p, q, 0) == 1.0
    assert abs(acceptance_probability(p, q, 1) - 2 / 3) < 1e-6


def test_residual_distribution_positive_part():
    p = torch.tensor([0.6, 0.1, 0.3])
    q = torch.tensor([0.2, 0.5, 0.3])
    r = residual_distribution(p, q)
    assert torch.isclose(r.sum(), torch.tensor(1.0))
    assert r[0] == 1.0 and r[1] == 0.0 and r[2] == 0.0


def test_residual_distribution_fallback_when_identical():
    p = torch.tensor([0.5, 0.5])
    r = residual_distribution(p, p.clone())
    assert torch.allclose(r, p)

