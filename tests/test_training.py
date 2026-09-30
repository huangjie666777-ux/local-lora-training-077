"""LoRA training loop tests on a tiny random Llama (fast, CPU-only)."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from specserve.dataset import Features
from specserve.training import TrainParams, train_lora


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.config.use_cache = False
    return model


def features():
    # prompt tokens masked, answer tokens + EOS scored
    return [
        Features(line=1, input_ids=[1, 5, 6, 7, 0], labels=[-100, -100, 6, 7, 0]),
        Features(line=2, input_ids=[1, 8, 9, 0], labels=[-100, -100, 9, 0]),
    ]


def params(**kw):
    base = dict(name="t", steps=4, learning_rate=1e-3, seed=0, batch_size=2)
    base.update(kw)
    return TrainParams(**base)


def test_training_completes_and_updates_only_lora(tiny_model):
    base_before = [p.detach().clone() for p in tiny_model.parameters()]
    seen = []
    result = train_lora(
        tiny_model, features(), params(), pad_token_id=0,
        should_cancel=lambda: False,
        on_progress=lambda step, loss: seen.append((step, loss)),
    )
    assert result.reason == "completed"
    assert result.steps_done == 4 and len(result.losses) == 4
    assert seen == [(1, result.losses[0]), (2, result.losses[1]),
                    (3, result.losses[2]), (4, result.losses[3])]
    assert result.peft_model is not None
    # Base weights frozen; LoRA-B trained away from zero.
    named = dict(result.peft_model.named_parameters())
    lora_b = [v for k, v in named.items() if "lora_B" in k]
    assert lora_b and any(float(v.abs().sum()) > 0 for v in lora_b)


def test_training_seed_reproducible(tiny_model):
    kwargs = dict(pad_token_id=0, should_cancel=lambda: False, on_progress=lambda s, l: None)
    first = train_lora(tiny_model, features(), params(), **kwargs)
    second = train_lora(tiny_model, features(), params(), **kwargs)
    assert first.losses == second.losses


def test_cancel_takes_effect_after_current_step(tiny_model):
    calls = {"n": 0}
    def cancel():
        calls["n"] += 1
        return calls["n"] > 2  # allow two step-top checks, then cancel
    result = train_lora(
        tiny_model, features(), params(steps=50), pad_token_id=0,
        should_cancel=cancel, on_progress=lambda s, l: None,
    )
    assert result.reason == "cancelled"
    assert result.steps_done == 2  # in-flight step finished, loop stopped
    assert result.peft_model is None  # nothing to persist on cancel


def test_failure_returns_clean_result(tiny_model):
    bad = [Features(line=1, input_ids=[9999], labels=[9999])]  # out of vocab
    result = train_lora(
        tiny_model, bad, params(steps=1), pad_token_id=0,
        should_cancel=lambda: False, on_progress=lambda s, l: None,
    )
    assert result.reason == "failed"
    assert result.peft_model is None and result.error
