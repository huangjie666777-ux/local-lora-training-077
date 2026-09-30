"""Real CPU LoRA training on a tiny Llama, plus save/load/identity checks."""
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from specserve.data import Sample, prepare_batch
from specserve.modeling import AdapterError, ModelRegistry
from specserve.training import (
    TrainConfig,
    TrainStatus,
    build_lora_model,
    run_training,
    validate_train_config,
)


def tiny_target():
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
    )
    torch.manual_seed(1)
    model = LlamaForCausalLM(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Tok:
    bos_token_id = None
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [10 + (ord(w[0]) % 40) for w in text.split()]
        return type("Out", (), {"input_ids": ids})()


@pytest.fixture
def samples():
    raw = [
        Sample("Question one", "answer alpha"),
        Sample("Question two", "answer beta"),
    ]
    return prepare_batch(raw, Tok(), max_length=64)


def test_config_limits_reject_out_of_range():
    assert validate_train_config(TrainConfig(name="x", steps=0))
    assert validate_train_config(TrainConfig(name="x", learning_rate=5.0))
    good = TrainConfig(name="x", steps=3, learning_rate=1e-4, seed=1, max_length=64)
    assert validate_train_config(good) == []


def test_lora_only_trainable_and_loss_decreases(samples):
    encoded, _ = samples
    target = tiny_target()
    before = {n: p.detach().clone() for n, p in target.named_parameters()}
    peft_model = build_lora_model(target)
    assert any(p.requires_grad for p in peft_model.parameters())
    base_trainable = [
        p for n, p in peft_model.named_parameters()
        if p.requires_grad and "lora_" not in n
    ]
    assert not base_trainable
    cfg = TrainConfig(name="x", steps=8, learning_rate=0.05, seed=0, max_length=64)
    status = TrainStatus(name="x")
    run_training(peft_model, encoded, cfg, status, lambda: False)
    assert status.state == "completed"
    assert status.reason == "configured_steps_reached"
    assert status.actual_steps == 8
    assert status.losses[-1] < status.losses[0]
    for n, p in peft_model.named_parameters():
        if "lora_" in n:
            continue
        base_weight = p.base_layer if hasattr(p, "base_layer") else p
        key = n.split("base_model.model.")[-1].replace(".base_layer", "")
        assert torch.equal(base_weight, before[key])


def test_cancel_takes_effect_after_step(samples):
    encoded, _ = samples
    peft_model = build_lora_model(tiny_target())
    cfg = TrainConfig(name="x", steps=10, learning_rate=1e-3, max_length=64)
    status = TrainStatus(name="x")
    counter = {"n": 0}

    def cancel():
        counter["n"] += 1
        return counter["n"] >= 3

    run_training(peft_model, encoded, cfg, status, cancel)
    assert status.state == "cancelled"
    assert status.actual_steps == 3
    assert "step 3" in status.reason


def test_save_adapter_meta_and_incompatible_base(tmp_path, samples, monkeypatch):
    import specserve.modeling as modeling

    encoded, summary = samples
    registry = ModelRegistry(adapters_dir=tmp_path)
    fixed = {"base_name": "SmolLM2-360M", "config_sha256": "abc"}
    monkeypatch.setattr(modeling, "base_model_identity", lambda target_dir=None: dict(fixed))
    peft_model = build_lora_model(tiny_target())
    cfg = TrainConfig(name="style-v1", steps=2, learning_rate=1e-3, max_length=64)
    status = TrainStatus(name="style-v1")
    run_training(peft_model, encoded, cfg, status, lambda: False)
    registry.save_adapter(
        peft_model, "style-v1", cfg.as_dict(), summary.as_dict(),
        "2026-09-30T00:00:00+00:00", status.last_loss, status.actual_steps,
    )
    saved = json.loads((tmp_path / "style-v1" / "adapter_meta.json").read_text())
    assert saved["base"]["config_sha256"] == "abc"
    assert registry.list_adapters()[0]["name"] == "style-v1"
    with pytest.raises(AdapterError):
        registry.save_adapter(
            peft_model, "style-v1", cfg.as_dict(), summary.as_dict(),
            "2026-09-30T00:00:00+00:00", 0.0, 2,
        )

    registry._pair = "OLD"
    monkeypatch.setattr(
        modeling, "base_model_identity",
        lambda target_dir=None: {"base_name": "other", "config_sha256": "zzz"},
    )
    with pytest.raises(AdapterError):
        registry.activate_adapter("style-v1")
    assert registry._pair == "OLD"


def test_corrupt_adapter_keeps_old_version(tmp_path, samples, monkeypatch):
    import specserve.modeling as modeling

    encoded, summary = samples
    registry = ModelRegistry(adapters_dir=tmp_path)
    fixed = {"base_name": "SmolLM2-360M", "config_sha256": "abc"}
    monkeypatch.setattr(modeling, "base_model_identity", lambda target_dir=None: dict(fixed))
    peft_model = build_lora_model(tiny_target())
    cfg = TrainConfig(name="style-v2", steps=2, learning_rate=1e-3, max_length=64)
    status = TrainStatus(name="style-v2")
    run_training(peft_model, encoded, cfg, status, lambda: False)
    registry.save_adapter(
        peft_model, "style-v2", cfg.as_dict(), summary.as_dict(),
        "2026-09-30T00:00:00+00:00", status.last_loss, status.actual_steps,
    )
    registry._pair = "OLD"
    monkeypatch.setattr(registry, "_load_base_target", tiny_target)
    (tmp_path / "style-v2" / "adapter_model.safetensors").write_text("garbage")
    with pytest.raises(AdapterError):
        registry.activate_adapter("style-v2")
    assert registry._pair == "OLD"


def test_invalid_adapter_name(tmp_path):
    registry = ModelRegistry(adapters_dir=tmp_path)
    with pytest.raises(AdapterError):
        registry._adapter_dir("../escape")
