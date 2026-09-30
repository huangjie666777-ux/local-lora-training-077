"""Adapter store: atomic save, listing, compatibility and corruption."""
import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from specserve.adapters import AdapterError, AdapterStore


@pytest.fixture(scope="module")
def peft_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    return get_peft_model(
        LlamaForCausalLM(config),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )


@pytest.fixture
def store(tmp_path):
    return AdapterStore(tmp_path / "adapters", base_identity={"hidden_size": 32})


def metadata():
    return {"lora": {"r": 4}, "training": {"steps": 2}, "samples": {"sample_count": 1}}


def test_save_list_and_check(store, peft_model):
    store.save_atomic(peft_model, "v1", metadata())
    entries = store.list()
    assert len(entries) == 1 and entries[0]["name"] == "v1" and entries[0]["usable"]
    meta = store.check_compatible("v1")
    assert meta["base_model"]["hidden_size"] == 32
    assert meta["training"]["steps"] == 2


def test_no_partial_dir_left_on_failure(store, peft_model, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(type(peft_model), "save_pretrained", boom)
    with pytest.raises(RuntimeError):
        store.save_atomic(peft_model, "broken", metadata())
    assert [p.name for p in store.directory.iterdir()] == []


def test_duplicate_name_rejected(store, peft_model):
    store.save_atomic(peft_model, "v1", metadata())
    with pytest.raises(AdapterError, match="already exists"):
        store.save_atomic(peft_model, "v1", metadata())


def test_corrupt_adapter_detected(store, peft_model):
    store.save_atomic(peft_model, "v1", metadata())
    (store.adapter_path("v1") / "adapter_model.safetensors").unlink()
    with pytest.raises(AdapterError, match="corrupt"):
        store.check_compatible("v1")
    entry = store.list()[0]
    assert entry["usable"] is False


def test_incompatible_base_rejected(store, peft_model):
    store.save_atomic(peft_model, "v1", metadata())
    meta_path = store.adapter_path("v1") / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["base_model"]["hidden_size"] = 999
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(AdapterError, match="different base model"):
        store.check_compatible("v1")


def test_unknown_and_invalid_names(store):
    with pytest.raises(AdapterError, match="unknown"):
        store.check_compatible("nope")
    with pytest.raises(AdapterError, match="invalid"):
        store.adapter_path("../escape")
