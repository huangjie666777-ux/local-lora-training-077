"""Regression tests for speculative-loop cache/stop bugs.

Uses tiny identical draft==target Llama models: greedy decoding then always
accepts every candidate, which is exactly the case that used to (a) leave the
draft cache missing the last candidate and (b) keep running forwards after a
cooperative stop.
"""
import types

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from specserve.engine import EngineParams, generate
from specserve.modeling import ModelPair


def tiny_model(seed=42, vocab=64):
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


class Tok:
    eos_token_id = 0
    bos_token_id = None

    def __call__(self, prompt, return_tensors=None):
        return types.SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i % 26)) for i in ids)


def pair():
    # Identical weights for draft and target: every greedy proposal matches.
    target = tiny_model(seed=42)
    draft = tiny_model(seed=42)
    return ModelPair(tokenizer=Tok(), draft=draft, target=target, eos_token_id=0)


def test_full_acceptance_draft_cache_stays_aligned():
    models = pair()
    params = EngineParams(prompt="abc", mode="speculative", max_new_tokens=16, draft_steps=4)
    events = list(generate(models, params, lambda: False))
    done = events[-1]
    assert done["type"] == "done"
    stats = done["stats"]
    # draft == target and greedy: all proposed candidates must be accepted
    assert stats["accepted"] == stats["candidates"]
    assert stats["tokens"] == 16


def test_stop_before_first_iteration_runs_no_verification():
    models = pair()
    params = EngineParams(prompt="abc", mode="speculative", max_new_tokens=16, draft_steps=4)
    events = list(generate(models, params, lambda: True))
    done = events[-1]
    assert done["stopped"] is True
    # Only the initial prompt forwards, no verification block.
    assert done["stats"]["target_forwards"] == 1
    assert done["stats"]["draft_forwards"] == 1
    assert done["stats"]["tokens"] == 0


def test_autoregressive_stop_skips_post_token_forward():
    models = pair()
    stop_after = {"n": 0}

    def should_stop():
        return stop_after["n"] >= 1

    events = []
    for ev in generate(
        models,
        EngineParams(prompt="abc", mode="autoregressive", max_new_tokens=8),
        should_stop,
    ):
        events.append(ev)
        if ev.get("type") == "token":
            stop_after["n"] += 1
    done = events[-1]
    assert done["stopped"] is True
    # One prompt forward; the final post-token forward must be skipped.
    assert done["stats"]["target_forwards"] == 1
