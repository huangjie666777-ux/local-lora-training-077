"""Engine tests run real CPU inference on the bundled SmolLM2 models."""
import pytest

from specserve.engine import EngineParams, generate
from specserve.modeling import ModelRegistry

PROMPT = "The little robot woke up in the middle of the workshop and"


@pytest.fixture(scope="session")
def models():
    return ModelRegistry().get()


def run(models, **kwargs):
    params = EngineParams(prompt=PROMPT, **kwargs)
    text, done = "", None
    for event in generate(models, params, lambda: False):
        if event["type"] == "token":
            text += event["text"]
        elif event["type"] == "done":
            done = event
    return text, done["stats"]


def test_autoregressive_greedy(models):
    text, stats = run(models, mode="autoregressive", max_new_tokens=12)
    assert len(text) > 0
    assert stats["tokens"] <= 12
    assert stats["target_forwards"] >= stats["tokens"]


def test_speculative_matches_autoregressive_greedy(models):
    """Temp=0 speculative output must equal plain greedy decoding."""
    ref, _ = run(models, mode="autoregressive", max_new_tokens=16)
    spec, stats = run(models, mode="speculative", max_new_tokens=16, draft_steps=4)
    assert spec == ref
    assert stats["candidates"] > 0
    assert stats["accepted"] <= stats["candidates"]
    # Whole point: fewer serial target forwards than tokens.
    assert stats["target_forwards"] < stats["tokens"]


def test_speculative_sampled_reproducible(models):
    kwargs = dict(mode="speculative", max_new_tokens=16, draft_steps=4, temperature=0.8, seed=123)
    first, _ = run(models, **kwargs)
    second, _ = run(models, **kwargs)
    assert first == second and len(first) > 0


def test_stop_event_halts_generation(models):
    params = EngineParams(prompt=PROMPT, mode="speculative", max_new_tokens=64)
    events = list(generate(models, params, lambda: True))
    assert events[-1]["stopped"] is True
    assert events[-1]["stats"]["tokens"] == 0



def test_full_accept_keeps_draft_cache_aligned(models):
    """Draft == target means every candidate must be accepted.

    If the draft cache lost the final candidate of a fully accepted
    block, later proposals would come from a corrupted cache and
    acceptance would drop below 100%.
    """
    from specserve.modeling import ModelPair

    twin = ModelPair(
        tokenizer=models.tokenizer,
        draft=models.target,  # same weights as target: perfect draft
        target=models.target,
        eos_token_id=models.eos_token_id,
    )
    params = EngineParams(prompt=PROMPT, mode="speculative", max_new_tokens=12, draft_steps=4)
    done = None
    for event in generate(twin, params, lambda: False):
        if event["type"] == "done":
            done = event
    stats = done["stats"]
    assert stats["candidates"] > 0
    assert stats["accepted"] == stats["candidates"]


def test_no_forward_after_stop_or_budget(models):
    """Autoregressive loop never runs a forward whose logits are unused."""
    # Budget spent: forwards == prompt pass + (tokens - 1) follow-ups.
    params = EngineParams(prompt=PROMPT, mode="autoregressive", max_new_tokens=4)
    done = None
    for event in generate(models, params, lambda: False):
        if event["type"] == "done":
            done = event
    stats = done["stats"]
    assert stats["tokens"] == 4
    assert stats["target_forwards"] == stats["tokens"]

    # Stop requested mid-request: no extra forward after the stop check.
    calls = {"n": 0}
    def stop_after_two_tokens():
        calls["n"] += 1
        return calls["n"] >= 4  # two tokens committed, second check trips
    params = EngineParams(prompt=PROMPT, mode="autoregressive", max_new_tokens=16)
    done = None
    for event in generate(models, params, stop_after_two_tokens):
        if event["type"] == "done":
            done = event
    stats = done["stats"]
    assert stats["tokens"] == 2
    assert stats["target_forwards"] == 2  # prompt pass + one follow-up only
