"""Decoding engine: autoregressive and speculative loops with KV caching.

Both loops keep per-request DynamicCache objects so history is never
recomputed. The speculative loop feeds the draft's candidate block to the
target in a single forward pass, then crops both caches back to the
confirmed prefix so unconfirmed tokens never leak into later steps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch
from transformers.cache_utils import DynamicCache

from .modeling import ModelPair
from .probability import (
    acceptance_probability,
    greedy_token,
    residual_distribution,
    sample_from_probs,
    temperature_probs,
)


@dataclass
class EngineParams:
    prompt: str
    mode: str = "speculative"  # "autoregressive" | "speculative"
    max_new_tokens: int = 64
    draft_steps: int = 4
    temperature: float = 0.0
    seed: int = 0


@dataclass
class Stats:
    candidates: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    tokens: int = 0
    started: float = field(default_factory=time.perf_counter)

    def snapshot(self) -> dict:
        return {
            "candidates": self.candidates,
            "accepted": self.accepted,
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "tokens": self.tokens,
            "elapsed_seconds": round(time.perf_counter() - self.started, 3),
        }


def _forward(model, input_ids: torch.Tensor, cache: DynamicCache):
    """Single forward pass reusing the KV cache with an explicit mask."""
    mask = torch.ones(
        (1, cache.get_seq_length() + input_ids.shape[1]),
        dtype=torch.long,
        device=input_ids.device,
    )
    return model(
        input_ids=input_ids,
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
    )


def _pick(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    if temperature <= 0:
        return greedy_token(logits)
    return sample_from_probs(temperature_probs(logits, temperature), generator)


def generate(models: ModelPair, params: EngineParams, should_stop: Callable[[], bool]) -> Iterator[dict]:
    if params.mode == "autoregressive":
        yield from _generate_autoregressive(models, params, should_stop)
    else:
        yield from _generate_speculative(models, params, should_stop)


def _emit_text(models: ModelPair, committed: list[int], emitted: int) -> tuple[str, int]:
    """Decode only confirmed tokens; returns (new_text, new_emitted_count)."""
    text = models.tokenizer.decode(committed, skip_special_tokens=True)
    chunk = text[emitted:]
    return chunk, len(text)


def _generate_autoregressive(models, params, should_stop) -> Iterator[dict]:
    tokenizer = models.tokenizer
    target = models.target
    generator = torch.Generator(device="cpu").manual_seed(params.seed)
    stats = Stats()
    ids = tokenizer(params.prompt, return_tensors="pt").input_ids
    cache = DynamicCache()
    with torch.inference_mode():
        out = _forward(target, ids, cache)
        stats.target_forwards += 1
        logits = out.logits[0, -1]
        committed: list[int] = []
        emitted = 0
        stopped = False
        for _ in range(params.max_new_tokens):
            if should_stop():
                stopped = True
                break
            token = _pick(logits, params.temperature, generator)
            if token == models.eos_token_id:
                break
            committed.append(token)
            stats.tokens = len(committed)
            chunk, emitted = _emit_text(models, committed, emitted)
            if chunk:
                yield {"type": "token", "text": chunk}
            if should_stop():
                # Cooperative stop: never launch the post-token forward.
                stopped = True
                break
            out = _forward(target, torch.tensor([[token]]), cache)
            stats.target_forwards += 1
            logits = out.logits[0, -1]
    yield {"type": "done", "stopped": stopped, "stats": stats.snapshot()}


def _generate_speculative(models, params, should_stop) -> Iterator[dict]:
    tokenizer = models.tokenizer
    draft, target = models.draft, models.target
    eos = models.eos_token_id
    generator = torch.Generator(device="cpu").manual_seed(params.seed)
    stats = Stats()
    ids = tokenizer(params.prompt, return_tensors="pt").input_ids
    target_cache = DynamicCache()
    draft_cache = DynamicCache()
    committed: list[int] = []
    emitted = 0
    stopped = False
    eos_hit = False
    with torch.inference_mode():
        out = _forward(target, ids, target_cache)
        stats.target_forwards += 1
        target_logits = out.logits[0, -1]
        out = _forward(draft, ids, draft_cache)
        stats.draft_forwards += 1
        draft_logits = out.logits[0, -1]
        pending: int | None = None  # last committed token not yet in the caches

        while len(committed) < params.max_new_tokens and not eos_hit:
            if should_stop():
                stopped = True
                break
            remaining = params.max_new_tokens - len(committed)
            steps = min(params.draft_steps, remaining)
            base = target_cache.get_seq_length()
            offset = 1 if pending is not None else 0

            # 1) Draft proposes candidates one token at a time.
            if pending is not None:
                out = _forward(draft, torch.tensor([[pending]]), draft_cache)
                stats.draft_forwards += 1
                draft_logits = out.logits[0, -1]
            candidates: list[int] = []
            draft_probs: list[torch.Tensor | None] = []
            for draft_idx in range(steps):
                if params.temperature <= 0:
                    token = greedy_token(draft_logits)
                    draft_probs.append(None)
                else:
                    probs = temperature_probs(draft_logits, params.temperature)
                    token = sample_from_probs(probs, generator)
                    draft_probs.append(probs)
                candidates.append(token)
                stats.candidates += 1
                if token == eos:  # truncate candidates at EOS
                    break
                # Advance after every non-EOS proposal, including the last
                # one. If fewer tokens are later accepted the cache is cropped
                # back; this guarantees a fully accepted block (no
                # replacement) still contains its final candidate, which used
                # to be missing and shifted draft positions by one.
                out = _forward(draft, torch.tensor([[token]]), draft_cache)
                stats.draft_forwards += 1
                draft_logits = out.logits[0, -1]

            if should_stop():
                # Cooperative cancel: crop speculative draft lookahead and do
                # not run the target verification forward at all.
                draft_cache.crop(base + offset)
                stopped = True
                break

            # 2) Target verifies the whole block in one forward pass.
            block = ([pending] if pending is not None else []) + candidates
            out = _forward(target, torch.tensor([block]), target_cache)
            stats.target_forwards += 1
            block_logits = out.logits[0]
            # Distribution scoring candidates[i]:
            #   with pending: block_logits[i]; without: stored logits for i==0
            #   else block_logits[i-1]. Bonus distribution follows the last.
            def target_dist(i: int) -> torch.Tensor:
                if pending is not None:
                    return block_logits[i]
                return target_logits if i == 0 else block_logits[i - 1]

            # 3) Accept the longest agreeing prefix, then one replacement.
            accepted = 0
            replacement: int | None = None
            for i, token in enumerate(candidates):
                dist = target_dist(i)
                if params.temperature <= 0:
                    want = greedy_token(dist)
                    if want == token:
                        accepted += 1
                    else:
                        replacement = want
                        break
                else:
                    p = temperature_probs(dist, params.temperature)
                    q = draft_probs[i]
                    r = float(torch.rand((), generator=generator))
                    if r < acceptance_probability(p, q, token):
                        accepted += 1
                    else:
                        replacement = sample_from_probs(residual_distribution(p, q), generator)
                        break
                if token == eos:
                    break
            stats.accepted += accepted

            # 4) Commit accepted tokens plus one correction/bonus token.
            new_tokens = list(candidates[:accepted])
            last_is_eos = bool(new_tokens) and new_tokens[-1] == eos
            if replacement is None and not last_is_eos and len(committed) + accepted < params.max_new_tokens:
                bonus_dist = block_logits[offset + len(candidates) - 1]
                replacement = _pick(bonus_dist, params.temperature, generator)
            if replacement is not None and not last_is_eos:
                new_tokens.append(replacement)

            # 5) Crop both caches to the confirmed prefix (pending excluded).
            keep = base + offset + accepted
            target_cache.crop(keep)
            draft_cache.crop(keep)
            pending = new_tokens[-1] if new_tokens else pending

            for token in new_tokens:
                if token == eos:
                    eos_hit = True
                    break
                committed.append(token)
            stats.tokens = len(committed)
            chunk, emitted = _emit_text(models, committed, emitted)
            if chunk:
                yield {"type": "token", "text": chunk}
            yield {"type": "stats", "stats": stats.snapshot()}
            if eos_hit:
                break
    yield {"type": "done", "stopped": stopped, "stats": stats.snapshot()}
