"""CPU LoRA training: real gradient steps on the target model, cancellable.

The trainer wraps a privately loaded copy of the base target model with
a PEFT LoRA adapter (the shared inference weights are never touched),
runs a fixed number of optimizer steps over the validated features and
reports progress after every step. Cancellation is cooperative: it
takes effect once the in-flight optimizer step finishes.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from .dataset import Features, collate

# Server-side guardrails; the API layer enforces the same ranges.
STEPS_RANGE = (1, 100)
LEARNING_RATE_RANGE = (1e-5, 5e-3)
MAX_LENGTH_RANGE = (32, 512)
SEED_RANGE = (0, 2**31 - 1)
DEFAULTS = dict(steps=10, learning_rate=2e-4, seed=0, max_length=192)
LORA_TARGETS = ["q_proj", "v_proj"]


@dataclass
class TrainParams:
    name: str
    steps: int = DEFAULTS["steps"]
    learning_rate: float = DEFAULTS["learning_rate"]
    seed: int = DEFAULTS["seed"]
    max_length: int = DEFAULTS["max_length"]
    batch_size: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def lora_config_dict(self) -> dict:
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(LORA_TARGETS),
        }


@dataclass
class TrainResult:
    peft_model: object | None
    steps_done: int
    final_loss: float | None
    reason: str  # "completed" | "cancelled" | "failed"
    losses: list[float] = field(default_factory=list)
    error: str | None = None


def load_trainable_base(target_dir) -> torch.nn.Module:
    """Load a private fp32 copy of the base target model for training."""
    model = AutoModelForCausalLM.from_pretrained(
        target_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    return model


def train_lora(
    base_model: torch.nn.Module,
    features: list[Features],
    params: TrainParams,
    pad_token_id: int,
    should_cancel: Callable[[], bool],
    on_progress: Callable[[int, float], None],
) -> TrainResult:
    """Run the LoRA training loop; returns the adapted model on completion.

    Cancel/failure yields a result with peft_model=None so callers never
    persist or activate a partial adapter.
    """
    if not features:
        raise ValueError("train_lora requires at least one feature")
    torch.manual_seed(params.seed)
    config = LoraConfig(
        r=params.lora_r,
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        target_modules=list(LORA_TARGETS),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, config)
    peft_model.train()
    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad), lr=params.learning_rate
    )
    order = torch.randperm(len(features), generator=torch.Generator().manual_seed(params.seed))
    losses: list[float] = []
    steps_done = 0
    reason = "completed"
    try:
        for step in range(params.steps):
            if should_cancel():
                reason = "cancelled"
                break
            start = (step * params.batch_size) % len(features)
            idx = [int(order[(start + k) % len(features)]) for k in range(params.batch_size)]
            batch = collate([features[k] for k in idx], pad_token_id)
            out = peft_model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = round(float(loss.detach()), 4)
            losses.append(value)
            steps_done += 1
            on_progress(step + 1, value)
    except Exception as exc:
        return TrainResult(
            peft_model=None,
            steps_done=steps_done,
            final_loss=losses[-1] if losses else None,
            reason="failed",
            losses=losses,
            error=str(exc),
        )
    if reason != "completed":
        return TrainResult(
            peft_model=None,
            steps_done=steps_done,
            final_loss=losses[-1] if losses else None,
            reason=reason,
            losses=losses,
        )
    peft_model.eval()
    return TrainResult(
        peft_model=peft_model,
        steps_done=steps_done,
        final_loss=losses[-1] if losses else None,
        reason=reason,
        losses=losses,
    )


class Cancellation:
    """Thread-safe cancel token shared between API and trainer thread."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def __call__(self) -> bool:
        return self._event.is_set()
