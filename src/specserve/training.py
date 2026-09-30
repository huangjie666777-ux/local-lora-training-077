"""Real CPU LoRA fine-tuning of the 360M target with cooperative cancel.

A fresh copy of the frozen base target is wrapped with PEFT LoRA; original
base weights and the 135M draft are never updated. Each optimizer step
performs one backward/optimizer update; cancellation is observed only at
step boundaries so an in-flight step always finishes.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import torch
from peft import LoraConfig, get_peft_model

from .data import EncodedSample

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

LIMITS = {
    "steps": (1, 200),
    "learning_rate": (1e-6, 1.0),
    "seed": (0, 2**31 - 1),
    "max_length": (32, 512),
}
DEFAULTS = {"steps": 10, "learning_rate": 2e-4, "seed": 0, "max_length": 256}


def clamp_limits() -> dict:
    return {"limits": LIMITS, "defaults": DEFAULTS}


@dataclass
class TrainConfig:
    name: str
    steps: int = DEFAULTS["steps"]
    learning_rate: float = DEFAULTS["learning_rate"]
    seed: int = DEFAULTS["seed"]
    max_length: int = DEFAULTS["max_length"]

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "max_length": self.max_length,
        }


def validate_train_config(cfg: TrainConfig) -> list[str]:
    errors: list[str] = []
    for key in ("steps", "learning_rate", "seed", "max_length"):
        value = getattr(cfg, key)
        lo, hi = LIMITS[key]
        if not lo <= value <= hi:
            errors.append(f"{key}={value} outside [{lo}, {hi}]")
    return errors


@dataclass
class TrainStatus:
    name: str
    state: str = "running"  # running | completed | cancelled | failed
    configured_steps: int = 0
    actual_steps: int = 0
    last_loss: float | None = None
    losses: list[float] = field(default_factory=list)
    reason: str = ""
    error: str = ""
    created_at: str = ""
    sample_summary: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "name": self.name,
                "state": self.state,
                "configured_steps": self.configured_steps,
                "actual_steps": self.actual_steps,
                "last_loss": self.last_loss,
                "losses": list(self.losses),
                "reason": self.reason,
                "error": self.error,
                "created_at": self.created_at,
                "sample_summary": dict(self.sample_summary),
            }


def build_lora_model(base_target):
    """Wrap a fresh frozen base target; only LoRA params require grad."""
    peft_model = get_peft_model(
        base_target,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=list(LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        ),
    )
    peft_model.train()
    return peft_model


def _collate(sample: EncodedSample, device="cpu"):
    """Single-sample batch: labels mask prompt and padding (none here)."""
    input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([sample.labels], dtype=torch.long, device=device)
    return input_ids, labels


def run_training(
    peft_model,
    samples: list[EncodedSample],
    cfg: TrainConfig,
    status: TrainStatus,
    is_cancelled: Callable[[], bool],
) -> None:
    """Run up to ``cfg.steps`` optimizer steps, cycling over samples.

    Cancellation takes effect after the current optimizer step completes.
    ``peft_model`` is owned solely by the training worker and discarded
    afterwards, so a cancel/failure cannot alter the serving model.
    """
    torch.manual_seed(cfg.seed)
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    cursor = 0
    with status.lock:
        status.configured_steps = cfg.steps
    for step in range(1, cfg.steps + 1):
        sample = samples[cursor % len(samples)]
        cursor += 1
        input_ids, labels = _collate(sample)
        optimizer.zero_grad(set_to_none=True)
        outputs = peft_model(input_ids=input_ids)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = loss_fn(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        with status.lock:
            status.actual_steps = step
            status.last_loss = loss_value
            status.losses.append(loss_value)
        if is_cancelled():
            with status.lock:
                status.state = "cancelled"
                status.reason = "cancelled after optimizer step %d" % step
            return
    with status.lock:
        status.state = "completed"
        status.reason = "configured_steps_reached"
