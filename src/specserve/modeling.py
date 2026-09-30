"""Model loading: shared tokenizer plus draft/target causal LMs on CPU."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = ROOT / "models" / "draft"
TARGET_DIR = ROOT / "models" / "target"


@dataclass
class ModelPair:
    tokenizer: object
    draft: object
    target: object
    eos_token_id: int


class ModelRegistry:
    """Lazily loads both models exactly once, thread-safe."""

    def __init__(self, draft_dir: Path = DRAFT_DIR, target_dir: Path = TARGET_DIR):
        self._draft_dir = Path(draft_dir)
        self._target_dir = Path(target_dir)
        self._lock = threading.Lock()
        self._pair: ModelPair | None = None

    def get(self) -> ModelPair:
        if self._pair is None:
            with self._lock:
                if self._pair is None:
                    self._pair = self._load()
        return self._pair

    def _load(self) -> ModelPair:
        tokenizer = AutoTokenizer.from_pretrained(self._target_dir)
        common = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True)
        draft = AutoModelForCausalLM.from_pretrained(self._draft_dir, **common)
        target = AutoModelForCausalLM.from_pretrained(self._target_dir, **common)
        draft.eval()
        target.eval()
        for model in (draft, target):
            for param in model.parameters():
                param.requires_grad_(False)
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = target.config.eos_token_id
        return ModelPair(tokenizer=tokenizer, draft=draft, target=target, eos_token_id=int(eos))

