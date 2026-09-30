"""Model loading: shared tokenizer, draft/target causal LMs, LoRA switching.

The draft (SmolLM2-135M) and the base target (SmolLM2-360M) weights are
loaded once and never mutated. Named LoRA adapters are attached to a
single PEFT wrapper around the base target; switching versions only
toggles adapter layers, and a failed load leaves the previous version
active.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = ROOT / "models" / "draft"
TARGET_DIR = ROOT / "models" / "target"
MANIFEST = ROOT / "models" / "manifest.json"


@dataclass
class ModelPair:
    tokenizer: object
    draft: object
    target: object
    eos_token_id: int
    adapter: str | None = None  # active adapter name, None = base model


def base_identity(target_dir: Path = TARGET_DIR) -> dict:
    """Identity recorded in adapter metadata and checked on load."""
    with open(Path(target_dir) / "config.json", encoding="utf-8") as fh:
        config = json.load(fh)
    repository = "unknown"
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        repository = manifest.get("target", {}).get("repository", repository)
    except Exception:
        pass
    return {
        "repository": repository,
        "hidden_size": config.get("hidden_size"),
        "vocab_size": config.get("vocab_size"),
    }


class ModelRegistry:
    """Lazily loads both models exactly once, thread-safe."""

    def __init__(self, draft_dir: Path = DRAFT_DIR, target_dir: Path = TARGET_DIR):
        self._draft_dir = Path(draft_dir)
        self._target_dir = Path(target_dir)
        self._lock = threading.Lock()
        self._tokenizer = None
        self._draft = None
        self._base_target = None
        self._peft = None  # single PEFT wrapper holding every loaded adapter
        self._active: str | None = None
        self._eos_token_id: int | None = None

    @property
    def target_dir(self) -> Path:
        return self._target_dir

    def get(self) -> ModelPair:
        """Current inference pair; called once per request so the version
        is pinned for the whole request and caches are never shared."""
        self._ensure_loaded()
        with self._lock:
            target = self._peft if self._active is not None else self._base_target
            return ModelPair(
                tokenizer=self._tokenizer,
                draft=self._draft,
                target=target,
                eos_token_id=self._eos_token_id,
                adapter=self._active,
            )

    def active_adapter(self) -> str | None:
        return self._active

    def set_adapter(self, name: str | None, adapter_path: Path | None = None) -> None:
        """Switch the target model to a named adapter or back to base.

        Raises on load failure; the previously active version is kept.
        """
        from peft import PeftModel

        self._ensure_loaded()
        with self._lock:
            if name is None:
                if self._peft is not None:
                    self._peft.disable_adapter_layers()
                self._active = None
                return
            if self._peft is None:
                peft = PeftModel.from_pretrained(
                    self._base_target, str(adapter_path), adapter_name=name
                )
                self._peft = peft
            else:
                self._peft.enable_adapter_layers()
                if name not in self._peft.peft_config:
                    self._peft.load_adapter(str(adapter_path), adapter_name=name)
                self._peft.set_adapter(name)
            self._active = name

    def _ensure_loaded(self) -> None:
        if self._base_target is None:
            with self._lock:
                if self._base_target is None:
                    self._load()

    def _load(self) -> None:
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
        self._tokenizer = tokenizer
        self._draft = draft
        self._base_target = target
        self._eos_token_id = int(eos)
