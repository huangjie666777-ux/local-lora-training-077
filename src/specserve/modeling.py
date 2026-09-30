"""Model loading: shared tokenizer plus draft/target causal LMs on CPU.

The 135M draft and the frozen 360M base target are always loaded from
``models/`` and never modified. Domain adaptation only swaps the target for
a PEFT-wrapped copy carrying a named LoRA adapter; the draft stays base.
Adapters live under ``adapters/`` and are persisted atomically.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = ROOT / "models" / "draft"
TARGET_DIR = ROOT / "models" / "target"
ADAPTERS_DIR = ROOT / "adapters"
BASE_NAME = "SmolLM2-360M"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
META_FILENAME = "adapter_meta.json"


@dataclass
class ModelPair:
    tokenizer: object
    draft: object
    target: object
    eos_token_id: int
    adapter: str | None = None


def base_model_identity(target_dir: Path = TARGET_DIR) -> dict:
    """Stable identity of the local base target used for an adapter."""
    config_bytes = (Path(target_dir) / "config.json").read_bytes()
    digest = hashlib.sha256(config_bytes).hexdigest()
    return {"base_name": BASE_NAME, "config_sha256": digest}


class AdapterError(RuntimeError):
    """Adapter metadata, compatibility or weight file problem."""


class ModelRegistry:
    """Loads both models lazily and manages named LoRA adapters.

    Callers must hold the service mutex while swapping the target; the
    registry itself only guards lazy initial loading.
    """

    def __init__(
        self,
        draft_dir: Path = DRAFT_DIR,
        target_dir: Path = TARGET_DIR,
        adapters_dir: Path = ADAPTERS_DIR,
    ):
        self._draft_dir = Path(draft_dir)
        self._target_dir = Path(target_dir)
        self._adapters_dir = Path(adapters_dir)
        self._load_lock = threading.Lock()
        self._pair: ModelPair | None = None

    # -- core models -------------------------------------------------------

    def get(self) -> ModelPair:
        if self._pair is None:
            with self._load_lock:
                if self._pair is None:
                    self._pair = self._load_pair(self._load_base_target())
        return self._pair

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self._target_dir)

    def _load_draft(self):
        draft = AutoModelForCausalLM.from_pretrained(
            self._draft_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        draft.eval()
        for param in draft.parameters():
            param.requires_grad_(False)
        return draft

    def _load_base_target(self):
        target = AutoModelForCausalLM.from_pretrained(
            self._target_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        target.eval()
        for param in target.parameters():
            param.requires_grad_(False)
        return target

    def _load_pair(self, target, adapter: str | None = None) -> ModelPair:
        tokenizer = self._load_tokenizer()
        draft = self._load_draft()
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = target.config.eos_token_id
        return ModelPair(
            tokenizer=tokenizer,
            draft=draft,
            target=target,
            eos_token_id=int(eos),
            adapter=adapter,
        )

    def _fresh_pair(self, target, adapter: str | None) -> ModelPair:
        """Replace draft/target wholesale; old objects are garbage collected."""
        pair = self._load_pair(target, adapter)
        self._pair = pair
        return pair

    # -- adapter store -----------------------------------------------------

    def list_adapters(self) -> list[dict]:
        if not self._adapters_dir.exists():
            return []
        out: list[dict] = []
        for path in sorted(self._adapters_dir.iterdir()):
            meta_path = path / META_FILENAME
            if not (path.is_dir() and meta_path.exists()):
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append(
                {
                    "name": meta.get("name", path.name),
                    "created_at": meta.get("created_at"),
                    "base_name": meta.get("base", {}).get("base_name"),
                    "steps": meta.get("config", {}).get("steps"),
                    "max_length": meta.get("config", {}).get("max_length"),
                    "sample_summary": meta.get("sample_summary", {}),
                }
            )
        return out

    def _adapter_dir(self, name: str) -> Path:
        if not name or len(name) > 32:
            raise AdapterError("adapter name must be 1-32 characters")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        if any(ch not in allowed for ch in name):
            raise AdapterError(
                "adapter name may only contain letters, digits, '-' and '_'"
            )
        return self._adapters_dir / name

    def save_adapter(
        self,
        peft_model,
        name: str,
        config: dict,
        sample_summary: dict,
        created_at: str,
        train_loss: float,
        actual_steps: int,
    ) -> dict:
        """Write adapter weights and metadata atomically (tmp dir, rename)."""
        target_dir = self._adapter_dir(name)
        if target_dir.exists():
            raise AdapterError(f"adapter {name!r} already exists")
        identity = base_model_identity(self._target_dir)
        meta = {
            "name": name,
            "created_at": created_at,
            "base": identity,
            "config": config,
            "sample_summary": sample_summary,
            "train_loss": train_loss,
            "actual_steps": actual_steps,
        }
        self._adapters_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=self._adapters_dir))
        try:
            peft_model.save_pretrained(str(tmp_dir))
            (tmp_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))
            os.replace(tmp_dir, target_dir)  # atomic on the same filesystem
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return meta

    def _read_meta(self, name: str) -> tuple[Path, dict]:
        path = self._adapter_dir(name)
        meta_path = path / META_FILENAME
        if not path.is_dir() or not meta_path.exists():
            raise AdapterError(f"adapter {name!r} not found")
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterError(f"adapter {name!r} metadata is corrupt: {exc}")
        if not (path / ADAPTER_CONFIG_FILENAME).exists():
            raise AdapterError(f"adapter {name!r} is missing {ADAPTER_CONFIG_FILENAME}")
        return path, meta

    def _check_base(self, meta: dict, name: str) -> None:
        if meta.get("base") != base_model_identity(self._target_dir):
            raise AdapterError(
                f"adapter {name!r} was trained against a different base model"
            )

    # -- live adapter switching -------------------------------------------

    def activate_adapter(self, name: str) -> dict:
        """Load a saved adapter onto a fresh base target.

        The previously served pair stays referenced until the new one loads
        successfully, so corrupt or incompatible files keep the old version.
        """
        path, meta = self._read_meta(name)
        self._check_base(meta, name)
        try:
            target = PeftModel.from_pretrained(self._load_base_target(), str(path))
            target.eval()
        except AdapterError:
            raise
        except Exception as exc:  # corrupt weights/config: keep old version
            raise AdapterError(f"adapter {name!r} could not be loaded: {exc}")
        self._fresh_pair(target, name)
        return meta

    def restore_base(self) -> None:
        """Switch back to the untouched 360M base model."""
        self._fresh_pair(self._load_base_target(), None)
