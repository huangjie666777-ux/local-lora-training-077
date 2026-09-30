"""Named LoRA adapter store: atomic save, listing, compatibility checks.

Adapters live under adapters/<name>/ with the PEFT files plus a
metadata.json recording the base-model identity, the LoRA/training
configuration and a digest of the training samples. Saves are atomic:
files land in a temporary sibling directory which is then renamed, so a
crash never leaves a half-written adapter behind.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = ROOT / "adapters"
METADATA_FILE = "metadata.json"
REQUIRED_PEFT_FILES = ("adapter_config.json", "adapter_model.safetensors")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")


class AdapterError(Exception):
    """Raised for unknown, corrupt or incompatible adapters."""


def valid_name(name: str) -> bool:
    return bool(NAME_PATTERN.match(name or ""))


class AdapterStore:
    def __init__(self, directory: Path = ADAPTERS_DIR, base_identity: dict | None = None):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Base-model identity captured from the live target config, e.g.
        # {"repository": ..., "hidden_size": 960, "vocab_size": 49152}.
        self._base_identity = dict(base_identity or {})

    @property
    def directory(self) -> Path:
        return self._dir

    def adapter_path(self, name: str) -> Path:
        if not valid_name(name):
            raise AdapterError(f"invalid adapter name: {name!r}")
        return self._dir / name

    def list(self) -> list[dict]:
        """Metadata of every well-formed adapter, corrupt ones flagged."""
        entries: list[dict] = []
        for child in sorted(self._dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            meta_path = child / METADATA_FILE
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                missing = [f for f in REQUIRED_PEFT_FILES if not (child / f).exists()]
                meta["usable"] = not missing
                if missing:
                    meta["problem"] = "missing files: " + ", ".join(missing)
            except Exception:
                meta = {"name": child.name, "usable": False, "problem": "corrupt metadata"}
            meta.setdefault("name", child.name)
            entries.append(meta)
        return entries

    def check_compatible(self, name: str) -> dict:
        """Return metadata if the adapter exists and matches the base model."""
        path = self.adapter_path(name)
        if not path.is_dir():
            raise AdapterError(f"unknown adapter: {name}")
        for required in (METADATA_FILE, *REQUIRED_PEFT_FILES):
            if not (path / required).is_file():
                raise AdapterError(f"adapter {name!r} is corrupt: missing {required}")
        try:
            meta = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
            with open(path / "adapter_config.json", encoding="utf-8") as fh:
                json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise AdapterError(f"adapter {name!r} is corrupt: {exc}") from exc
        base = meta.get("base_model", {})
        for key, expected in self._base_identity.items():
            actual = base.get(key)
            if actual is not None and actual != expected:
                raise AdapterError(
                    f"adapter {name!r} targets a different base model "
                    f"({key}={actual!r}, expected {expected!r})"
                )
        return meta

    def save_atomic(self, peft_model, name: str, metadata: dict) -> Path:
        """Save adapter weights and metadata, then atomically rename in place."""
        target = self.adapter_path(name)
        if target.exists():
            raise AdapterError(f"adapter {name!r} already exists")
        tmp = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=self._dir))
        try:
            peft_model.save_pretrained(tmp)
            meta = dict(metadata)
            meta.setdefault("name", name)
            meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            meta.setdefault("base_model", self._base_identity)
            (tmp / METADATA_FILE).write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.rename(tmp, target)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return target
