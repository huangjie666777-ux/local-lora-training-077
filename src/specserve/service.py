"""Request lifecycle and the single shared engine mutex.

Generation, training and adapter switching share one non-reentrant lock:
whatever holds it owns the models. Invalid requests are validated before
the lock is touched, so they never occupy the slot. Training runs on a
background worker thread; cancellation is observed only after the current
optimizer step finishes. Failed or cancelled training never changes the
adapter version used for inference.
"""
from __future__ import annotations

import copy
import queue
import threading
from datetime import datetime, timezone
from typing import Iterator

from . import engine
from .data import prepare_batch
from .engine import EngineParams
from .modeling import AdapterError, ModelRegistry
from .training import (
    TrainConfig,
    TrainStatus,
    build_lora_model,
    run_training,
    validate_train_config,
)


class BusyError(Exception):
    """Raised when the model mutex is already held."""


class TrainingError(RuntimeError):
    """Training request rejected (bad config, no samples, ...)."""


class GenerationService:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._train_thread: threading.Thread | None = None
        self._train_cancel = threading.Event()
        self._train_status: TrainStatus | None = None
        self._train_lock = threading.Lock()

    # -- generation --------------------------------------------------------

    def stop(self) -> bool:
        """Ask the active generation to stop after its current forward."""
        if self._mutex.locked():
            self._stop.set()
            return True
        return False

    def stream(self, params: EngineParams) -> Iterator[dict]:
        if not self._mutex.acquire(blocking=False):
            raise BusyError("engine is busy (generation, training or adapter switch)")
        self._stop.clear()
        return self._iterate(params)

    def _iterate(self, params: EngineParams) -> Iterator[dict]:
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                # Snapshot the pair under the lock: a request pins the exact
                # adapter version for its whole lifetime; caches are fresh.
                models = self.registry.get()
                pinned = copy.copy(models)
                for event in engine.generate(pinned, params, self._stop.is_set):
                    events.put(event)
            except Exception as exc:  # surfaced to the client, never silent
                events.put({"type": "error", "message": str(exc)})
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                yield event
        finally:
            self._stop.set()
            thread.join()
            self._mutex.release()

    # -- training ----------------------------------------------------------

    def training_status(self) -> dict | None:
        with self._train_lock:
            status = self._train_status
        return status.snapshot() if status is not None else None

    def start_training(
        self, name: str, samples, steps: int, learning_rate: float, seed: int, max_length: int
    ) -> dict:
        cfg = TrainConfig(
            name=name,
            steps=int(steps),
            learning_rate=float(learning_rate),
            seed=int(seed),
            max_length=int(max_length),
        )
        errors = validate_train_config(cfg)
        if errors:
            raise TrainingError("; ".join(errors))
        # Full batch validation happens before the mutex is ever taken.
        pair = self.registry.get()
        encoded, summary = prepare_batch(samples, pair.tokenizer, cfg.max_length)
        with self._train_lock:
            busy = self._train_thread is not None and self._train_thread.is_alive()
        if busy or not self._mutex.acquire(blocking=False):
            raise BusyError("engine is busy (generation, training or adapter switch)")
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        status = TrainStatus(
            name=name,
            configured_steps=cfg.steps,
            created_at=created_at,
            sample_summary=summary.as_dict(),
        )
        cancel = threading.Event()
        thread = threading.Thread(
            target=self._train_worker,
            args=(cfg, encoded, summary.as_dict(), status, cancel),
            daemon=True,
        )
        with self._train_lock:
            self._train_thread = thread
            self._train_cancel = cancel
            self._train_status = status
        thread.start()
        return status.snapshot()

    def _train_worker(self, cfg, encoded, summary_dict, status: TrainStatus, cancel) -> None:
        peft_model = None
        try:
            peft_model = build_lora_model(self.registry._load_base_target())
            run_training(peft_model, encoded, cfg, status, cancel.is_set)
            with status.lock:
                finished = status.state
            if finished == "cancelled" or cancel.is_set():
                with status.lock:
                    status.state = "cancelled"
                    if not status.reason:
                        status.reason = "cancelled before adapter activation"
                return
            if finished == "completed":
                self.registry.save_adapter(
                    peft_model,
                    cfg.name,
                    cfg.as_dict(),
                    summary_dict,
                    status.created_at,
                    status.last_loss if status.last_loss is not None else 0.0,
                    status.actual_steps,
                )
                # Activate the freshly saved adapter; serving version changes
                # only here, after a fully successful save.
                self.registry.activate_adapter(cfg.name)
        except AdapterError as exc:
            with status.lock:
                status.state = "failed"
                status.error = str(exc)
                status.reason = "adapter_save_or_switch_failed"
        except Exception as exc:
            with status.lock:
                status.state = "failed"
                status.error = str(exc)
                status.reason = "training_error"
        finally:
            del peft_model
            self._mutex.release()

    def cancel_training(self) -> bool:
        with self._train_lock:
            thread = self._train_thread
            cancel = self._train_cancel
        if thread is not None and thread.is_alive():
            cancel.set()
            return True
        return False

    # -- adapters ----------------------------------------------------------

    def list_adapters(self) -> dict:
        pair = self.registry.get()
        return {"active": pair.adapter, "adapters": self.registry.list_adapters()}

    def activate_adapter(self, name: str) -> dict:
        if not self._mutex.acquire(blocking=False):
            raise BusyError("engine is busy (generation, training or adapter switch)")
        try:
            return self.registry.activate_adapter(name)
        finally:
            self._mutex.release()

    def restore_base(self) -> None:
        if not self._mutex.acquire(blocking=False):
            raise BusyError("engine is busy (generation, training or adapter switch)")
        try:
            self.registry.restore_base()
        finally:
            self._mutex.release()
