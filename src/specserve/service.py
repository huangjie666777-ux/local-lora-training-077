"""Request lifecycle: one slot shared by generation, training and switching.

A single non-blocking lock serializes every heavy operation. Invalid
requests are rejected before touching the slot; failed or cancelled
jobs always clean up their state and never change the active inference
version.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Iterator

from . import engine
from .adapters import AdapterStore
from .dataset import Features, Sample, summarize
from .engine import EngineParams
from .modeling import ModelRegistry, base_identity
from .training import Cancellation, TrainParams, load_trainable_base, train_lora


class BusyError(Exception):
    """Raised when another generation/training/switch operation is active."""


class GenerationService:
    def __init__(
        self,
        registry: ModelRegistry | None = None,
        store: AdapterStore | None = None,
    ):
        self.registry = registry or ModelRegistry()
        if store is None:
            target_dir = getattr(self.registry, "target_dir", None)
            identity = base_identity(target_dir) if target_dir else {}
            store = AdapterStore(base_identity=identity)
        self.store = store
        self._slot = threading.Lock()
        self._stop = threading.Event()
        self._status_lock = threading.Lock()
        self._cancel: Cancellation | None = None
        self._train_status: dict = {"state": "idle"}

    # -- generation -----------------------------------------------------

    def stop(self) -> bool:
        """Ask the active request (if any) to stop after its current forward."""
        if self._slot.locked():
            self._stop.set()
            return True
        return False

    def stream(self, params: EngineParams) -> Iterator[dict]:
        """Return an event iterator; raises BusyError if the slot is taken.

        The slot is acquired eagerly so invalid/busy requests fail before
        the HTTP response starts, and only released once the worker thread
        has finished its in-flight forward pass.
        """
        if not self._slot.acquire(blocking=False):
            raise BusyError("another generation request is active")
        self._stop.clear()
        return self._iterate(params)

    def _iterate(self, params: EngineParams) -> Iterator[dict]:
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                models = self.registry.get()
                for event in engine.generate(models, params, self._stop.is_set):
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
            # Client disconnected or stream closed: stop cooperatively and
            # wait for the current forward pass to finish before freeing.
            self._stop.set()
            thread.join()
            self._slot.release()

    # -- training -------------------------------------------------------

    def train_status(self) -> dict:
        with self._status_lock:
            return dict(self._train_status)

    def _set_status(self, **updates) -> None:
        with self._status_lock:
            self._train_status.update(updates)

    def start_train(
        self,
        params: TrainParams,
        samples: list[Sample],
        features: list[Features],
    ) -> dict:
        """Launch async LoRA training; raises BusyError if the slot is taken.

        Validation happens before this call, so an invalid request never
        occupies the slot.
        """
        if not self._slot.acquire(blocking=False):
            raise BusyError("another generation or training request is active")
        cancel = Cancellation()
        self._cancel = cancel
        self._set_status(
            state="running",
            adapter=params.name,
            step=0,
            steps=params.steps,
            loss=None,
            reason=None,
            sample_count=len(samples),
        )

        def on_progress(step: int, loss: float) -> None:
            self._set_status(step=step, loss=loss)

        def worker():
            try:
                model = load_trainable_base(self.registry.target_dir)
                result = train_lora(
                    model,
                    features,
                    params,
                    pad_token_id=self.registry.get().eos_token_id,
                    should_cancel=cancel,
                    on_progress=on_progress,
                )
                if result.reason == "completed" and result.peft_model is not None:
                    metadata = {
                        "lora": params.lora_config_dict(),
                        "training": {
                            "steps": result.steps_done,
                            "learning_rate": params.learning_rate,
                            "seed": params.seed,
                            "max_length": params.max_length,
                            "batch_size": params.batch_size,
                            "final_loss": result.final_loss,
                            "reason": result.reason,
                        },
                        "samples": summarize(samples, features),
                    }
                    self.store.save_atomic(result.peft_model, params.name, metadata)
                # Cancelled/failed runs persist nothing and leave the
                # active inference version untouched.
                self._set_status(
                    state=result.reason,
                    step=result.steps_done,
                    loss=result.final_loss,
                    reason=result.error or result.reason,
                )
            except Exception as exc:
                self._set_status(state="failed", reason=str(exc))
            finally:
                self._cancel = None
                self._slot.release()

        threading.Thread(target=worker, daemon=True).start()
        return self.train_status()

    def cancel_train(self) -> bool:
        """Request cancellation; takes effect after the current optimizer step."""
        cancel = self._cancel
        if cancel is not None and self.train_status().get("state") == "running":
            cancel.cancel()
            return True
        return False

    # -- adapter versions ------------------------------------------------

    def versions(self) -> dict:
        return {"active": self.registry.active_adapter(), "adapters": self.store.list()}

    def select_adapter(self, name: str | None) -> dict:
        """Activate an adapter or restore the base model (name=None).

        Busy operations reject the switch; corrupt or incompatible
        adapters raise AdapterError and the old version stays active.
        """
        if not self._slot.acquire(blocking=False):
            raise BusyError("another generation or training request is active")
        try:
            if name is None:
                self.registry.set_adapter(None)
            else:
                meta = self.store.check_compatible(name)
                self.registry.set_adapter(name, self.store.adapter_path(name))
                _ = meta
            return self.versions()
        finally:
            self._slot.release()
