"""Request lifecycle: one active generation, cooperative stop, clean release."""
from __future__ import annotations

import queue
import threading
from typing import Iterator

from . import engine
from .engine import EngineParams
from .modeling import ModelRegistry


class BusyError(Exception):
    """Raised when another generation request is still active."""


class GenerationService:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self._slot = threading.Lock()
        self._stop = threading.Event()

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
