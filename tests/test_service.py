import threading
import time

import pytest
from fastapi.testclient import TestClient

from specserve import app as app_module
from specserve.engine import EngineParams
from specserve.service import BusyError, GenerationService


class FakeRegistry:
    def get(self):
        return object()


def fake_generate(models, params, should_stop):
    for i in range(3):
        while not should_stop():
            time.sleep(0.01)
            break
        yield {"type": "token", "text": f"t{i}"}
    yield {"type": "done", "stopped": should_stop(), "stats": {}}


@pytest.fixture
def service(monkeypatch):
    svc = GenerationService(registry=FakeRegistry())
    monkeypatch.setattr("specserve.service.engine.generate", fake_generate)
    return svc


def params():
    return EngineParams(prompt="hello", max_new_tokens=4)


def test_single_active_request(service):
    stream = service.stream(params())
    next(stream)
    with pytest.raises(BusyError):
        service.stream(params())
    list(stream)
    # Slot freed afterwards.
    assert list(service.stream(params()))


def test_stop_releases_slot(service):
    stream = service.stream(params())
    assert service.stop() is True
    list(stream)
    assert service.stop() is False
    assert list(service.stream(params()))


def test_health():
    client = TestClient(app_module.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_invalid_params_rejected_without_slot(monkeypatch):
    monkeypatch.setattr(app_module, "service", GenerationService(registry=FakeRegistry()))
    client = TestClient(app_module.app)
    bad = client.post("/api/generate", json={"prompt": "", "mode": "speculative"})
    assert bad.status_code == 422
    bad = client.post("/api/generate", json={"prompt": "hi", "mode": "nope"})
    assert bad.status_code == 422
    # Slot must still be free: a valid request can run.
    monkeypatch.setattr("specserve.service.engine.generate", fake_generate)
    ok = client.post("/api/generate", json={"prompt": "hi", "mode": "autoregressive"})
    assert ok.status_code == 200
    assert "t0" in ok.text



# -- LoRA training lifecycle ----------------------------------------------

import types

from specserve.adapters import AdapterStore
from specserve.dataset import Features, Sample
from specserve.training import TrainParams, TrainResult


class FakeRegistryWithEos:
    def get(self):
        return types.SimpleNamespace(eos_token_id=0)

    @property
    def target_dir(self):
        return "unused"

    def active_adapter(self):
        return None

    def set_adapter(self, name, adapter_path=None):
        self.selected = name


def train_inputs():
    return (
        TrainParams(name="t1", steps=5),
        [Sample(line=1, prompt="p", completion="c")],
        [Features(line=1, input_ids=[1, 2, 0], labels=[-100, 2, 0])],
    )


@pytest.fixture
def train_service(monkeypatch, tmp_path):
    svc = GenerationService(
        registry=FakeRegistryWithEos(),
        store=AdapterStore(tmp_path / "adapters", base_identity={}),
    )
    monkeypatch.setattr("specserve.service.engine.generate", fake_generate)
    monkeypatch.setattr("specserve.service.load_trainable_base", lambda _dir: object())
    return svc


def wait_state(svc, state, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if svc.train_status()["state"] == state:
            return True
        time.sleep(0.02)
    return False


def test_training_occupies_slot_and_cancel_cleans_up(train_service, monkeypatch):
    started = threading.Event()

    def fake_train(model, features, params, pad_token_id, should_cancel, on_progress):
        started.set()
        step = 0
        while not should_cancel() and step < params.steps:
            step += 1
            on_progress(step, 1.0)
            time.sleep(0.02)
        return TrainResult(peft_model=None, steps_done=step, final_loss=1.0, reason="cancelled")

    monkeypatch.setattr("specserve.service.train_lora", fake_train)
    svc = train_service
    svc.start_train(*train_inputs())
    assert started.wait(2)
    assert svc.train_status()["state"] == "running"
    # Slot is shared: generation is rejected while training runs.
    with pytest.raises(BusyError):
        svc.stream(params())
    assert svc.cancel_train() is True
    assert wait_state(svc, "cancelled")
    status = svc.train_status()
    assert status["reason"] == "cancelled"
    # Nothing persisted, slot free again, cancel no longer active.
    assert svc.store.list() == []
    assert svc.cancel_train() is False
    assert list(svc.stream(params()))


def test_completed_training_saves_adapter(train_service, monkeypatch):
    class FakePeft:
        def save_pretrained(self, path):
            from pathlib import Path

            Path(path, "adapter_config.json").write_text("{}")
            Path(path, "adapter_model.safetensors").write_bytes(b"x")

    def fake_train(model, features, params, pad_token_id, should_cancel, on_progress):
        on_progress(1, 0.5)
        return TrainResult(peft_model=FakePeft(), steps_done=1, final_loss=0.5, reason="completed")

    monkeypatch.setattr("specserve.service.train_lora", fake_train)
    svc = train_service
    svc.start_train(*train_inputs())
    assert wait_state(svc, "completed")
    entries = svc.store.list()
    assert len(entries) == 1 and entries[0]["name"] == "t1"
    assert entries[0]["training"]["final_loss"] == 0.5
    assert entries[0]["samples"]["sample_count"] == 1


def test_failed_training_keeps_state_clean(train_service, monkeypatch):
    def fake_train(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("specserve.service.train_lora", fake_train)
    svc = train_service
    svc.start_train(*train_inputs())
    assert wait_state(svc, "failed")
    assert "boom" in svc.train_status()["reason"]
    assert svc.store.list() == []
    assert list(svc.stream(params()))  # slot released


def test_train_endpoint_validation_and_busy(monkeypatch, tmp_path):
    svc = GenerationService(
        registry=FakeRegistryWithEos(),
        store=AdapterStore(tmp_path / "adapters", base_identity={}),
    )
    monkeypatch.setattr(app_module, "service", svc)
    client = TestClient(app_module.app)

    bad = client.post("/api/train", json={"dataset": "not json\n{\"prompt\": \"x\"}"})
    assert bad.status_code == 422
    errors = bad.json()["detail"]["errors"]
    assert [e["line"] for e in errors] == [1, 2]
    assert svc.train_status()["state"] == "idle"  # invalid request never occupies

    bad = client.post("/api/train", json={"dataset": "{\"prompt\": \"a\", \"completion\": \"b\"}", "steps": 10**6})
    assert bad.status_code == 422

    # Unknown adapter keeps the current version.
    resp = client.post("/api/adapters/select", json={"name": "ghost"})
    assert resp.status_code == 422
    assert svc.versions()["active"] is None
    # Restoring base always works.
    assert client.post("/api/adapters/select", json={"name": "base"}).status_code == 200
