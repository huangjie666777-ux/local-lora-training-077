import threading
import types
import time

import pytest
from fastapi.testclient import TestClient

from specserve import app as app_module
from specserve.data import DataValidationError, Sample
from specserve.engine import EngineParams
from specserve.service import BusyError, GenerationService, TrainingError


class FakeRegistry:
    def __init__(self):
        self.pair = types.SimpleNamespace(adapter=None, tokenizer=object())
        self.adapters = []
        self.active = None
        self.restored = False

    def get(self):
        return self.pair

    def list_adapters(self):
        return self.adapters

    def activate_adapter(self, name):
        self.active = name
        return {"name": name}

    def restore_base(self):
        self.restored = True
        self.active = None

    def _load_base_target(self):
        return object()

    def save_adapter(self, *args, **kwargs):
        return {}


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
    monkeypatch.setattr("specserve.service.engine.generate", fake_generate)
    ok = client.post("/api/generate", json={"prompt": "hi", "mode": "autoregressive"})
    assert ok.status_code == 200
    assert "t0" in ok.text


def test_invalid_training_does_not_occupy_slot(monkeypatch):
    monkeypatch.setattr(app_module, "service", GenerationService(registry=FakeRegistry()))
    client = TestClient(app_module.app)
    # bad JSONL -> 400 before mutex
    r = client.post("/api/training/start", json={"name": "v", "jsonl": "{broken"})
    assert r.status_code == 400
    # out-of-range hyperparams -> 400
    r = client.post("/api/training/start", json={
        "name": "v", "jsonl": '{"prompt":"p","completion":"c"}', "steps": 9999})
    assert r.status_code == 400
    # generation still served -> slot free
    monkeypatch.setattr("specserve.service.engine.generate", fake_generate)
    ok = client.post("/api/generate", json={"prompt": "hi", "mode": "autoregressive"})
    assert ok.status_code == 200


class FakeStatus:
    def __init__(self):
        self.state = "running"
        self.actual_steps = 0
        self.last_loss = None
        self.losses = []
        self.reason = ""
        self.error = ""
        self.configured_steps = 3
        self.created_at = "now"
        self.sample_summary = {}
        self.name = "v"
        self.lock = threading.Lock()

    def snapshot(self):
        return {
            "state": self.state, "actual_steps": self.actual_steps,
            "last_loss": self.last_loss, "losses": list(self.losses),
            "reason": self.reason, "error": self.error,
            "configured_steps": self.configured_steps, "created_at": "now",
            "sample_summary": {}, "name": "v",
        }


def test_training_cancel_after_step_and_mutex(service, monkeypatch):
    import specserve.service as svc_mod

    samples = [Sample("p one", "c one"), Sample("p two", "c two")]

    class FakeEncoded:
        input_ids = [1, 2, 3]
        labels = [-100, 2, 3]
        answer_tokens = 2

    monkeypatch.setattr(
        "specserve.service.prepare_batch",
        lambda samples, tok, maxlen: ([FakeEncoded(), FakeEncoded()],
                                      types.SimpleNamespace(as_dict=lambda: {})),
    )

    def fake_build(base):
        return object()

    started = threading.Event()

    def fake_run(model, encoded, cfg, status, is_cancelled):
        started.set()
        for step in range(1, cfg.steps + 1):
            time.sleep(0.05)
            with status.lock:
                status.actual_steps = step
                status.last_loss = 1.0 / step
                status.losses.append(1.0 / step)
            if is_cancelled():
                status.state = "cancelled"
                status.reason = f"cancelled after optimizer step {step}"
                return
        status.state = "completed"
        status.reason = "configured_steps_reached"

    monkeypatch.setattr(svc_mod, "build_lora_model", fake_build)
    monkeypatch.setattr(svc_mod, "run_training", fake_run)
    monkeypatch.setattr(service.registry, "save_adapter", lambda *a, **k: {})
    monkeypatch.setattr(service.registry, "activate_adapter", lambda name: {})

    # patch TrainStatus construction to use our fake-compatible real object
    from specserve.training import TrainStatus as RealStatus

    status0 = service.start_training(
        "v", samples, steps=5, learning_rate=2e-4, seed=0, max_length=64
    )
    assert status0["state"] == "running"
    assert started.wait(2)
    # busy: second training rejected with 409 semantics
    with pytest.raises(BusyError):
        service.start_training("w", samples, steps=1, learning_rate=2e-4, seed=0, max_length=64)
    # and generation is rejected too
    with pytest.raises(BusyError):
        service.stream(params())
    assert service.cancel_training() is True
    deadline = time.time() + 5
    while time.time() < deadline:
        snap = service.training_status()
        if snap["state"] != "running":
            break
        time.sleep(0.05)
    snap = service.training_status()
    assert snap["state"] == "cancelled"
    assert snap["actual_steps"] >= 1
    assert "step" in snap["reason"]
    # mutex released after cancel cleanup
    assert list(service.stream(params()))


def test_failed_training_keeps_active_version(service, monkeypatch):
    import specserve.service as svc_mod

    samples = [Sample("p", "c")]

    class FakeEncoded:
        input_ids = [1, 2, 3]
        labels = [-100, 2, 3]
        answer_tokens = 2

    monkeypatch.setattr(
        "specserve.service.prepare_batch",
        lambda samples, tok, maxlen: ([FakeEncoded()],
                                      types.SimpleNamespace(as_dict=lambda: {})),
    )
    monkeypatch.setattr(svc_mod, "build_lora_model", lambda base: object())

    def boom(model, encoded, cfg, status, is_cancelled):
        raise RuntimeError("boom")

    monkeypatch.setattr(svc_mod, "run_training", boom)
    service.start_training("v", samples, steps=1, learning_rate=2e-4, seed=0, max_length=64)
    deadline = time.time() + 5
    while time.time() < deadline:
        snap = service.training_status()
        if snap["state"] != "running":
            break
        time.sleep(0.05)
    snap = service.training_status()
    assert snap["state"] == "failed"
    assert "boom" in snap["error"]
    # No activate happened, slot released.
    assert service.registry.active is None
    assert list(service.stream(params()))


def test_adapter_switching_uses_mutex(service):
    service.activate_adapter("style-v1")
    assert service.registry.active == "style-v1"
    service.restore_base()
    assert service.registry.restored is True

    # While generation runs, switching is rejected.
    stream = service.stream(params())
    next(stream)
    with pytest.raises(BusyError):
        service.activate_adapter("x")
    with pytest.raises(BusyError):
        service.restore_base()
    list(stream)


def test_adapter_listing_reports_active(service):
    service.registry.adapters = [{"name": "a"}, {"name": "b"}]
    listing = service.list_adapters()
    assert listing["active"] is None
    assert [a["name"] for a in listing["adapters"]] == ["a", "b"]
