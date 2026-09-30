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

