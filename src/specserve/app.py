"""FastAPI surface: generation SSE, LoRA training and adapter versions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .data import DataValidationError, parse_jsonl
from .engine import EngineParams
from .service import BusyError, GenerationService, TrainingError
from .training import DEFAULTS, LIMITS

app = FastAPI(title="Speculative Inference + Local LoRA Adaptation")
ROOT = Path(__file__).resolve().parents[2]
service = GenerationService()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    mode: Literal["autoregressive", "speculative"] = "speculative"
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    draft_steps: int = Field(default=4, ge=1, le=16)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class TrainRequest(BaseModel):
    name: str = Field(min_length=1, max_length=32)
    jsonl: str = Field(min_length=1)
    steps: int = Field(default=DEFAULTS["steps"])
    learning_rate: float = Field(default=DEFAULTS["learning_rate"])
    seed: int = Field(default=DEFAULTS["seed"])
    max_length: int = Field(default=DEFAULTS["max_length"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.post("/api/stop")
def stop():
    return {"stopping": service.stop()}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    params = EngineParams(
        prompt=req.prompt,
        mode=req.mode,
        max_new_tokens=req.max_new_tokens,
        draft_steps=req.draft_steps,
        temperature=req.temperature,
        seed=req.seed,
    )
    try:
        events = service.stream(params)
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    def sse():
        try:
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            events.close()

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/api/training/limits")
def training_limits():
    return {"limits": LIMITS, "defaults": DEFAULTS}


@app.post("/api/training/validate")
def training_validate(req: TrainRequest):
    """Parse the JSONL and locate every bad line without occupying the slot."""
    try:
        samples = parse_jsonl(req.jsonl)
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors})
    return {"valid": True, "samples": len(samples), "errors": []}


@app.get("/api/training/status")
def training_status():
    status = service.training_status()
    if status is None:
        return {"state": "idle"}
    return status


@app.post("/api/training/cancel")
def training_cancel():
    return {"cancelling": service.cancel_training()}


@app.post("/api/training/start")
def training_start(req: TrainRequest):
    try:
        samples = parse_jsonl(req.jsonl)
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors})
    try:
        status = service.start_training(
            req.name,
            samples,
            req.steps,
            req.learning_rate,
            req.seed,
            req.max_length,
        )
    except TrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors})
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return status


@app.get("/api/adapters")
def adapters_list():
    return service.list_adapters()


class AdapterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=32)


@app.post("/api/adapters/activate")
def adapters_activate(req: AdapterRequest):
    try:
        meta = service.activate_adapter(req.name)
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"active": req.name, "meta": meta}


@app.post("/api/adapters/restore")
def adapters_restore():
    try:
        service.restore_base()
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"active": None}
