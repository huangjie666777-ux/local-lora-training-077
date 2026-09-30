"""FastAPI surface: generation, LoRA training, adapter versions, health."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .adapters import AdapterError, valid_name
from .dataset import build_features, parse_jsonl
from .engine import EngineParams
from .service import BusyError, GenerationService
from .training import (
    DEFAULTS,
    LEARNING_RATE_RANGE,
    MAX_LENGTH_RANGE,
    SEED_RANGE,
    STEPS_RANGE,
    TrainParams,
)

app = FastAPI(title="Speculative Inference")
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
    dataset: str = Field(min_length=1, max_length=2_000_000)
    name: str | None = Field(default=None, max_length=40)
    steps: int = Field(default=DEFAULTS["steps"], ge=STEPS_RANGE[0], le=STEPS_RANGE[1])
    learning_rate: float = Field(
        default=DEFAULTS["learning_rate"],
        ge=LEARNING_RATE_RANGE[0],
        le=LEARNING_RATE_RANGE[1],
    )
    seed: int = Field(default=DEFAULTS["seed"], ge=SEED_RANGE[0], le=SEED_RANGE[1])
    max_length: int = Field(
        default=DEFAULTS["max_length"], ge=MAX_LENGTH_RANGE[0], le=MAX_LENGTH_RANGE[1]
    )


class SelectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)  # "base" restores the base model


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
    # Pydantic validation runs first, so invalid requests never take the slot.
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


@app.post("/api/train")
def train(req: TrainRequest):
    """Validate the whole JSONL batch, then start async LoRA training."""
    name = req.name or time.strftime("adapter-%Y%m%d-%H%M%S", time.gmtime())
    if not valid_name(name):
        raise HTTPException(status_code=422, detail=f"invalid adapter name: {name!r}")
    if service.store.adapter_path(name).exists():
        raise HTTPException(status_code=422, detail=f"adapter {name!r} already exists")
    samples, errors = parse_jsonl(req.dataset)
    if not errors:
        models = service.registry.get()
        features, errors = build_features(
            samples, models.tokenizer, models.eos_token_id, req.max_length
        )
    if errors:
        # Full-batch validation: every bad line is reported, nothing trains.
        raise HTTPException(
            status_code=422,
            detail={"message": "dataset validation failed", "errors": [e.as_dict() for e in errors]},
        )
    if not samples:
        raise HTTPException(status_code=422, detail="dataset contains no samples")
    params = TrainParams(
        name=name,
        steps=req.steps,
        learning_rate=req.learning_rate,
        seed=req.seed,
        max_length=req.max_length,
    )
    try:
        return service.start_train(params, samples, features)
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/train/status")
def train_status():
    return service.train_status()


@app.post("/api/train/cancel")
def train_cancel():
    return {"cancelling": service.cancel_train()}


@app.get("/api/adapters")
def adapters():
    return service.versions()


@app.post("/api/adapters/select")
def select_adapter(req: SelectRequest):
    name = None if req.name == "base" else req.name
    try:
        return service.select_adapter(name)
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except AdapterError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
