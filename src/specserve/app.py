"""FastAPI surface: validation, SSE streaming, stop and health routes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engine import EngineParams
from .service import BusyError, GenerationService

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

