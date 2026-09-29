from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI(title="Speculative Inference")
ROOT = Path(__file__).resolve().parents[2]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")
