# Speculative Inference

Python 3.10.12, PyTorch 2.6.0+cpu, Transformers 4.51.3 and FastAPI 0.115.12 application scaffold. No decoding engine or application tests have been implemented.

The local `.venv` contains the exact versions in `requirements.lock`. Use `.venv/bin/python`; activation is optional. Local model files are independent copies in `models/draft` for SmolLM2-135M and `models/target` for SmolLM2-360M. Both are Apache-2.0 pretrained English language models. No cloud credentials or GPU are needed. `models/manifest.json` pins upstream revisions, original URLs and SHA-256 checksums. Large safetensors files are excluded from Git but already downloaded locally; a fresh clone can restore them with `.venv/bin/python scripts/fetch_models.py`.

Install for a fresh clone using a Python virtual environment and:
```sh
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
```

Start the scaffold:
```sh
PYTHONPATH=src .venv/bin/python -m uvicorn specserve.app:app --host 127.0.0.1 --port 8000
```

The app exposes `/health`, a web page at `/`, and a streaming API. Run tests with `.venv/bin/python -m pytest`. Model cards are retained with each model and contain limitations and attribution.

## How it works

- `src/specserve/modeling.py` lazily loads SmolLM2-135M (draft) and SmolLM2-360M (target) with the shared tokenizer, float32 on CPU.
- `src/specserve/probability.py` implements temperature-normalized probabilities, greedy/sampling picks, the `min(1, p/q)` acceptance rule and the normalized positive-part residual distribution used after rejections.
- `src/specserve/engine.py` holds both decoding loops. Each request gets fresh `DynamicCache` KV caches; history is never recomputed. The draft proposes candidates one token at a time; the target verifies the whole block in a single forward pass; both caches are then cropped back to the confirmed prefix so unconfirmed tokens never leak out. With temperature 0 the accepted prefix matches target greedy decoding exactly and the first divergence is replaced by the target token; with temperature > 0 standard speculative rejection sampling preserves the target sampling distribution. EOS and the length cap truncate candidates.
- `src/specserve/service.py` enforces a single active request (busy requests get HTTP 409), cooperative stop, and releases the slot only after the in-flight forward pass finishes. Invalid parameters are rejected by validation before the slot is taken.
- `src/specserve/app.py` wires the HTTP routes; `web/index.html` is the page.

## API

`POST /api/generate` streams Server-Sent Events with JSON body:

```json
{"prompt": "The little robot", "mode": "speculative", "max_new_tokens": 64, "draft_steps": 4, "temperature": 0.0, "seed": 0}
```

`mode` is `speculative` or `autoregressive`. Events are `token` (confirmed text only), `stats`, `done` and `error`. `POST /api/stop` stops the active request. The stats event reports candidate/accepted counts, target/draft forward counts and elapsed time; no speedup is promised — speculative decoding trades extra draft compute for fewer serial target forward passes.

Example:

```sh
curl -N -X POST http://127.0.0.1:8000/api/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"The little robot","mode":"speculative","max_new_tokens":32,"draft_steps":4,"temperature":0,"seed":0}'
```
