# Speculative Inference + Local LoRA Adaptation

FastAPI demo on CPU: speculative decoding with SmolLM2-360M (target) and
SmolLM2-135M (draft), plus local LoRA domain adaptation of the target
model from private JSONL samples. Draft and base target weights are
never modified.

## Layout

- `src/specserve/engine.py` — autoregressive / speculative decode loops (KV-cached)
- `src/specserve/modeling.py` — model loading, adapter attach/switch/restore
- `src/specserve/dataset.py` — JSONL parsing, full-batch validation, loss masking
- `src/specserve/training.py` — CPU LoRA training loop (async, cancellable)
- `src/specserve/adapters.py` — named adapter store, atomic save, compatibility checks
- `src/specserve/service.py` — one mutex slot shared by generation, training, switching
- `src/specserve/app.py` — HTTP API
- `examples/style_demo.jsonl` — tiny English style demo (pirate-flavored answers)
- `adapters/<name>/` — saved adapters (PEFT files + `metadata.json`)

## Run

`bash
python -m venv .venv && .venv/bin/pip install -e .  # deps already pinned in .venv
PYTHONPATH=src .venv/bin/uvicorn specserve.app:app --host 127.0.0.1 --port 8077
# open http://127.0.0.1:8077/ for the page: samples -> train -> versions -> generate
`

Run tests: `.venv/bin/python -m pytest`

## Training data (JSONL)

One JSON object per line with non-empty `prompt` and `completion`:

`json
{"prompt": "Question: What do bees make?\nAnswer:", "completion": "Ahoy! Bees make sweet honey, matey!"}
`

The whole batch is validated before training starts; every bad line is
reported with its 1-based line number. Concatenation rule:
`encode(prompt) + encode(completion) + [EOS]`. Prompt tokens and padding
are masked (`-100`); only answer tokens and EOS carry loss. Samples
longer than `max_length` or without answer tokens are rejected — never
silently truncated.

## API quick tour (curl)

`bash
# train (async; 409 while busy, 422 with per-line errors when invalid)
curl -X POST localhost:8077/api/train -H 'Content-Type: application/json' -d '{
  "dataset": "{\"prompt\": \"Q: hi?\\nA:\", \"completion\": \"Ahoy! Hi, matey!\"}",
  "name": "my-style", "steps": 10, "learning_rate": 0.0002, "seed": 0, "max_length": 192}'
curl localhost:8077/api/train/status        # state, actual steps, loss, stop reason
curl -X POST localhost:8077/api/train/cancel  # takes effect after the current step

# versions (persisted under adapters/, listed again after restart)
curl localhost:8077/api/adapters
curl -X POST localhost:8077/api/adapters/select -H 'Content-Type: application/json' -d '{"name":"my-style"}'
curl -X POST localhost:8077/api/adapters/select -H 'Content-Type: application/json' -d '{"name":"base"}'

# generate (SSE); the adapter applies to the target in BOTH modes, draft unchanged
curl -N -X POST localhost:8077/api/generate -H 'Content-Type: application/json' -d \
  '{"prompt":"Question: What do bees make?\nAnswer:","mode":"speculative","max_new_tokens":24}'
curl -X POST localhost:8077/api/stop
`

## Limits and resources

- Train params: `steps` 1–100 (default 10), `learning_rate` 1e-5–5e-3
  (default 2e-4), `seed` 0..2^31-1, `max_length` 32–512 (default 192),
  batch size 4, LoRA r=8 / alpha=16 / dropout 0.05 on `q_proj`,`v_proj`.
- Everything runs on CPU in fp32. Training loads a private copy of the
  360M target (~1.4 GB) next to the inference pair (~2 GB); expect
  roughly 3.5–4 GB RAM peak and ~0.2–1 s per step for short sequences.
- One operation at a time: generation, training and adapter switching
  share a single slot; busy requests get 409, invalid ones 422 without
  taking the slot. Failed/cancelled training persists nothing and keeps
  the active version. Corrupt or incompatible adapters are rejected on
  select and the previous version stays active. Each generation request
  pins the version active at its start and uses fresh KV caches.

## Engine fixes included

- Full-accept rounds no longer drop the last candidate from the draft
  KV cache (it is forwarded before cropping).
- The autoregressive loop no longer runs a forward pass after a stop
  request or after the token budget is spent.
