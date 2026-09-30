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

## Prepared training dependencies

The local virtual environment also includes PEFT 0.15.2, Accelerate 1.6.0 and psutil 7.0.0. Their exact versions are recorded in requirements.lock. This dependency preparation does not add training or adapter-management functionality.

## Local LoRA domain adaptation

The target SmolLM2-360M can now be adapted locally with private English
prompt/completion demonstrations. LoRA (PEFT 0.15.2, r=8 on
q/k/v/o_proj) is trained on CPU. The original base weights and the 135M
draft model are never modified; the draft stays base for both decoding
modes while the adapter only wraps the target.

### Data rules

- Import JSONL through the page (section 1). Each non-empty line must be a
  JSON object with non-empty string `prompt` and `completion`. Empty lines
  are skipped; anything else is an error.
- The whole file is validated and every error is reported with its 1-based
  line number before training can start. Nothing is trained on a partial
  batch.
- One concatenation rule is used for every sample: `<bos>prompt\n\n<completion><eos>`.
  BOS and prompt tokens are label-masked (-100, no loss); completion tokens
  and the trailing EOS participate in the standard causal LM loss.
- Sequences longer than `max_length` are rejected (no silent truncation),
  as are completions contributing no supervised token.

### Training lifecycle and resource limits

Training runs asynchronously on a background thread and is mutually
exclusive with generation and adapter switching; while anything holds the
engine, new training/generation/switch requests get HTTP 409. Invalid
parameters and data are rejected before the mutex is acquired, so they
never occupy the slot.

Bounded ranges (short-sequence CPU defaults in parentheses):

| Parameter | Range | Default |
| --- | --- | --- |
| `steps` | 1–200 | 10 |
| `learning_rate` | 1e-6–1.0 | 2e-4 |
| `seed` | 0–2^31-1 | 0 |
| `max_length` | 32–512 | 256 |

Cancellation (`POST /api/training/cancel`) is cooperative and takes effect
only after the current optimizer step completes. The status endpoint
reports configured vs actual steps, per-step losses, final loss and the
termination reason (`configured_steps_reached`, `cancelled after optimizer
step N`, or an error). Failed or cancelled runs only train on a discarded
worker copy of the base target, so the serving inference version never
changes; the mutex is always released afterwards.

A 360M fp32 target is roughly 1.5 GB of RAM plus the draft and PEFT
workspace; training one short sample (~32 tokens) is on the order of one
second per CPU-bound optimizer step on this machine. Keep demonstrations
short and step counts small on CPU. No GPU is used.

### Named adapters and versions

On success the adapter is saved atomically (temp directory + rename) under
`adapters/<name>/` together with `adapter_meta.json`: base-model identity
(`SmolLM2-360M` plus the SHA-256 of the base `config.json`), training
config, sample summary, loss and timestamps. Names are limited to letters,
digits, `-` and `_` (max 32 chars); existing names are not overwritten.

After a restart, `GET /api/adapters` lists saved adapters while the base
model is served by default; use `POST /api/adapters/activate` to load one
or `POST /api/adapters/restore` to return to the untouched base. Loading an
adapter whose base identity differs or whose files are corrupt fails with
HTTP 400 and keeps the previously served version in place. Each generation
request snapshots the model pair at start (its adapter version is fixed for
the whole request) and always builds fresh `DynamicCache` objects; no old
KV cache is ever reused.

### Training / storage / request API

- `GET /api/training/limits` — ranges and defaults.
- `POST /api/training/validate` — parse and locate JSONL errors without
  taking the engine slot.
- `POST /api/training/start` — full validation, then async training.
- `GET /api/training/status` — live/final status, losses, reason.
- `POST /api/training/cancel` — cancel after the current optimizer step.
- `GET /api/adapters` — active version and saved adapters.
- `POST /api/adapters/activate` / `POST /api/adapters/restore`.

Example (train, poll, then generate with the adapted target):

```sh
python3 - <<'PY' > /tmp/train.json
import json
rows = [
  {"prompt": "Q: What is support ticket status 101? A:", "completion": " Ticket 101 is open and assigned to the billing team."},
  {"prompt": "Q: How should I greet a customer? A:", "completion": " Hello, thanks for contacting support. How can I help today?"},
]
print(json.dumps({"name": "support-style", "steps": 6,
                  "jsonl": "\n".join(json.dumps(r) for r in rows)}))
PY
curl -X POST http://127.0.0.1:8000/api/training/start \
  -H 'Content-Type: application/json' --data @/tmp/train.json
curl -s http://127.0.0.1:8000/api/training/status
curl -N -X POST http://127.0.0.1:8000/api/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"Q: How should I greet a customer? A:","mode":"speculative","max_new_tokens":24,"draft_steps":4,"temperature":0,"seed":0}'
```

The web page (`/`) chains the workflow: paste/load the small English
example, validate it with line-localized errors, train and watch live loss,
list/activate/restore adapter versions, and continue generating with either
decoding mode.

### Decoding fixes included

- When the target accepted a full draft block without a replacement token,
  the draft KV cache used to miss the last candidate (it was never fed), so
  the next round re-fed the pending token and shifted draft positions by
  one. The draft now advances after every non-EOS proposal and caches are
  cropped back to the confirmed prefix.
- Cooperative stop no longer launches an extra draft/verification forward
  (speculative) or the post-token target forward (autoregressive) after a
  stop is observed.
