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

The scaffold exposes `/health` and a blank application page. For package build use `.venv/bin/python -m build --no-isolation`; for tests use `.venv/bin/python -m pytest` after tests are implemented. No inference quality or speedup is implied by the scaffold. Model cards are retained with each model and contain limitations and attribution.
