"""JSONL sample parsing, full-batch validation and causal-loss features.

Every line must be a JSON object with a non-empty "prompt" and
"completion". The whole batch is validated before anything is
returned for training: malformed lines are reported with their 1-based
line number, overlong samples are rejected (never silently truncated)
and samples without a usable answer are dropped with an error.

Feature building follows one concatenation rule:

    input_ids = encode(prompt) + encode(completion) + [eos]

Prompt tokens and right-padding are masked with -100 so only the answer
tokens and the final EOS contribute to the causal loss.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

IGNORE_INDEX = -100


@dataclass
class Sample:
    line: int
    prompt: str
    completion: str


@dataclass
class DatasetError:
    line: int
    message: str

    def as_dict(self) -> dict:
        return {"line": self.line, "message": self.message}


@dataclass
class Features:
    line: int
    input_ids: list[int]
    labels: list[int]


def parse_jsonl(text: str) -> tuple[list[Sample], list[DatasetError]]:
    """Parse JSONL text, collecting every error instead of failing fast."""
    samples: list[Sample] = []
    errors: list[DatasetError] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue  # tolerate blank lines (e.g. trailing newline)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(DatasetError(lineno, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(obj, dict):
            errors.append(DatasetError(lineno, "line must be a JSON object"))
            continue
        prompt = obj.get("prompt")
        completion = obj.get("completion")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(DatasetError(lineno, "missing or empty 'prompt'"))
            continue
        if not isinstance(completion, str) or not completion.strip():
            errors.append(DatasetError(lineno, "missing or empty 'completion'"))
            continue
        samples.append(Sample(line=lineno, prompt=prompt, completion=completion))
    return samples, errors


def build_features(
    samples: list[Sample],
    tokenizer,
    eos_token_id: int,
    max_length: int,
) -> tuple[list[Features], list[DatasetError]]:
    """Tokenize samples into (input_ids, labels) pairs.

    Rejects samples whose tokenized length exceeds max_length and
    samples whose completion yields no answer tokens. Nothing is
    truncated silently.
    """
    features: list[Features] = []
    errors: list[DatasetError] = []
    for sample in samples:
        prompt_ids = list(tokenizer(sample.prompt, add_special_tokens=True).input_ids)
        answer_ids = list(tokenizer(sample.completion, add_special_tokens=False).input_ids)
        if not answer_ids:
            errors.append(
                DatasetError(sample.line, "completion produces no answer tokens")
            )
            continue
        answer_ids = answer_ids + [int(eos_token_id)]
        input_ids = prompt_ids + answer_ids
        if len(input_ids) > max_length:
            errors.append(
                DatasetError(
                    sample.line,
                    f"tokenized length {len(input_ids)} exceeds max_length "
                    f"{max_length}; sample rejected instead of truncated",
                )
            )
            continue
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        features.append(Features(line=sample.line, input_ids=input_ids, labels=labels))
    return features, errors


def collate(batch: list[Features], pad_token_id: int) -> dict:
    """Right-pad a batch; padding never contributes to the loss."""
    import torch

    width = max(len(f.input_ids) for f in batch)
    input_ids, labels, attention = [], [], []
    for feat in batch:
        pad = width - len(feat.input_ids)
        input_ids.append(feat.input_ids + [pad_token_id] * pad)
        labels.append(feat.labels + [IGNORE_INDEX] * pad)
        attention.append([1] * len(feat.input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def summarize(samples: list[Sample], features: list[Features]) -> dict:
    """Compact, privacy-aware digest of the training batch."""
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.prompt.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(sample.completion.encode("utf-8"))
        digest.update(b"\x01")
    lengths = [len(f.input_ids) for f in features]
    return {
        "sample_count": len(samples),
        "prompt_chars_total": sum(len(s.prompt) for s in samples),
        "completion_chars_total": sum(len(s.completion) for s in samples),
        "max_sequence_length": max(lengths) if lengths else 0,
        "content_sha256": digest.hexdigest()[:16],
    }
