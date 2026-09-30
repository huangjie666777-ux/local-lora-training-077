"""JSONL sample import, full-batch validation and unified tokenization.

Every sample uses one concatenation rule:

    <bos>prompt\n\n<completion><eos>

Prompt (and BOS) tokens are masked out of the causal LM loss; completion
tokens and the trailing EOS participate. Sequences are never silently
truncated: anything longer than max_length is rejected, and a sample whose
completion contributes no supervised token is rejected as well.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

PROMPT_SUFFIX = "\n\n"


class DataValidationError(ValueError):
    """Collects every bad input line so the UI can point at all of them."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Sample:
    prompt: str
    completion: str


@dataclass(frozen=True)
class EncodedSample:
    input_ids: list[int]
    labels: list[int]      # -100 where the loss must not attend
    answer_tokens: int     # supervised positions (completion + EOS)


@dataclass(frozen=True)
class SampleSummary:
    count: int
    total_tokens: int
    answer_tokens: int
    max_total_tokens: int
    prompts: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "total_tokens": self.total_tokens,
            "answer_tokens": self.answer_tokens,
            "max_total_tokens": self.max_total_tokens,
            "prompts": list(self.prompts),
        }


def parse_jsonl(text: str) -> list[Sample]:
    """Parse and validate the whole JSONL blob.

    All lines are checked and every error (with its 1-based line number) is
    returned together; nothing starts training before the full batch passes.
    Blank/whitespace-only lines are ignored and do not count as samples.
    """
    samples: list[Sample] = []
    errors: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {lineno}: expected a JSON object")
            continue
        prompt = obj.get("prompt")
        completion = obj.get("completion")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"line {lineno}: missing or empty 'prompt'")
            continue
        if not isinstance(completion, str) or not completion.strip():
            errors.append(f"line {lineno}: missing or empty 'completion'")
            continue
        if len(obj) > 2:
            extra = sorted(set(obj) - {"prompt", "completion"})
            errors.append(f"line {lineno}: unexpected fields {extra}")
            continue
        samples.append(Sample(prompt=prompt.strip(), completion=completion.strip()))
    if not samples and not errors:
        errors.append("input contains no samples")
    if errors:
        raise DataValidationError(errors)
    return samples


def encode_sample(sample: Sample, tokenizer, max_length: int) -> EncodedSample:
    """Apply the single concatenation rule; reject, never truncate."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    prompt_text = sample.prompt + PROMPT_SUFFIX
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(sample.completion, add_special_tokens=False).input_ids
    if eos is not None:
        answer_ids = answer_ids + [eos]
    prefix = [bos] if bos is not None else []
    input_ids = prefix + prompt_ids + answer_ids
    if len(input_ids) > max_length:
        raise DataValidationError([
            f"prompt {sample.prompt[:40]!r}: sequence length {len(input_ids)} "
            f"exceeds max_length {max_length}"
        ])
    if not answer_ids:
        raise DataValidationError([
            f"prompt {sample.prompt[:40]!r}: completion has no supervised tokens"
        ])
    labels = [-100] * (len(prefix) + len(prompt_ids)) + list(answer_ids)
    return EncodedSample(input_ids, labels, len(answer_ids))


def prepare_batch(samples: list[Sample], tokenizer, max_length: int) -> tuple[list[EncodedSample], SampleSummary]:
    """Validate/encode every sample; raise with all errors if any fail."""
    encoded: list[EncodedSample] = []
    errors: list[str] = []
    for sample in samples:
        try:
            encoded.append(encode_sample(sample, tokenizer, max_length))
        except DataValidationError as exc:
            errors.extend(exc.errors)
    if errors:
        raise DataValidationError(errors)
    total = sum(len(item.input_ids) for item in encoded)
    answers = sum(item.answer_tokens for item in encoded)
    longest = max((len(item.input_ids) for item in encoded), default=0)
    preview = tuple(sample.prompt for sample in samples[:5])
    return encoded, SampleSummary(len(encoded), total, answers, longest, preview)

