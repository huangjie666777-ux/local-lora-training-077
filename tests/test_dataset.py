"""Dataset validation: line-located errors, loss masking, no truncation."""
import pytest

from specserve.dataset import (
    IGNORE_INDEX,
    build_features,
    collate,
    parse_jsonl,
    summarize,
)


class FakeTokenizer:
    """One token per whitespace-separated word, plus BOS on demand."""

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        ids = []
        if add_special_tokens:
            ids.append(1)  # BOS
        for word in text.split():
            ids.append(self.vocab.setdefault(word, len(self.vocab) + 2))
        return type("Enc", (), {"input_ids": ids})()


def test_parse_valid_lines():
    text = '{"prompt": "hi", "completion": "there"}\n{"prompt": "a", "completion": "b"}'
    samples, errors = parse_jsonl(text)
    assert len(samples) == 2 and not errors
    assert samples[0].line == 1 and samples[1].line == 2


def test_errors_carry_line_numbers():
    text = "\n".join(
        [
            '{"prompt": "ok", "completion": "fine"}',
            "not json",
            '{"prompt": "", "completion": "x"}',
            '{"prompt": "x"}',
            "[1, 2]",
        ]
    )
    samples, errors = parse_jsonl(text)
    assert [s.line for s in samples] == [1]
    assert [e.line for e in errors] == [2, 3, 4, 5]
    assert "prompt" in errors[1].message
    assert "completion" in errors[2].message


def test_blank_lines_tolerated():
    samples, errors = parse_jsonl('\n{"prompt": "a", "completion": "b"}\n\n')
    assert len(samples) == 1 and not errors


def test_features_mask_prompt_and_append_eos():
    samples, _ = parse_jsonl('{"prompt": "one two", "completion": "three"}')
    features, errors = build_features(samples, FakeTokenizer(), eos_token_id=0, max_length=32)
    assert not errors
    feat = features[0]
    # BOS + 2 prompt tokens masked; answer token + EOS scored.
    assert feat.labels[:3] == [IGNORE_INDEX] * 3
    assert feat.labels[3:] == feat.input_ids[3:]
    assert feat.labels[-1] == 0  # EOS participates in the loss


def test_overlong_rejected_not_truncated():
    samples, _ = parse_jsonl('{"prompt": "a b c d e", "completion": "f g"}')
    features, errors = build_features(samples, FakeTokenizer(), eos_token_id=0, max_length=4)
    assert not features
    assert len(errors) == 1 and "exceeds max_length" in errors[0].message


def test_empty_answer_tokens_rejected():
    samples, _ = parse_jsonl('{"prompt": "hello", "completion": "   "}') or (None, None)
    # whitespace-only completion never reaches feature building
    assert samples == []
    samples = [type("S", (), {"line": 7, "prompt": "hi", "completion": "???"})()]
    class EmptyTokenizer:
        def __call__(self, text, add_special_tokens=True):
            return type("Enc", (), {"input_ids": []})()
    features, errors = build_features(samples, EmptyTokenizer(), eos_token_id=0, max_length=8)
    assert not features and errors[0].line == 7


def test_collate_pads_without_loss():
    samples, _ = parse_jsonl('{"prompt": "a", "completion": "b"}\n{"prompt": "a b c", "completion": "d e"}')
    features, _ = build_features(samples, FakeTokenizer(), eos_token_id=0, max_length=32)
    batch = collate(features, pad_token_id=0)
    width = batch["input_ids"].shape[1]
    assert batch["input_ids"].shape == (2, width)
    short_labels = batch["labels"][0]
    # Shorter row padded with IGNORE_INDEX at the tail.
    assert int(short_labels[-1]) == IGNORE_INDEX
    assert int(batch["attention_mask"][0][-1]) == 0


def test_summarize_counts_and_digest():
    samples, _ = parse_jsonl('{"prompt": "a", "completion": "b"}')
    features, _ = build_features(samples, FakeTokenizer(), eos_token_id=0, max_length=32)
    summary = summarize(samples, features)
    assert summary["sample_count"] == 1
    assert summary["max_sequence_length"] == len(features[0].input_ids)
    assert len(summary["content_sha256"]) == 16
