"""JSONL parsing, line-localized errors and tokenization rules."""
import pytest
import types

from specserve.data import (
    DataValidationError,
    Sample,
    encode_sample,
    parse_jsonl,
    prepare_batch,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        ids = [100 + (ord(w[0]) % 50) for w in text.split()]
        return types.SimpleNamespace(input_ids=ids)


def test_parse_valid_jsonl():
    text = '{"prompt": "hi", "completion": "there"}\n\n{"prompt":"a","completion":"b"}\n'
    samples = parse_jsonl(text)
    assert len(samples) == 2
    assert samples[0].prompt == "hi"


def test_errors_are_line_localized_and_batched():
    text = "\n".join(
        [
            '{"prompt": "ok", "completion": "yes"}',
            "{not json",
            '{"prompt": "  ", "completion": "x"}',
            '{"prompt": "p"}',
            '{"prompt": "p", "completion": "c", "extra": 1}',
            "[]",
        ]
    )
    with pytest.raises(DataValidationError) as exc:
        parse_jsonl(text)
    joined = " ".join(exc.value.errors)
    assert "line 2" in joined
    assert "line 3" in joined
    assert "line 4" in joined
    assert "line 5" in joined
    assert "line 6" in joined


def test_empty_input_rejected():
    with pytest.raises(DataValidationError):
        parse_jsonl("\n  \n")


def test_encode_masks_prompt_and_includes_eos():
    from specserve.modeling import ModelRegistry

    tok = ModelRegistry().get().tokenizer
    sample = parse_jsonl('{"prompt": "Question here", "completion": "Answer text"}')[0]
    enc = encode_sample(sample, tok, max_length=512)
    assert enc.input_ids[-1] == tok.eos_token_id
    assert enc.labels[-1] == tok.eos_token_id
    assert enc.labels[0] == -100
    supervised = [lab for lab in enc.labels if lab != -100]
    assert tok.eos_token_id in supervised
    assert len(supervised) == enc.answer_tokens


def test_overslong_rejected_not_truncated():
    long_prompt = "word " * 100
    with pytest.raises(DataValidationError):
        encode_sample(Sample(long_prompt, "ok"), FakeTokenizer(), max_length=32)


def test_completion_with_no_tokens_rejected():
    class Weird(FakeTokenizer):
        eos_token_id = None

        def __call__(self, text, add_special_tokens=True):
            # Prompt is suffixed with newlines; the completion yields nothing.
            if not text.endswith("\n\n"):
                return types.SimpleNamespace(input_ids=[])
            return super().__call__(text, add_special_tokens)

    with pytest.raises(DataValidationError):
        encode_sample(Sample("zzz", "blank"), Weird(), max_length=64)


def test_prepare_batch_collects_all_length_errors():
    samples = [Sample("word " * 50, "ok"), Sample("short", "fine")]
    with pytest.raises(DataValidationError) as exc:
        prepare_batch(samples, FakeTokenizer(), max_length=10)
    assert any("exceeds max_length" in e for e in exc.value.errors)
