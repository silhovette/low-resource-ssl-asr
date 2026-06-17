from __future__ import annotations

import _bootstrap  # noqa: F401

from src.metrics import cer, wer
from src.utils import normalize_text
from src.vocab import build_vocab, text_to_ids, ids_to_text


def main() -> None:
    text = "Hello, World!"
    norm = normalize_text(text)
    assert norm == "hello world"
    vocab = build_vocab([text])
    ids = text_to_ids(text, vocab)
    assert ids_to_text(ids, vocab) == "hello world"
    assert wer("hello world", "hello world") == 0.0
    assert cer("hello world", "hello world") == 0.0
    print("smoke test passed")


if __name__ == "__main__":
    main()
