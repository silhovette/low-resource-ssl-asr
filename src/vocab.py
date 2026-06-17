from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

from .utils import normalize_text


PAD = "<pad>"
UNK = "<unk>"
BLANK = "<blank>"


def build_vocab(texts: Iterable[str]) -> List[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        norm = normalize_text(text)
        for ch in norm:
            counter[ch] += 1
    vocab = [BLANK, PAD, UNK]
    for ch, _ in sorted(counter.items()):
        if ch not in vocab:
            vocab.append(ch)
    return vocab


def text_to_ids(text: str, vocab: Sequence[str]) -> List[int]:
    lookup = {ch: i for i, ch in enumerate(vocab)}
    unk = lookup[UNK]
    ids = []
    for ch in normalize_text(text):
        ids.append(lookup.get(ch, unk))
    return ids


def ids_to_text(ids: Sequence[int], vocab: Sequence[str]) -> str:
    chars = []
    for idx in ids:
        token = vocab[int(idx)]
        if token in {BLANK, PAD, UNK}:
            continue
        chars.append(token)
    return " ".join("".join(chars).split())
