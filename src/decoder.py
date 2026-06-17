from __future__ import annotations

from typing import List, Sequence

import torch

from .vocab import BLANK, ids_to_text


def greedy_ctc_decode(logits: torch.Tensor, vocab: Sequence[str], blank_id: int = 0) -> List[str]:
    preds = logits.argmax(dim=-1).tolist()
    results: List[str] = []
    for seq in preds:
        collapsed = []
        prev = None
        for idx in seq:
            if idx == blank_id:
                prev = idx
                continue
            if idx == prev:
                continue
            collapsed.append(idx)
            prev = idx
        results.append(ids_to_text(collapsed, vocab))
    return results
