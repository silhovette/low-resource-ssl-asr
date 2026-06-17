from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "ok"
            else:
                choices = [
                    (dp[i - 1][j] + 1, "del"),
                    (dp[i][j - 1] + 1, "ins"),
                    (dp[i - 1][j - 1] + 1, "sub"),
                ]
                dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])
    i, j = n, m
    subs = dels = ins = ok = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "ok":
            ok += 1
            i -= 1
            j -= 1
        elif op == "sub":
            subs += 1
            i -= 1
            j -= 1
        elif op == "del":
            dels += 1
            i -= 1
        elif op == "ins":
            ins += 1
            j -= 1
        else:
            break
    return subs, dels, ins, ok


def _split_words(text: str) -> List[str]:
    return [tok for tok in text.strip().split() if tok]


def wer(ref: str, hyp: str) -> float:
    ref_words = _split_words(ref)
    hyp_words = _split_words(hyp)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    subs, dels, ins, _ = _edit_distance(ref_words, hyp_words)
    return (subs + dels + ins) / len(ref_words)


def cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    subs, dels, ins, _ = _edit_distance(ref_chars, hyp_chars)
    return (subs + dels + ins) / len(ref_chars)


def aggregate_error_stats(ref: str, hyp: str) -> dict:
    ref_words = _split_words(ref)
    hyp_words = _split_words(hyp)
    subs, dels, ins, ok = _edit_distance(ref_words, hyp_words)
    n = max(1, len(ref_words))
    return {
        "wer": (subs + dels + ins) / n,
        "subs": subs,
        "dels": dels,
        "ins": ins,
        "correct": ok,
        "ref_words": len(ref_words),
    }
