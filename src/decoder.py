from __future__ import annotations

from typing import List, Optional, Sequence

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


def _build_pyctcdecode_decoder(labels: List[str], kenlm_path: str, alpha: float = 0.5, beta: float = 1.5):
    """Build a CTC beam-search decoder backed by a KenLM n-gram model.

    Requires:  pip install pyctcdecode
    The KenLM .arpa file must already exist (see ``scripts/train_lm.py``).
    """
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        raise ImportError(
            "pyctcdecode is required for LM-enhanced beam search. "
            "Install with: pip install pyctcdecode"
        )
    # pyctcdecode expects labels *without* the blank token — blank is handled
    # internally and placed at the last position of the logit tensor.
    # Our vocab layout is: idx 0 = <blank>, idx 1 = <pad>, idx 2 = <unk>,
    # then characters.  We strip the special tokens for pyctcdecode.
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=kenlm_path,
        alpha=alpha,   # LM weight
        beta=beta,     # word insertion bonus
    )
    return decoder


def ctc_beam_search_decode(
    logits: torch.Tensor,
    vocab: Sequence[str],
    kenlm_path: Optional[str] = None,
    beam_width: int = 50,
    alpha: float = 0.5,
    beta: float = 1.5,
    blank_id: int = 0,
) -> List[str]:
    """CTC beam-search decode with optional KenLM n-gram rescoring.

    When ``kenlm_path`` is provided and ``pyctcdecode`` is installed, a
    KenLM-augmented prefix beam search is used.  Otherwise falls back to a
    simple (no-LM) prefix beam search.

    Parameters
    ----------
    logits : Tensor, shape (B, T, V)
    vocab : list of token strings (length V).  Index 0 must be ``<blank>``.
    kenlm_path : path to .arpa KenLM file, or None for no LM.
    beam_width : beam size.
    alpha : LM weight (only used with kenlm_path).
    beta : word-insertion bonus (only used with kenlm_path).
    blank_id : index of the CTC blank token (default 0).

    Returns
    -------
    List[str] — one decoded transcript per batch element.
    """
    if kenlm_path is not None:
        return _beam_search_with_lm(
            logits, vocab, kenlm_path, beam_width, alpha, beta, blank_id
        )
    # Fallback: simple prefix beam search (no LM)
    return _beam_search_no_lm(logits, vocab, beam_width, blank_id)


def _beam_search_no_lm(
    logits: torch.Tensor, vocab: Sequence[str], beam_width: int, blank_id: int
) -> List[str]:
    """Simple CTC prefix beam search without a language model.

    Based on the prefix-beam algorithm (Graves 2006).  Each beam entry stores
    the log-probability of two families of paths: those ending in blank (p_b)
    and those ending in a non-blank token (p_nb).
    """
    NEG_INF = -1e30

    log_probs = logits.log_softmax(dim=-1).cpu()  # (B, T, V)
    results: List[str] = []
    for b in range(log_probs.size(0)):
        lp = log_probs[b]  # (T, V)
        # beam: dict  prefix → (p_b, p_nb)
        # Initially only the empty prefix exists; both families start at
        # -inf because no frames have been processed yet.
        beam = {(): (NEG_INF, 0.0)}
        for t in range(lp.size(0)):
            frame = lp[t]
            new_beam: dict = {}
            for prefix, (p_b, p_nb) in beam.items():
                for v in range(frame.size(0)):
                    prob = float(frame[v].item())
                    if prob < -1e9:
                        continue
                    if v == blank_id:
                        new_b = _logadd(p_b + prob, p_nb + prob)
                        old = new_beam.get(prefix, (NEG_INF, NEG_INF))
                        new_beam[prefix] = (new_b, old[1])
                    elif prefix and prefix[-1] == v:
                        # CTC collapse: repeated token stays on same prefix
                        new_nb = _logadd(p_nb + prob,
                                         new_beam.get(prefix, (NEG_INF, NEG_INF))[1])
                        new_beam[prefix] = (new_beam.get(prefix, (NEG_INF, NEG_INF))[0],
                                             new_nb)
                        # Also create a new prefix with this token
                        new_pre = prefix + (v,)
                        new_b2 = _logadd(p_b + prob, p_nb + prob)
                        old2 = new_beam.get(new_pre, (NEG_INF, NEG_INF))
                        new_beam[new_pre] = (new_b2, old2[1])
                    else:
                        new_pre = prefix + (v,)
                        new_b3 = _logadd(p_b + prob, p_nb + prob)
                        old3 = new_beam.get(new_pre, (NEG_INF, NEG_INF))
                        new_beam[new_pre] = (new_b3, old3[1])
            # Prune to beam_width
            scored = sorted(
                ((_logadd(pb, pnb), pre, pb, pnb) for pre, (pb, pnb) in new_beam.items()),
                reverse=True, key=lambda x: x[0],
            )
            beam = {pre: (pb, pnb) for _, pre, pb, pnb in scored[:beam_width]}
        best = max(beam.items(), key=lambda kv: _logadd(kv[1][0], kv[1][1]))[0]
        results.append(ids_to_text(list(best), vocab))
    return results


def _beam_search_with_lm(
    logits: torch.Tensor,
    vocab: Sequence[str],
    kenlm_path: str,
    beam_width: int,
    alpha: float,
    beta: float,
    blank_id: int,
) -> List[str]:
    """Beam search with KenLM rescoring via pyctcdecode."""
    import numpy as np

    # Build label list excluding special tokens (blank, pad, unk are 0,1,2)
    labels_for_decoder = [v for v in vocab if v not in ("<blank>", "<pad>", "<unk>")]

    special = {"<blank>", "<pad>", "<unk>"}
    pyctc_idx = {v: i for i, v in enumerate(labels_for_decoder)}
    pyctc_blank = len(labels_for_decoder)  # blank goes last in pyctcdecode

    decoder = _build_pyctcdecode_decoder(labels_for_decoder, kenlm_path, alpha, beta)

    log_probs = logits.log_softmax(dim=-1).cpu().numpy()  # (B, T, V)
    results: List[str] = []
    for b in range(log_probs.shape[0]):
        T, V_ours = log_probs[b].shape
        V_pyctc = len(labels_for_decoder) + 1  # chars + blank
        lp_pyctc = np.full((T, V_pyctc), -1e10, dtype=np.float32)
        for our_idx, token in enumerate(vocab):
            if token in special:
                continue
            lp_pyctc[:, pyctc_idx[token]] = log_probs[b, :, our_idx]
        lp_pyctc[:, pyctc_blank] = log_probs[b, :, blank_id]

        text = decoder.decode(lp_pyctc, beam_width=beam_width)
        results.append(text)
    return results


def _logadd(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) in a numerically stable way."""
    import math
    if a < b:
        a, b = b, a
    if a == float("-inf") or a - b > 50:
        return a
    return a + math.log1p(math.exp(b - a))
