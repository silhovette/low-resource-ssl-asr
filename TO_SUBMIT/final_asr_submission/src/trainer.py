from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .decoder import greedy_ctc_decode
from .metrics import cer, wer
from .utils import ensure_dir, read_jsonl, set_seed


@dataclass
class EvalResult:
    wer: float
    cer: float
    num_examples: int
    rtf: float


def ctc_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor, blank_id: int = 0) -> torch.Tensor:
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    targets = []
    for i, length in enumerate(target_lengths.tolist()):
        targets.extend(labels[i, :length].tolist())
    targets = torch.tensor(targets, dtype=torch.long, device=logits.device)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=blank_id, zero_infinity=True)


def evaluate_ctc(model, dataloader, vocab, device, blank_id: int = 0) -> EvalResult:
    model.eval()
    total_wer = 0.0
    total_cer = 0.0
    total_audio_seconds = 0.0
    inference_seconds = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            count += len(batch["texts"])
            total_audio_seconds += sum(float(row.get("duration", 0.0)) for row in batch["rows"])
            labels = batch["label_ids"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            start = time.perf_counter()
            if "input_values" in batch:
                inputs = batch["input_values"].to(device)
                mask = batch.get("attention_mask")
                mask = mask.to(device) if mask is not None else None
                logits = model(inputs, mask)
            else:
                features = batch["features"].to(device)
                feat_lengths = batch["feature_lengths"].to(device)
                logits = model(features, feat_lengths)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            preds = greedy_ctc_decode(logits, vocab, blank_id=blank_id)
            for ref, hyp in zip(batch["texts"], preds):
                total_wer += wer(ref, hyp)
                total_cer += cer(ref, hyp)
    denom = max(1, count)
    rtf = inference_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0
    return EvalResult(total_wer / denom, total_cer / denom, count, rtf)


def write_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def write_error_examples(path: str | Path, examples: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ref", "hyp", "wer", "cer"])
        writer.writeheader()
        for row in examples:
            writer.writerow(row)
