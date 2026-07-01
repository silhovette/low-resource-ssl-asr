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


def evaluate_ctc(model, dataloader, vocab, device, blank_id: int = 0,
                 kenlm_path: str | None = None, beam_width: int = 50,
                 lm_alpha: float = 0.5, lm_beta: float = 1.5) -> EvalResult:
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
            if kenlm_path is not None:
                from .decoder import ctc_beam_search_decode
                preds = ctc_beam_search_decode(
                    logits, vocab, kenlm_path=kenlm_path,
                    beam_width=beam_width, alpha=lm_alpha, beta=lm_beta,
                    blank_id=blank_id,
                )
            else:
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


def train_one_epoch(
    model, dataloader, optimizer, device,
    blank_id: int = 0, grad_clip: float = 1.0,
) -> float:
    """Run one training epoch.  Returns the average CTC loss."""
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        labels = batch["label_ids"].to(device)
        target_lengths = batch["label_lengths"].to(device)
        if "input_values" in batch:
            logits = model(batch["input_values"].to(device),
                           batch["attention_mask"].to(device))
            raw_lengths = batch["input_lengths"].to(device)
            if hasattr(model, "encoder") and hasattr(model.encoder, "_get_feat_extract_output_lengths"):
                input_lengths = model.encoder._get_feat_extract_output_lengths(raw_lengths).to(device)
            else:
                input_lengths = torch.full((logits.shape[0],), logits.shape[1],
                                           dtype=torch.long, device=device)
        else:
            logits = model(batch["features"].to(device),
                           batch["feature_lengths"].to(device))
            input_lengths = batch["feature_lengths"].to(device)
        loss = ctc_loss_from_logits(logits, labels, input_lengths, target_lengths, blank_id=blank_id)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(1, steps)


def train_with_finetune(
    model, train_loader, dev_loader, vocab, device,
    config: dict, blank_id: int = 0,
) -> dict:
    """Two-phase training: frozen warm-up → encoder fine-tuning.

    Config keys used (all under ``training`` except where noted):
      * epochs          — total number of epochs (warmup + finetune)
      * finetune_start_epoch — first epoch where encoder layers are unfrozen
      * finetune_layers — how many top transformer layers to unfreeze (0 = all)
      * finetune_lr     — learning rate for the fine-tuning phase
      * learning_rate   — learning rate for the warm-up (frozen) phase
      * batch_size, max_audio_seconds, max_eval_audio_seconds, grad_clip,
        num_workers — standard training knobs.

    Returns a list of per-epoch metric dicts (``history``).
    """
    epochs = int(config["training"]["epochs"])
    finetune_start = int(config["training"].get("finetune_start_epoch", epochs + 1))
    finetune_layers = int(config["training"].get("finetune_layers", 3))
    finetune_lr = float(config["training"].get("finetune_lr",
                        float(config["training"]["learning_rate"]) * 0.1))
    warmup_lr = float(config["training"]["learning_rate"])
    grad_clip = float(config["training"].get("grad_clip", 1.0))

    history = []
    best_dev = float("inf")
    optimizer = None
    lr = warmup_lr

    for epoch in range(1, epochs + 1):
        # ---- switch to fine-tuning phase ----
        if epoch == finetune_start:
            if hasattr(model, "unfreeze_transformer_layers"):
                if finetune_layers == 0:
                    n = model.unfreeze_all()
                else:
                    n = model.unfreeze_transformer_layers(finetune_layers)
                print(f"[finetune] unfroze {n} transformer layers "
                      f"(trainable params: {model.trainable_param_count:,})")
            lr = finetune_lr
            # Rebuild optimizer with new lr for all trainable params
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), lr=lr)

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), lr=lr)

        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     blank_id=blank_id, grad_clip=grad_clip)

        dev = evaluate_ctc(model, dev_loader, vocab, device, blank_id=blank_id)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_wer": dev.wer,
            "dev_cer": dev.cer,
            "dev_rtf": dev.rtf,
        }
        history.append(row)
        print(row)
        if dev.wer < best_dev:
            best_dev = dev.wer
    return history, best_dev
