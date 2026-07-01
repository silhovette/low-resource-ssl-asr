from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import SpeechManifestDataset, collate_mel_batch, collate_ssl_batch
from src.decoder import ctc_beam_search_decode, greedy_ctc_decode
from src.metrics import cer, wer
from src.models import MelCTCModel, SSLCTCModel
from src.trainer import ctc_loss_from_logits, evaluate_ctc, train_one_epoch, train_with_finetune, write_metrics
from src.utils import ensure_dir, read_jsonl, set_seed
from src.vocab import build_vocab


def _make_loader(manifest, vocab, cfg, model_type, shuffle):
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    max_seconds_key = "max_audio_seconds" if shuffle else "max_eval_audio_seconds"
    ds = SpeechManifestDataset(
        manifest,
        vocab=vocab,
        sample_rate=int(model_cfg.get("sample_rate", 16000)),
        max_audio_seconds=float(train_cfg.get(max_seconds_key, train_cfg.get("max_audio_seconds", 20))),
    )
    if model_type == "mel_ctc":
        collate = lambda batch: collate_mel_batch(
            batch,
            sample_rate=int(model_cfg.get("sample_rate", 16000)),
            n_mels=int(model_cfg.get("n_mels", 80)),
        )
    else:
        collate = collate_ssl_batch
    return DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate,
    )


def _build_model(cfg, vocab_size):
    m = cfg["model"]
    if m["type"] == "mel_ctc":
        return MelCTCModel(
            vocab_size=vocab_size,
            n_mels=int(m.get("n_mels", 80)),
            encoder_dim=int(m.get("encoder_dim", 256)),
            encoder_layers=int(m.get("encoder_layers", 4)),
            dropout=float(m.get("dropout", 0.1)),
        )
    if m["type"] == "ssl_ctc":
        return SSLCTCModel(
            pretrained_name=m["pretrained_name"],
            vocab_size=vocab_size,
            freeze_encoder=bool(m.get("freeze_encoder", True)),
            hidden_layer=str(m.get("hidden_layer", "last")),
            dropout=float(m.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown model type: {m['type']}")


def _save_predictions(model, loader, vocab, device, out_path, blank_id):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            if "input_values" in batch:
                logits = model(batch["input_values"].to(device), batch["attention_mask"].to(device))
            else:
                logits = model(batch["features"].to(device), batch["feature_lengths"].to(device))
            hyps = greedy_ctc_decode(logits, vocab, blank_id=blank_id)
            for row, ref, hyp in zip(batch["rows"], batch["texts"], hyps):
                rows.append(
                    {
                        "id": row["id"],
                        "ref": ref,
                        "hyp": hyp,
                        "wer": wer(ref, hyp),
                        "cer": cer(ref, hyp),
                    }
                )
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ref", "hyp", "wer", "cer"])
        writer.writeheader()
        writer.writerows(rows)


def _save_predictions_lm(model, loader, vocab, device, out_path, blank_id, lm_path, beam_width, lm_alpha, lm_beta):
    """Save predictions using LM-enhanced beam search (slower than greedy)."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            if "input_values" in batch:
                logits = model(batch["input_values"].to(device), batch["attention_mask"].to(device))
            else:
                logits = model(batch["features"].to(device), batch["feature_lengths"].to(device))
            hyps = ctc_beam_search_decode(
                logits, vocab, kenlm_path=lm_path,
                beam_width=beam_width, alpha=lm_alpha, beta=lm_beta,
                blank_id=blank_id,
            )
            for row, ref, hyp in zip(batch["rows"], batch["texts"], hyps):
                rows.append(
                    {
                        "id": row["id"],
                        "ref": ref,
                        "hyp": hyp,
                        "wer": wer(ref, hyp),
                        "cer": cer(ref, hyp),
                    }
                )
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ref", "hyp", "wer", "cer"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = ensure_dir(config["output_dir"])
    set_seed(int(config.get("seed", 1337)))

    train_rows = read_jsonl(config["train_manifest"])
    vocab = build_vocab(row["text"] for row in train_rows)
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    device_name = config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = _build_model(config, len(vocab)).to(device)
    train_loader = _make_loader(config["train_manifest"], vocab, config, config["model"]["type"], True)
    dev_loader = _make_loader(config["dev_manifest"], vocab, config, config["model"]["type"], False)
    test_loader = _make_loader(config["test_manifest"], vocab, config, config["model"]["type"], False)
    print(
        {
            "device": str(device),
            "train_examples": len(train_loader.dataset),
            "dev_examples": len(dev_loader.dataset),
            "test_examples": len(test_loader.dataset),
        }
    )

    train_cfg = config["training"]
    blank_id = int(config.get("decode", {}).get("blank_id", 0))
    decode_cfg = config.get("decode", {})

    # ---- LM path for beam-search eval (optional) ----
    lm_path = decode_cfg.get("lm_path", None)
    lm_beam = int(decode_cfg.get("lm_beam_width", 50))
    lm_alpha = float(decode_cfg.get("lm_alpha", 0.5))
    lm_beta = float(decode_cfg.get("lm_beta", 1.5))

    # ---- Two-phase fine-tuning or standard training ----
    finetune_start = int(train_cfg.get("finetune_start_epoch", 0))
    if finetune_start > 0:
        print(f"[setup] two-phase training: freeze epochs 1–{finetune_start - 1}, "
              f"unfreeze from epoch {finetune_start}")
        history, best_dev = train_with_finetune(
            model, train_loader, dev_loader, vocab, device, config, blank_id=blank_id,
        )
    else:
        # Standard training (all params trainable or all frozen — no mid-training change)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(train_cfg["learning_rate"]),
        )
        best_dev = float("inf")
        history = []
        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device,
                blank_id=blank_id,
                grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            )
            dev = evaluate_ctc(model, dev_loader, vocab, device, blank_id=blank_id)
            row = {
                "epoch": epoch, "train_loss": train_loss,
                "dev_wer": dev.wer, "dev_cer": dev.cer, "dev_rtf": dev.rtf,
            }
            history.append(row)
            print(row)
            if dev.wer < best_dev:
                best_dev = dev.wer
                torch.save({"model": model.state_dict(), "vocab": vocab, "config": config},
                           out / "best.pt")

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    dev = evaluate_ctc(model, dev_loader, vocab, device, blank_id=blank_id)
    test = evaluate_ctc(model, test_loader, vocab, device, blank_id=blank_id)
    metrics = {
        "run_name": config["run_name"],
        "best_dev_wer": dev.wer,
        "best_dev_cer": dev.cer,
        "test_wer": test.wer,
        "test_cer": test.cer,
        "dev_rtf": dev.rtf,
        "test_rtf": test.rtf,
        "history": history,
    }

    # ---- Optional LM eval (saves separate metrics & predictions) ----
    if lm_path is not None:
        from pathlib import Path as _Path
        if _Path(lm_path).exists():
            print(f"[lm-eval] beam={lm_beam} alpha={lm_alpha} beta={lm_beta}")
            dev_lm = evaluate_ctc(model, dev_loader, vocab, device, blank_id=blank_id,
                                  kenlm_path=lm_path, beam_width=lm_beam,
                                  lm_alpha=lm_alpha, lm_beta=lm_beta)
            test_lm = evaluate_ctc(model, test_loader, vocab, device, blank_id=blank_id,
                                   kenlm_path=lm_path, beam_width=lm_beam,
                                   lm_alpha=lm_alpha, lm_beta=lm_beta)
            metrics["lm_dev_wer"] = dev_lm.wer
            metrics["lm_dev_cer"] = dev_lm.cer
            metrics["lm_test_wer"] = test_lm.wer
            metrics["lm_test_cer"] = test_lm.cer
            metrics["lm_dev_rtf"] = dev_lm.rtf
            metrics["lm_test_rtf"] = test_lm.rtf
            print(f"[lm-eval] test WER: greedy={test.wer:.3f} → +LM={test_lm.wer:.3f}")
            # Save LM predictions
            _save_predictions_lm(model, test_loader, vocab, device,
                                 out / "test_predictions_lm.csv",
                                 blank_id, lm_path, lm_beam, lm_alpha, lm_beta)
        else:
            print(f"[lm-eval] WARNING: LM path not found — {lm_path}")

    write_metrics(out / "metrics.json", metrics)
    _save_predictions(model, test_loader, vocab, device, out / "test_predictions.csv", blank_id)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
