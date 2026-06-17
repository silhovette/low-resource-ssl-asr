from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
import yaml

from scripts.train_asr import _build_model, _make_loader
from src.decoder import greedy_ctc_decode
from src.metrics import cer, wer
from src.trainer import evaluate_ctc, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--update-metrics", action="store_true")
    parser.add_argument("--max-eval-audio-seconds", type=float, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    vocab = ckpt["vocab"]
    if args.max_eval_audio_seconds is not None:
        config.setdefault("training", {})["max_eval_audio_seconds"] = args.max_eval_audio_seconds

    device_name = config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = _build_model(config, len(vocab)).to(device)
    model.load_state_dict(ckpt["model"])

    manifest = config[f"{args.split}_manifest"]
    loader = _make_loader(manifest, vocab, config, config["model"]["type"], False)
    result = evaluate_ctc(model, loader, vocab, device, blank_id=int(config.get("decode", {}).get("blank_id", 0)))
    print(
        json.dumps(
            {
                "run_name": config["run_name"],
                "split": args.split,
                "wer": result.wer,
                "cer": result.cer,
                "num_examples": result.num_examples,
                "rtf": result.rtf,
            },
            indent=2,
        )
    )

    pred_path = Path(args.predictions) if args.predictions else checkpoint_path.parent / f"{args.split}_predictions.csv"
    _save_predictions(model, loader, vocab, device, pred_path, int(config.get("decode", {}).get("blank_id", 0)))

    metrics_path = checkpoint_path.parent / "metrics.json"
    if args.update_metrics and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics[f"{args.split}_wer"] = result.wer
        metrics[f"{args.split}_cer"] = result.cer
        metrics[f"{args.split}_rtf"] = result.rtf
        if args.split == "dev":
            metrics["best_dev_wer"] = result.wer
            metrics["best_dev_cer"] = result.cer
        write_metrics(metrics_path, metrics)


def _save_predictions(model, loader, vocab, device, out_path: Path, blank_id: int) -> None:
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
                rows.append({"id": row["id"], "ref": ref, "hyp": hyp, "wer": wer(ref, hyp), "cer": cer(ref, hyp)})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ref", "hyp", "wer", "cer"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
