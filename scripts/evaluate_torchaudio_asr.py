from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
import torchaudio
from tqdm import tqdm

from src.metrics import cer, wer
from src.utils import ensure_dir, normalize_text, read_jsonl


class GreedyDecoder:
    def __init__(self, labels):
        self.labels = labels
        self.blank = 0

    def __call__(self, emissions):
        ids = torch.argmax(emissions, dim=-1)
        tokens = []
        prev = None
        for idx in ids.tolist():
            if idx == self.blank:
                prev = idx
                continue
            if idx == prev:
                continue
            token = self.labels[idx]
            tokens.append(" " if token == "|" else token.lower())
            prev = idx
        return normalize_text("".join(tokens))


def _load_audio(path, sample_rate):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bundles",
        nargs="+",
        default=["WAV2VEC2_ASR_BASE_10M", "WAV2VEC2_ASR_BASE_100H"],
    )
    parser.add_argument("--max-items", type=int, default=40)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)[: args.max_items]
    out = ensure_dir(args.output_dir)
    summary = []
    for bundle_name in args.bundles:
        bundle = getattr(torchaudio.pipelines, bundle_name)
        model = bundle.get_model().eval()
        labels = bundle.get_labels()
        decoder = GreedyDecoder(labels)
        pred_rows = []
        total_wer = total_cer = total_audio = total_infer = 0.0
        with torch.inference_mode():
            for row in tqdm(rows, desc=bundle_name):
                waveform = _load_audio(row["audio_path"], bundle.sample_rate)
                audio_seconds = waveform.shape[-1] / bundle.sample_rate
                start = time.perf_counter()
                emissions, _ = model(waveform)
                total_infer += time.perf_counter() - start
                hyp = decoder(emissions[0])
                ref = row["text_norm"]
                row_wer = wer(ref, hyp)
                row_cer = cer(ref, hyp)
                total_wer += row_wer
                total_cer += row_cer
                total_audio += audio_seconds
                pred_rows.append(
                    {
                        "id": row["id"],
                        "ref": ref,
                        "hyp": hyp,
                        "wer": row_wer,
                        "cer": row_cer,
                    }
                )
        denom = max(1, len(rows))
        metrics = {
            "run_name": bundle_name.lower(),
            "bundle": bundle_name,
            "num_examples": len(rows),
            "test_wer": total_wer / denom,
            "test_cer": total_cer / denom,
            "test_rtf": total_infer / total_audio if total_audio > 0 else 0.0,
        }
        run_dir = ensure_dir(out / bundle_name.lower())
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "ref", "hyp", "wer", "cer"])
            writer.writeheader()
            writer.writerows(pred_rows)
        summary.append(metrics)
        print(json.dumps(metrics, indent=2))
    (out / "pilot_asr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
