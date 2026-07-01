from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def _metric(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _fmt(value):
    if value is None or value == "":
        return "--"
    return f"{float(value):.3f}"


def _latex_escape(text):
    return str(text).replace("_", "\\_")


DISPLAY_NAMES = {
    "hubert_asr_large": "HuBERT ASR",
    "hubert_1h_frozen_hf_cuda": "HuBERT last",
    "hubert_1h_finetune6_hf_cuda": "HuBERT last (FT top-6)",
    "hubert_1h_lm_hf_cuda": "HuBERT last + LM",
    "mel_1h_hf_cuda": "Log-mel",
    "pilot_mel_ctc": "Log-mel CTC",
    "wav2vec2_asr_base_10m": "wav2vec2 ASR (10m)",
    "wav2vec2_1h_frozen_hf_cuda": "wav2vec2 last",
    "wav2vec2_1h_finetune6_hf_cuda": "wav2vec2 layer 6 (FT top-6)",
    "wav2vec2_1h_layer0_hf_cuda": "wav2vec2 layer 0",
    "wav2vec2_1h_layer3_hf_cuda": "wav2vec2 layer 3",
    "wav2vec2_1h_layer6_hf_cuda": "wav2vec2 layer 6",
    "wav2vec2_1h_layer6_lm_hf_cuda": "wav2vec2 layer 6 + LM",
    "wav2vec2_1h_layer9_hf_cuda": "wav2vec2 layer 9",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for metrics_path in Path(args.outputs).glob("**/metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
            if metrics_path.parent.parent.name == "pilot_ssl":
                continue
            if not str(metrics.get("run_name", "")).endswith("_hf_cuda"):
                continue
            runs.append(metrics)
    runs.sort(key=lambda row: row.get("run_name", ""))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "dev_wer", "dev_cer", "test_wer", "test_cer", "test_rtf"])
        for row in runs:
            writer.writerow(
                [
                    row.get("run_name", ""),
                    _metric(row, "best_dev_wer", "dev_wer"),
                    _metric(row, "best_dev_cer", "dev_cer"),
                    _metric(row, "test_wer"),
                    _metric(row, "test_cer"),
                    _metric(row, "test_rtf"),
                ]
            )
    table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Low-resource ASR performance using 1 hour of LibriSpeech train-clean-100.}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "System & Dev WER & Test WER & Test CER \\\\",
        "\\midrule",
    ]
    for row in runs:
        table_lines.append(
            f"{_latex_escape(DISPLAY_NAMES.get(row.get('run_name', ''), row.get('run_name', '')))} & {_fmt(_metric(row, 'best_dev_wer', 'dev_wer'))} & {_fmt(_metric(row, 'test_wer'))} & {_fmt(_metric(row, 'test_cer'))} \\\\"
        )
    table_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "report_tables.tex").write_text("\n".join(table_lines), encoding="utf-8")
    final_run_names = {row.get("run_name", "") for row in runs}
    prediction_files = [
        path
        for path in Path(args.outputs).glob("*/test_predictions.csv")
        if path.parent.name in final_run_names
    ]
    with (out / "error_examples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "id", "ref", "hyp", "wer", "cer"])
        for pred_path in prediction_files:
            run_name = pred_path.parent.name
            with pred_path.open("r", encoding="utf-8") as pf:
                for row in sorted(csv.DictReader(pf), key=lambda r: float(r["wer"]), reverse=True)[:10]:
                    writer.writerow([run_name, row["id"], row["ref"], row["hyp"], row["wer"], row["cer"]])
    print(f"wrote summary artifacts to {out}")


if __name__ == "__main__":
    main()
