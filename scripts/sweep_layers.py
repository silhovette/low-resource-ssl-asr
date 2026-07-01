"""Run layer-sweep experiments and collect results into a summary CSV.

Usage
-----
  # Run all layer sweep experiments (5 configs):
  python scripts/sweep_layers.py --run

  # Collect results from already-completed runs:
  python scripts/sweep_layers.py --collect
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAYER_CONFIGS = [
    ("layer0",  "configs/wav2vec2_1h_layer0.yaml", "outputs/wav2vec2_1h_layer0_hf_cuda"),
    ("layer3",  "configs/wav2vec2_1h_layer3.yaml", "outputs/wav2vec2_1h_layer3_hf_cuda"),
    ("layer6",  "configs/wav2vec2_1h_layer6_hf_cuda.yaml", "outputs/wav2vec2_1h_layer6_hf_cuda"),
    ("layer9",  "configs/wav2vec2_1h_layer9.yaml", "outputs/wav2vec2_1h_layer9_hf_cuda"),
    ("layer12", "configs/wav2vec2_1h_frozen_hf_cuda.yaml", "outputs/wav2vec2_1h_frozen_hf_cuda"),
]


def run_all():
    """Run each layer config sequentially."""
    for name, cfg, outdir in LAYER_CONFIGS:
        print(f"\n{'='*60}\n  LAYER SWEEP: {name}\n{'='*60}")
        subprocess.run(
            [sys.executable, "scripts/train_asr.py", "--config", cfg],
            check=True,
        )


def collect() -> dict:
    """Collect dev/test WER/CER from each output directory."""
    results = {}
    for name, cfg, outdir in LAYER_CONFIGS:
        metrics_path = Path(outdir) / "metrics.json"
        if not metrics_path.exists():
            print(f"  [SKIP] {name}: no metrics.json in {outdir}")
            results[name] = None
            continue
        m = json.loads(metrics_path.read_text("utf-8"))
        results[name] = {
            "layer": int(name.replace("layer", "")),
            "test_wer": m.get("test_wer", m.get("best_dev_wer", 1.0)),
            "test_cer": m.get("test_cer", m.get("best_dev_cer", 1.0)),
            "dev_wer": m.get("best_dev_wer", 1.0),
            "dev_cer": m.get("best_dev_cer", 1.0),
        }
        print(f"  layer {results[name]['layer']}: dev WER={results[name]['dev_wer']:.3f}  test WER={results[name]['test_wer']:.3f}")

    # Save CSV
    out_csv = Path("results/layer_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "dev_wer", "dev_cer", "test_wer", "test_cer"])
        writer.writeheader()
        for name, r in results.items():
            if r:
                writer.writerow(r)
    print(f"\nSaved to {out_csv}")
    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run all layer sweep experiments")
    group.add_argument("--collect", action="store_true", help="Collect results from existing outputs")
    args = parser.parse_args()

    if args.run:
        run_all()
    else:
        collect()


if __name__ == "__main__":
    main()
