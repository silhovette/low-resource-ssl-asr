from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data import prepare_split_manifest, select_by_hours
from src.utils import ensure_dir, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-split", default="dev-clean")
    parser.add_argument("--test-split", default="test-clean")
    parser.add_argument("--train-hours", nargs="*", type=float, default=[1.0, 5.0, 10.0])
    args = parser.parse_args()

    root = Path(args.librispeech_root)
    out = ensure_dir(args.output_dir)
    train_rows = prepare_split_manifest(root / "train-clean-100", out / "train-clean-100_full.jsonl")
    prepare_split_manifest(root / args.dev_split, out / f"{args.dev_split}.jsonl")
    prepare_split_manifest(root / args.test_split, out / f"{args.test_split}.jsonl")
    for hours in args.train_hours:
        rows = select_by_hours(train_rows, hours)
        label = f"{hours:g}h"
        write_jsonl(out / f"train-clean-100_{label}.jsonl", rows)
        total = sum(float(row.get("duration", 0.0)) for row in rows) / 3600.0
        print(f"{label}: {len(rows)} utterances, {total:.2f} hours")
    print(f"wrote manifests to {out}")


if __name__ == "__main__":
    main()
