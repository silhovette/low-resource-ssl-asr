from __future__ import annotations

import argparse
import random
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data import prepare_split_manifest
from src.utils import ensure_dir, write_jsonl


def _filter_rows(rows, max_seconds, max_items):
    rows = [row for row in rows if 0 < float(row.get("duration", 0.0)) <= max_seconds]
    rows.sort(key=lambda row: (row["id"], row["audio_path"]))
    return rows[:max_items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--train-items", type=int, default=120)
    parser.add_argument("--dev-items", type=int, default=40)
    parser.add_argument("--test-items", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    train_all = prepare_split_manifest(args.train_root, out / "pilot_train_all.jsonl")
    eval_all = prepare_split_manifest(args.eval_root, out / "pilot_eval_all.jsonl")
    train = _filter_rows(train_all, args.max_seconds, args.train_items)
    eval_rows = _filter_rows(eval_all, args.max_seconds, args.dev_items + args.test_items)
    dev = eval_rows[: args.dev_items]
    test = eval_rows[args.dev_items : args.dev_items + args.test_items]
    write_jsonl(out / "pilot_train.jsonl", train)
    write_jsonl(out / "pilot_dev.jsonl", dev)
    write_jsonl(out / "pilot_test.jsonl", test)
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        hours = sum(float(row.get("duration", 0.0)) for row in rows) / 3600.0
        print(f"pilot_{name}: {len(rows)} utterances, {hours:.3f} hours")


if __name__ == "__main__":
    main()
