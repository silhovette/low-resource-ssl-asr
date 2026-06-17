from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8"))
    for cfg in plan["runs"]:
        print(f"running: {cfg}")
        subprocess.run([sys.executable, "scripts/train_asr.py", "--config", cfg], check=True)


if __name__ == "__main__":
    main()
