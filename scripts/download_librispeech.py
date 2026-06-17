from __future__ import annotations

import argparse
import csv
import os
import shutil
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401

from datasets import load_dataset
import requests
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--splits", nargs="+", default=["train-clean-100", "dev-clean", "test-clean"])
    parser.add_argument("--source", choices=["hf", "openslr", "auto"], default="auto")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.source in {"auto", "hf"}:
        try:
            _download_from_hf(root, args.splits)
            return
        except Exception as exc:
            if args.source == "hf":
                raise
            print(f"HF download failed, falling back to OpenSLR: {exc}")

    _download_from_openslr(root, args.splits)


def _download_from_hf(root: Path, splits: list[str]) -> None:
    for split in splits:
        hf_split = _hf_split_name(split)
        rows = load_dataset(
            "openslr/librispeech_asr",
            "all",
            split=hf_split,
            cache_dir=str(root / ".hf_cache"),
        )
        out_dir = root / "LibriSpeech" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in tqdm(rows, desc=f"hf:{split}"):
            speaker = str(row["speaker_id"])
            chapter = str(row["chapter_id"])
            utt_id = row["id"]
            chapter_dir = out_dir / speaker / chapter
            chapter_dir.mkdir(parents=True, exist_ok=True)
            audio_path = chapter_dir / f"{utt_id}.flac"
            audio_path.write_bytes(row["audio"]["bytes"])
            txt_path = chapter_dir / f"{speaker}-{chapter}.txt"
            if not txt_path.exists():
                txt_path.write_text("", encoding="utf-8")
            with txt_path.open("a", encoding="utf-8") as f:
                f.write(f"{utt_id} {row['text']}\n")
        print(f"wrote {split} to {out_dir}")


def _hf_split_name(split: str) -> str:
    mapping = {
        "train-clean-100": "train.clean.100",
        "dev-clean": "validation.clean",
        "test-clean": "test.clean",
    }
    return mapping.get(split, split.replace("-", "."))


def _download_from_openslr(root: Path, splits: list[str]) -> None:
    urls = {
        "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    }
    for split in splits:
        url = urls[split]
        archive = root / f"{split}.tar.gz"
        if not archive.exists():
            _download_file(url, archive)
        extract_root = root / "LibriSpeech"
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=root)
        print(f"extracted {split} under {root}")


def _download_file(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


if __name__ == "__main__":
    main()
