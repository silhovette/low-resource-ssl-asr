from __future__ import annotations

import argparse
import os
from pathlib import Path
from io import BytesIO

import _bootstrap  # noqa: F401

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from src.utils import ensure_dir, normalize_text, write_jsonl


SPLIT_MAP = {
    "train-clean-100": "train.clean.100",
    "dev-clean": "validation.clean",
    "test-clean": "test.clean",
}


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, Path.cwd().resolve())).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/hf_librispeech")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--train-hours", type=float, default=1.0)
    parser.add_argument("--dev-items", type=int, default=200)
    parser.add_argument("--test-items", type=int, default=200)
    parser.add_argument("--cache-dir", default="data/.hf_cache")
    args = parser.parse_args()

    root = ensure_dir(args.root)
    manifest_dir = ensure_dir(args.manifest_dir)
    train = materialize_split(
        "train-clean-100",
        root,
        args.cache_dir,
        max_hours=args.train_hours,
        max_items=None,
    )
    dev = materialize_split("dev-clean", root, args.cache_dir, max_items=args.dev_items)
    test = materialize_split("test-clean", root, args.cache_dir, max_items=args.test_items)
    write_jsonl(manifest_dir / f"train-clean-100_{args.train_hours:g}h_hf.jsonl", train)
    write_jsonl(manifest_dir / "dev-clean_hf.jsonl", dev)
    write_jsonl(manifest_dir / "test-clean_hf.jsonl", test)
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        hours = sum(float(row["duration"]) for row in rows) / 3600.0
        print(f"{name}: {len(rows)} utterances, {hours:.3f} hours")


def materialize_split(
    split_name: str,
    root: Path,
    cache_dir: str,
    max_hours: float | None = None,
    max_items: int | None = None,
) -> list[dict]:
    rows = []
    seconds = 0.0
    split_root = ensure_dir(root / "LibriSpeech" / split_name)
    for parquet_path in _parquet_files(split_name, cache_dir):
        table = pq.read_table(parquet_path)
        data = table.to_pydict()
        for i in tqdm(range(table.num_rows), desc=f"materialize:{split_name}:{Path(parquet_path).name}"):
            if max_items is not None and len(rows) >= max_items:
                return rows
            if max_hours is not None and rows and seconds / 3600.0 >= max_hours:
                return rows
            audio = data["audio"][i]
            flac_bytes = audio["bytes"]
            duration = _duration_from_flac_bytes(flac_bytes)
            speaker = str(data["speaker_id"][i])
            chapter = str(data["chapter_id"][i])
            utt_id = data["id"][i]
            chapter_dir = ensure_dir(split_root / speaker / chapter)
            audio_path = chapter_dir / f"{utt_id}.flac"
            if not audio_path.exists():
                audio_path.write_bytes(flac_bytes)
            text = data["text"][i]
            rows.append(
                {
                    "id": utt_id,
                    "audio_path": _portable_path(audio_path),
                    "text": text,
                    "text_norm": normalize_text(text),
                    "duration": duration,
                }
            )
            seconds += duration
    return rows


def _duration_from_flac_bytes(flac_bytes: bytes) -> float:
    info = sf.info(BytesIO(flac_bytes))
    return float(info.frames) / float(info.samplerate)


def _parquet_files(split_name: str, cache_dir: str) -> list[str]:
    hf_split = SPLIT_MAP[split_name]
    prefix = f"all/{hf_split}/"
    files = [
        name
        for name in list_repo_files("openslr/librispeech_asr", repo_type="dataset")
        if name.startswith(prefix) and name.endswith(".parquet")
    ]
    files.sort()
    return [
        hf_hub_download(
            "openslr/librispeech_asr",
            filename=name,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        for name in files
    ]


if __name__ == "__main__":
    main()
