from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .utils import normalize_text, read_jsonl, write_jsonl


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, Path.cwd().resolve())).as_posix()


def _collect_transcripts(split_root: Path) -> List[dict]:
    rows: List[dict] = []
    for txt in split_root.rglob("*.txt"):
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, transcript = parts
                flac = txt.with_name(utt_id + ".flac")
                wav = txt.with_name(utt_id + ".wav")
                audio = flac if flac.exists() else wav
                if not audio.exists():
                    continue
                rows.append(
                    {
                        "id": utt_id,
                        "audio_path": _portable_path(audio),
                        "text": transcript.strip(),
                        "text_norm": normalize_text(transcript),
                        "duration": _audio_duration_seconds(audio),
                    }
                )
    return rows


def _audio_duration_seconds(path: Path) -> float:
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        if info.sample_rate:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return 0.0


def build_manifest(split_dir: str | Path, output_path: str | Path) -> List[dict]:
    split_dir = Path(split_dir)
    rows = _collect_transcripts(split_dir)
    write_jsonl(output_path, rows)
    return rows


def select_by_hours(rows: Sequence[dict], target_hours: float) -> List[dict]:
    selected: List[dict] = []
    total_seconds = 0.0
    for row in rows:
        duration = float(row.get("duration", 0.0))
        if duration <= 0:
            continue
        if total_seconds / 3600.0 >= target_hours:
            break
        selected.append(dict(row))
        total_seconds += duration
    return selected


def prepare_split_manifest(split_root: str | Path, output_path: str | Path) -> List[dict]:
    rows = _collect_transcripts(Path(split_root))
    write_jsonl(output_path, rows)
    return rows
