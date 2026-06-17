from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .utils import read_jsonl
from .vocab import text_to_ids


class SpeechManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        vocab: Sequence[str] | None = None,
        sample_rate: int = 16000,
        max_audio_seconds: float | None = None,
    ):
        self.rows = read_jsonl(manifest_path)
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        if max_audio_seconds is not None:
            self.rows = [
                row
                for row in self.rows
                if float(row.get("duration", 0.0)) <= 0.0 or float(row.get("duration", 0.0)) <= max_audio_seconds
            ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        item = dict(row)
        waveform, sample_rate = torchaudio.load(row["audio_path"])
        waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        item["waveform"] = waveform
        if self.vocab is not None:
            item["label_ids"] = text_to_ids(row["text"], self.vocab)
        return item


def collate_text_batch(batch: List[dict]) -> dict:
    texts = [x.get("text_norm") or x["text"] for x in batch]
    ids = [x["label_ids"] for x in batch]
    max_len = max(len(x) for x in ids) if ids else 0
    labels = torch.full((len(ids), max_len), fill_value=-100, dtype=torch.long)
    label_lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    for i, seq in enumerate(ids):
        labels[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {
        "texts": texts,
        "label_ids": labels,
        "label_lengths": label_lengths,
        "rows": batch,
    }


def collate_ssl_batch(batch: List[dict]) -> dict:
    base = collate_text_batch(batch)
    lengths = torch.tensor([len(x["waveform"]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    inputs = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        wav = item["waveform"]
        inputs[i, : wav.numel()] = wav
        mask[i, : wav.numel()] = 1
    base.update({"input_values": inputs, "attention_mask": mask, "input_lengths": lengths})
    return base


def collate_mel_batch(batch: List[dict], sample_rate: int = 16000, n_mels: int = 80) -> dict:
    base = collate_text_batch(batch)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    feats = []
    lengths = []
    for item in batch:
        feat = mel(item["waveform"]).transpose(0, 1)
        feat = torch.log1p(feat)
        feats.append(feat)
        lengths.append(feat.shape[0])
    max_len = max(lengths)
    feat_dim = feats[0].shape[1]
    padded = torch.zeros(len(batch), max_len, feat_dim)
    for i, feat in enumerate(feats):
        padded[i, : feat.shape[0]] = feat
    base.update({"features": padded, "feature_lengths": torch.tensor(lengths, dtype=torch.long)})
    return base
