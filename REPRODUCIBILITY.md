# Reproducibility Checklist

## 1. Create CUDA Environment

```powershell
py -3.10 -m venv .venv_cuda
.\.venv_cuda\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv_cuda\Scripts\python.exe -m pip install transformers datasets huggingface_hub pyarrow soundfile jiwer pyyaml tqdm pytest
```

The local conda path was not used because its Python package cache produced
`CondaVerificationError`. The isolated venv avoids modifying `base`.

## 2. Prepare LibriSpeech Manifests

```powershell
.\.venv_cuda\Scripts\python.exe scripts\prepare_hf_librispeech_subset.py `
  --root data/hf_librispeech `
  --manifest-dir data/manifests `
  --train-hours 1 `
  --dev-items 3000 `
  --test-items 3000 `
  --cache-dir data/.hf_cache
```

This writes:

- `train-clean-100_1h_hf.jsonl`: 269 utterances, 1.004 hours
- `dev-clean_hf.jsonl`: 2703 utterances, 5.388 hours
- `test-clean_hf.jsonl`: 2620 utterances, 5.403 hours

## 3. Train 1h Systems

```powershell
.\.venv_cuda\Scripts\python.exe scripts\train_asr.py --config configs\mel_1h_hf_cuda.yaml
.\.venv_cuda\Scripts\python.exe scripts\train_asr.py --config configs\wav2vec2_1h_frozen_hf_cuda.yaml
.\.venv_cuda\Scripts\python.exe scripts\train_asr.py --config configs\hubert_1h_frozen_hf_cuda.yaml
.\.venv_cuda\Scripts\python.exe scripts\train_asr.py --config configs\wav2vec2_1h_layer6_hf_cuda.yaml
```

## 4. Evaluate Saved Checkpoints

The SSL results in the final table use the checkpoint metrics already written
under `outputs/*_hf_cuda/metrics.json`. A full 60-second-limit re-evaluation of
wav2vec2 unexpectedly ran for about an hour, so it was stopped. The retained
30-second-limit evaluation covers 2694/2703 `dev-clean` utterances and
2611/2620 `test-clean` utterances; only the longest 9 utterances in each split
are excluded. This keeps the result defensible without spending hours on an
unproductive rerun.

To export predictions from a checkpoint:

```powershell
.\.venv_cuda\Scripts\python.exe scripts\evaluate_checkpoint.py `
  --checkpoint outputs\hubert_1h_frozen_hf_cuda\best.pt `
  --split test `
  --predictions outputs\hubert_1h_frozen_hf_cuda\test_predictions.csv `
  --update-metrics
```

## 5. Regenerate Tables and Report

```powershell
.\.venv_cuda\Scripts\python.exe scripts\summarize_results.py --outputs outputs --report-dir results
cd report
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
bibtex build\main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
```

## Validation

```powershell
.\.venv_cuda\Scripts\python.exe -m pytest tests\test_metrics.py
.\.venv_cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
