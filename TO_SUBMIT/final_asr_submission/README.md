# Low-Resource SSL ASR Project

This project implements a low-resource automatic speech recognition study with
pretrained speech self-supervised representations. The main experiment uses
approximately 1 hour from LibriSpeech `train-clean-100` and evaluates on
LibriSpeech `dev-clean` and `test-clean` manifests prepared from the Hugging
Face `openslr/librispeech_asr` parquet mirror.

## Main Systems

- log-mel features + small character CTC baseline
- wav2vec 2.0 base hidden states + CTC head
- HuBERT base hidden states + CTC head
- layer ablation: wav2vec 2.0 final layer versus layer 6

## Environment

The completed CUDA runs used a local virtual environment:

```powershell
py -3.10 -m venv .venv_cuda
.\.venv_cuda\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv_cuda\Scripts\python.exe -m pip install transformers datasets huggingface_hub pyarrow soundfile jiwer pyyaml tqdm pytest
```

Verified runtime:

- PyTorch `2.7.1+cu128`
- CUDA available: yes
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU

## Data Preparation

Use the parquet-based helper instead of the slow OpenSLR tar download when the
network is unstable:

```powershell
.\.venv_cuda\Scripts\python.exe scripts\prepare_hf_librispeech_subset.py `
  --root data/hf_librispeech `
  --manifest-dir data/manifests `
  --train-hours 1 `
  --dev-items 3000 `
  --test-items 3000 `
  --cache-dir data/.hf_cache
```

Generated manifests:

- `data/manifests/train-clean-100_1h_hf.jsonl`: 269 utterances, 1.004 hours
- `data/manifests/dev-clean_hf.jsonl`: 2703 utterances, 5.388 hours
- `data/manifests/test-clean_hf.jsonl`: 2620 utterances, 5.403 hours

## Run Experiments

```powershell
.\.venv_cuda\Scripts\python.exe scripts\train_asr.py --config configs\mel_1h_hf_cuda.yaml
.\.venv_cuda\Scripts\python.exe scripts\train_asr.py --config configs\wav2vec2_1h_frozen_hf_cuda.yaml
.\.venv_cuda\Scripts\python.exe scripts\train_asr.py --config configs\hubert_1h_frozen_hf_cuda.yaml
.\.venv_cuda\Scripts\python.exe scripts\train_asr.py --config configs\wav2vec2_1h_layer6_hf_cuda.yaml
```

Or run the listed plan:

```powershell
.\.venv_cuda\Scripts\python.exe scripts\run_experiments.py --plan configs\experiment_plan_hf_cuda_1h.yaml
```

## Results

Regenerate summary tables and qualitative error examples:

```powershell
.\.venv_cuda\Scripts\python.exe scripts\summarize_results.py --outputs outputs --report-dir results
```

Current final metrics are in `results/summary.csv`. The report PDF is
`report/build/main.pdf`.

## Notes

Raw LibriSpeech audio and local virtual environments are intentionally excluded
from the submission package. Submit `TO_SUBMIT/final_asr_submission.zip` and
`TO_SUBMIT/final_report.pdf`.
