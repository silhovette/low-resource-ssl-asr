# 补充实验运行指南

本文档说明如何运行三项高优先级改进实验：**(1) Layer Sweep**、(2) **Fine-tuning**、(3) **n-gram LM 融合**。

所有代码修改已完成，你不需要改动任何源码。以下按依赖关系排列，建议**按顺序执行**。

---

## 环境准备

在运行任何实验之前，确保已安装所有依赖：

```bash
# 进入项目根目录
cd low-resource-ssl-asr

# 基础依赖（已有）
pip install torch==2.7.1 torchaudio==2.7.1
pip install transformers datasets soundfile jiwer pyyaml tqdm pytest

# 新增依赖（用于 LM 实验，Phase 4）
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install pyctcdecode
```

**说明**：如果你只想跑 Layer Sweep 和 Fine-tuning（Phase 2-3），不需要安装 `kenlm` 和 `pyctcdecode`。

---

## Phase 1: 确认基线结果已存在

```bash
ls outputs/mel_1h_hf_cuda/metrics.json
ls outputs/wav2vec2_1h_frozen_hf_cuda/metrics.json
ls outputs/wav2vec2_1h_layer6_hf_cuda/metrics.json
ls outputs/hubert_1h_frozen_hf_cuda/metrics.json
```

如果这 4 个文件都在，Phase 1 已完成，跳过。

---

## Phase 2: Layer Sweep — 扫描 wav2vec2 各层表征

### 目的
验证"中间层比最终层好"的趋势是否在 layer 0→3→6→9→12 上连续成立，让论文的 layer ablation 有完整数据支撑。

### 新增实验（3 个）
| 配置 | 说明 |
|------|------|
| `configs/wav2vec2_1h_layer0.yaml` | wav2vec2 第 0 层（最浅） |
| `configs/wav2vec2_1h_layer3.yaml` | wav2vec2 第 3 层 |
| `configs/wav2vec2_1h_layer9.yaml` | wav2vec2 第 9 层 |

已有的 `layer6` 和 `layer12`（last）配置不需要重跑。

### 运行

```bash
# 方式一：逐个跑（推荐，能看每个的训练进度）
python scripts/train_asr.py --config configs/wav2vec2_1h_layer0.yaml
python scripts/train_asr.py --config configs/wav2vec2_1h_layer3.yaml
python scripts/train_asr.py --config configs/wav2vec2_1h_layer9.yaml

# 方式二：用 sweep 脚本一键跑完 5 个 layer
python scripts/sweep_layers.py --run
```

每个实验约需 **10-15 分钟**（RTX 4060 8GB, batch_size=2, 5 epochs）。

### 收集结果 & 生成 Layer Sweep 图

```bash
# 收集各层结果到 results/layer_sweep.csv
python scripts/sweep_layers.py --collect

# 重新生成所有论文图（包含新的 layer_sweep.pdf）
python scripts/generate_figures.py
```

生成的 `report/figures/layer_sweep.pdf` 是 WER/CER vs Layer 的折线图，可以直接放进论文。

### 预期效果

Layer 0、3 的 WER 应该比 layer 6 差，layer 9 应该接近 layer 6。预期是一条 **U 形曲线**：中间层最优，最浅和最深都差。这会让论文的核心结论非常有说服力。

---

## Phase 3: Fine-tuning — 冻结预热 + 微调编码器

### 目的
回答"如果部分解冻编码器，WER 能改善多少？"——这是论文当前最大的逻辑缺口。

### 实验设计

| 配置 | 说明 |
|------|------|
| `configs/wav2vec2_1h_finetune6.yaml` | wav2vec2 layer6: 前 3 epoch 冻结 → 后 5 epoch 解冻顶部 6 层 |
| `configs/hubert_1h_finetune6.yaml` | HuBERT last: 前 3 epoch 冻结 → 后 5 epoch 解冻顶部 6 层 |

**两阶段训练**：
- **Phase A（epoch 1-3）**：编码器全冻结，只训练 CTC 头（~1.2M 参数），lr=0.001
- **Phase B（epoch 4-8）**：解冻顶部 6 层 Transformer（额外约 40M 参数可训练），lr=0.0001

总共 8 个 epoch，与冻结版 (5 epoch) 做公平对比时，建议也跑一个 **8-epoch 全冻结版** 作为对照：

```bash
# 8 个 epoch 的全冻结对照（不微调）
# 只需修改 configs/wav2vec2_1h_layer6_hf_cuda.yaml 的 epochs: 5 → epochs: 8
# 然后把 output_dir 改为 outputs/wav2vec2_1h_layer6_8ep_hf_cuda
```

### 运行

```bash
# wav2vec2 fine-tune（约 25 分钟）
python scripts/train_asr.py --config configs/wav2vec2_1h_finetune6.yaml

# HuBERT fine-tune（约 25 分钟）
python scripts/train_asr.py --config configs/hubert_1h_finetune6.yaml
```

训练日志中你会看到：
```
[setup] two-phase training: freeze epochs 1–3, unfreeze from epoch 4
...
[finetune] unfroze 6 transformer layers (trainable params: 42,123,456)
```

### 预期效果

| 系统 | 冻结 WER | Fine-tune WER (预期) |
|------|---------|---------------------|
| wav2vec2 layer6 | 0.667 | **0.30–0.45** |
| HuBERT last | 0.662 | **0.28–0.42** |

Fine-tuning 应该是**最大的单次 WER 改善**。关键调参：
- 如果过拟合（dev WER 先降后升）：减少 `finetune_layers` 到 3
- 如果不收敛（dev WER 不变）：增大 `finetune_lr` 到 0.0003

---

## Phase 4: n-gram LM 融合 — CTC Beam Search + KenLM

### 目的
展示加入简单语言模型后的改善。论文当前用纯贪婪解码，加了 LM 后 WER 应下降 3-10 个绝对百分点。

### 第一步：训练 KenLM 语言模型

```bash
# 从训练集文本训练字符级 5-gram LM
python scripts/train_lm.py \
    --manifest data/manifests/train-clean-100_1h_hf.jsonl \
    --output lm/char_5gram.arpa \
    --order 5

# 可选：如果有完整 LibriSpeech LM 语料，训一个更强的词级 4-gram：
# python scripts/train_lm.py \
#     --corpus /path/to/librispeech-lm-norm.txt \
#     --output lm/word_4gram.arpa \
#     --order 4 --word-level
```

这会输出 `lm/char_5gram.arpa`（约 500KB-2MB）。

**说明**：由于只有 1 小时训练文本，这个 LM 较小。如果你能拿到 LibriSpeech 完整 LM 语料（`librispeech-lm-norm.txt`，公开可下载），效果会好很多。用 `--word-level` 训练词级 LM 通常比字符级更好——但 pyctcdecode 需要配合子词 vocab，这里建议先用字符级快速验证。

### 第二步：用 LM 增强的 Beam Search 评估

训练阶段**仍然用贪婪解码**（和之前一样），只在最终评估时切换为 beam search + LM。

```bash
# wav2vec2 layer6 + LM 评估
python scripts/train_asr.py --config configs/wav2vec2_1h_layer6_lm.yaml

# HuBERT + LM 评估
python scripts/train_asr.py --config configs/hubert_1h_lm.yaml
```

注意：这两个 config 引用的 `decode.lm_path: lm/char_5gram.arpa` 必须存在。

### 如何工作

`train_asr.py` 在训练结束后会：
1. 加载最优模型（贪婪解码选出的）
2. 额外跑一遍 LM 增强的 beam search 评估
3. 在 `metrics.json` 中写入 `lm_test_wer`, `lm_test_cer` 等字段
4. 保存 `test_predictions_lm.csv`（含 LM 解码结果）

输出日志中你会看到：
```
[lm-eval] beam=50 alpha=0.5 beta=1.5
[lm-eval] test WER: greedy=0.662 → +LM=0.58x
```

### 调参

LM 融合有两个关键超参，在 config 中修改：
- `lm_alpha`（LM 权重，默认 0.5）：越大 LM 权重越高。调到 0.3–1.0 试试
- `lm_beta`（插入奖励，默认 1.5）：越大越偏向更少的词。调到 0.0–3.0 试试

建议在 dev set 上扫描 `alpha ∈ [0.3, 0.5, 0.7, 1.0]` 和 `beta ∈ [0.5, 1.0, 1.5, 2.0]`。

### 预期效果

| 系统 | 贪婪 WER | +LM WER (预期) |
|------|---------|---------------|
| wav2vec2 layer6 | 0.667 | **0.58–0.64** |
| HuBERT last | 0.662 | **0.57–0.63** |

LM 改善幅度取决于 LM 质量。由于我们只有 1 小时训练文本，LM 较小，改善可能只有 3-5 个百分点。用完整 LibriSpeech LM 语料（约 40M 词）训练的话，改善可到 5-10 个百分点。

---

## Phase 5: 汇总与论文更新

### 汇总所有结果

```bash
# 生成汇总表和 LaTeX 表格
python scripts/summarize_results.py --outputs outputs --report-dir results

# 重新生成全部论文图
python scripts/generate_figures.py

# 编译论文
cd report
pdflatex -output-directory build -interaction=nonstopmode main.tex
bibtex build/main
pdflatex -output-directory build -interaction=nonstopmode main.tex
pdflatex -output-directory build -interaction=nonstopmode main.tex
```

### 论文修改建议

运行完所有实验后，以下是你需要在论文中修改的地方：

1. **Results 表格**：`results/report_tables.tex` 已自动更新，但需在 `\input` 前确认内容正确
2. **新增 Layer Sweep 图**：在论文中 `\ref{fig:layer}` 附近增加 `\ref{fig:layersweep}` 的引用
3. **Fine-tuning 讨论**：把 `\section{Discussion}` 中的 "Why Not Fine-Tune?" 改成 "Fine-tuning Results"，加入实测数据
4. **Limitations**：LM 部分从 "未做" 改为 "已做初步实验"
5. **Conclusion**：更新量化结论

论文中新增图片的 LaTeX 引用：

```latex
% Layer sweep figure
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{\figdir/layer_sweep.pdf}
\caption{WER and CER as a function of wav2vec2 hidden layer index.
Intermediate layers (3--9) consistently outperform the shallowest and
deepest layers, confirming the U-shaped pattern reported in prior work.}
\label{fig:layersweep}
\end{figure*}
```

---

## 时间估算

| Phase | 内容 | 预计耗时 (RTX 4060) |
|-------|------|-------------------|
| Phase 2 | 3 个 layer sweep 实验 | 30-45 分钟 |
| Phase 3 | 2 个 fine-tuning 实验 | 50-60 分钟 |
| Phase 4 | 训 LM + 2 个 LM eval 实验 | 5 + 30 分钟 |
| Phase 5 | 汇总 + 编译 | 2 分钟 |
| **总计** | | **约 2-2.5 小时** |

所有实验可串行跑，一个下午完成。

---

## 故障排查

### CUDA OOM（显存不足）
- 减小 `batch_size` 到 1
- 减小 `max_audio_seconds` 到 12
- Fine-tuning 比纯冻结吃更多显存（梯度要存），如果 OOM 减少 `finetune_layers` 到 3

### kenlm 安装失败
- macOS: `brew install kenlm` 然后 `pip install https://github.com/kpu/kenlm/archive/master.zip`
- Linux: 确保装了 `cmake`, `boost`, `eigen`
- 如果实在装不上，跳过 Phase 4，Phase 2-3 不受影响

### pyctcdecode 报错
- 确保 KenLM `.arpa` 文件路径正确且文件存在
- pyctcdecode 只在最终评估时使用，不影响训练

### 训练不收敛
- 确认 `learning_rate` 和 `finetune_lr` 数量级正确
- Fine-tuning 从 epoch 4 开始，如果前 3 epoch dev WER 还很高（>0.9），说明预热不够
- 检查 CUDA 是否可用：`python -c "import torch; print(torch.cuda.is_available())"`
