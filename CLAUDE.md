# CLAUDE.md — Low-Resource SSL ASR 项目完全解析

## 一、项目总览

这是一个**课程大作业**，选的是"方向一：基于语音自监督表征的低资源 ASR"。简单来说，目标是：**给定一段英语语音，系统自动输出对应的文字**。核心约束是只能使用极少量的标注数据（约 1 小时），然后利用预训练好的"语音自监督模型"来弥补数据的不足。

项目已完成全部代码实现、实验训练、结果汇总和论文撰写。最终提交物在 `TO_SUBMIT/final_asr_submission.zip` 和 `TO_SUBMIT/final_report.pdf`。

**小组成员**（来自 `report/main.tex`）：骆渝骅 (524531910035)、胡昊旻 (524531910040)、韩岳成 (524531910029)。

---

## 二、背景知识（面向初学者）

### 2.1 什么是 ASR？

ASR = Automatic Speech Recognition（自动语音识别），就是把音频信号转成文字。比如你对着手机说话，手机把它转成文字——这就是 ASR。

### 2.2 什么是"低资源"？

通常训练一个好的语音识别系统需要成百上千小时的标注数据（音频 + 对应文字）。"低资源"指的是只有很少的标注数据——本项目只用 **1 小时**的 LibriSpeech 训练数据。这是一个非常极限的设定，类似于"只有一本有声书的量，能不能训练出一个勉强可用的语音识别系统？"

### 2.3 什么是"语音自监督表征"（SSL）？

这是项目的核心概念。自监督学习（Self-Supervised Learning, SSL）的思路是：先让模型在海量**无标注**音频上"自学"（不需要文字标签，只需要音频本身），学会理解语音的基本结构（音素、音节等）。这就像一个人听了大量外语，虽然不知道具体意思，但已经能分辨不同的发音。

这个"自学"阶段产出的是一个 **预训练模型**（pretrained model）。给定一段音频波形，这个模型能输出一串向量（称为 hidden states 或"表征/representations"），这些向量捕获了音频中的语音信息。

本项目使用了两种预训练模型：
- **wav2vec 2.0**（Meta/FAIR 出品，2020年）：用对比学习的方式在海量无标注音频上预训练
- **HuBERT**（Meta/FAIR 出品，2021年）：改进版，用聚类的方式生成"伪标签"来预训练

这些模型都是公开的，从 Hugging Face 模型库加载（`facebook/wav2vec2-base` 和 `facebook/hubert-base-ls960`）。

### 2.4 什么是 CTC？

CTC = Connectionist Temporal Classification，是一种常用于语音识别的损失函数。它的核心问题是：**音频的帧率远高于文字的速率**（比如 1 秒音频可能有 50 帧特征，但对应的文字可能只有 2 个字符）。CTC 不需要精确知道每帧对应哪个字符，而是让模型自由学习对齐关系。它引入了一个特殊的"空白"符号（blank），解码时去掉连续重复的字符和空白即可得到文字。

### 2.5 什么是 WER 和 CER？

- **WER (Word Error Rate，词错误率)**：把识别结果和参考答案按词对齐，计算替换(Substitution)、删除(Deletion)、插入(Insertion) 错误的总和除以参考词数。越低越好，0 表示完美。
- **CER (Character Error Rate，字符错误率)**：同上，但是按**字符**（字母）来计算。

---

## 三、项目文件结构详解

```
low-resource-ssl-asr/
├── src/                          # 核心源代码库
│   ├── __init__.py               # Python 包标记
│   ├── models.py                 # 模型定义
│   ├── data.py                   # 数据预处理（manifest 构建）
│   ├── dataset.py                # PyTorch Dataset 和 DataLoader
│   ├── decoder.py                # CTC 贪婪解码器
│   ├── metrics.py                # WER/CER 评价指标
│   ├── trainer.py                # 训练/评估逻辑
│   ├── utils.py                  # 工具函数
│   └── vocab.py                  # 字符级词表构建
│
├── scripts/                      # 可执行脚本
│   ├── _bootstrap.py             # 路径引导（让 scripts 能找到 src）
│   ├── train_asr.py              # 主训练脚本
│   ├── run_experiments.py        # 批量运行实验
│   ├── evaluate_checkpoint.py    # 加载已保存的模型并评估
│   ├── prepare_hf_librispeech_subset.py  # 数据准备
│   ├── summarize_results.py      # 汇总结果、生成 LaTeX 表格
│   ├── smoke_test.py             # 冒烟测试（快速验证管线是否正常）
│   ├── prepare_librispeech.py    # 备用数据准备脚本（从原始 OpenSLR 下载）
│   ├── prepare_pilot_data.py     # 早期试点数据准备
│   └── evaluate_torchaudio_asr.py # 使用 torchaudio 内置 ASR 评估
│
├── configs/                      # YAML 配置文件
│   ├── experiment_plan.yaml      # 实验计划（列出所有要跑的配置）
│   ├── experiment_plan_hf_cuda_1h.yaml  # 同上（CUDA 版本）
│   ├── mel_1h_hf_cuda.yaml       # Log-mel 基线配置
│   ├── wav2vec2_1h_frozen_hf_cuda.yaml  # wav2vec2 最后一层配置
│   ├── wav2vec2_1h_layer6_hf_cuda.yaml  # wav2vec2 第6层配置（消融实验）
│   └── hubert_1h_frozen_hf_cuda.yaml    # HuBERT 配置
│
├── outputs/                      # 训练输出（每个实验一个子目录）
│   ├── mel_1h_hf_cuda/           # 包含 metrics.json, test_predictions.csv, vocab.json, config.yaml
│   ├── wav2vec2_1h_frozen_hf_cuda/
│   ├── wav2vec2_1h_layer6_hf_cuda/
│   └── hubert_1h_frozen_hf_cuda/
│
├── results/                      # 最终汇总结果
│   ├── summary.csv               # 所有实验的汇总指标
│   ├── report_tables.tex         # LaTeX 表格（可直接嵌入论文）
│   └── error_examples.csv        # 每个模型的最差预测样例（定性分析用）
│
├── data/
│   └── manifests/                # 数据清单（JSONL 格式）
│       ├── train-clean-100_1h_hf.jsonl    # 训练集：269条，1.004小时
│       ├── dev-clean_hf.jsonl             # 开发集：2703条，5.388小时
│       └── test-clean_hf.jsonl            # 测试集：2620条，5.403小时
│
├── report/                       # 论文
│   ├── main.tex                  # LaTeX 主文件（ICASSP 2026 格式）
│   ├── refs.bib                  # 参考文献
│   ├── spconf.sty                # ICASSP 样式文件
│   ├── IEEEbib.bst               # 参考文献格式
│   └── build/main.pdf            # 编译好的 PDF
│
├── tests/
│   └── test_metrics.py           # WER/CER 的单元测试
│
├── requirements.txt              # Python 依赖列表
├── README.md                     # 项目说明
├── REPRODUCIBILITY.md            # 复现步骤
├── SUBMISSION.md                 # 提交内容清单
├── background.md                 # 课程作业完整背景（含 TTS 方向）
├── project_requirements.md       # ASR 方向的作业要求
├── TO_SUBMIT/                    # 待提交的打包材料
└── .gitignore                    # 排除虚拟环境、缓存、原始音频、模型权重
```

---

## 四、系统架构详解

### 4.1 整体流程

```
原始音频 (.flac)
    │
    ▼
语音波形 → [特征提取] → 特征向量序列 → [编码器] → 隐层表征 → [CTC 头] → 字符概率分布 → [CTC 解码] → 文字
```

对于不同的系统，区别主要在于"特征提取"和"编码器"：

| 系统 | 特征提取 | 编码器 | 编码器是否训练 |
|------|---------|--------|----------------|
| **Log-mel 基线** | 80维 log-mel 频谱 | 4层全连接网络 | 是（从头训练） |
| **wav2vec2 last** | 原始波形（16kHz） | wav2vec2-base | **否（冻结）** |
| **wav2vec2 layer6** | 原始波形（16kHz） | wav2vec2-base 第6层 | **否（冻结）** |
| **HuBERT last** | 原始波形（16kHz） | hubert-base-ls960 | **否（冻结）** |

### 4.2 模型设计 (`src/models.py`)

#### MelCTCModel（基线系统）
- 输入：80 维 log-mel 频谱特征
- 编码器：4 层全连接网络 (Linear → ReLU → Dropout)，每层 256 维
- CTC 头：一个线性层，将 256 维映射到词表大小
- **特点**：没有任何预训练知识，完全从 1 小时标注数据上学习。这是一个"纯底限"对照。

#### SSLCTCModel（SSL 系统）
- 输入：原始 16kHz 音频波形
- 编码器：从 Hugging Face 加载预训练模型（wav2vec2 或 HuBERT）
- 冻结策略：编码器参数 `requires_grad = False`，只训练 CTC 投影头
- 投影头：Linear(768→768) → ReLU → Dropout → Linear(768→词表大小)
- **支持指定任意隐藏层**：
  - `"last"` 或 `"-1"`：使用最后一层的输出
  - `"mean_last_4"`：取最后 4 层的平均值
  - 数字如 `"6"`：使用第 6 层的输出（0-indexed）
- 由于编码器冻结，需要处理 CNN 下采样带来的长度变化（`_get_feat_extract_output_lengths`）

### 4.3 数据处理 (`src/data.py`, `src/dataset.py`, `src/vocab.py`)

**数据来源**：LibriSpeech（一个公开的英语朗读语音数据集），通过 Hugging Face 的 `openslr/librispeech_asr` parquet 格式镜像下载。选择这种格式而不是原始 tar 包是因为原始下载链路不稳定。

**数据清单（Manifest）**：JSONL 文件，每行是一条 JSON 记录：
```json
{"id": "61-70968-0000", "audio_path": "data/hf_librispeech/LibriSpeech/train-clean-100/61/70968/61-70968-0000.flac", "text": "THE original text", "text_norm": "the original text", "duration": 5.2}
```

**文本规范化** (`utils.py` 中的 `normalize_text`)：
1. 转小写
2. Unicode 弯引号 `'` 替换为 ASCII 单引号 `'`
3. 删除标点符号（保留字母、数字、空格、单引号）
4. 合并多余空格

**词表** (`vocab.py`)：
- 字符级（character-level）词表，不区分大小写
- 特殊符号：`<blank>`(CTC 空白)、`<pad>`(填充)、`<unk>`(未知字符)
- 从训练集的规范化文本中统计所有出现的字符
- 包含 26 个小写字母、单引号、空格

**批处理** (`dataset.py`)：
- `collate_mel_batch`：对 log-mel 系统，提取 Mel 频谱并填充到相同长度
- `collate_ssl_batch`：对 SSL 系统，填充原始波形到相同长度，生成 attention_mask
- 两者都共享 `collate_text_batch` 来填充标签序列

### 4.4 CTC 解码 (`src/decoder.py`)

**贪婪解码**（greedy decode）：
1. 对每一帧取概率最大的字符
2. 合并连续相同字符（CTC 的去重规则）
3. 去掉空白符号 `<blank>`
4. 将字符 ID 序列转回文本字符串

不需要额外的语言模型，解码非常快。

### 4.5 评价指标 (`src/metrics.py`)

- **WER**：先按空格分词，然后用编辑距离（Levenshtein 距离）计算词级错误率
- **CER**：去掉空格后在字符级计算编辑距离
- **aggregate_error_stats**：返回详细的替换/删除/插入/正确计数
- 编辑距离用经典的动态规划算法实现（O(n×m) 复杂度）

### 4.6 训练流程 (`src/trainer.py`, `scripts/train_asr.py`)

**训练循环**：
1. 构建字符级词表（从训练集）
2. 创建 DataLoader（训练/开发/测试）
3. AdamW 优化器，只优化 `requires_grad=True` 的参数
4. 每个 epoch：
   - 前向传播 → CTC loss → 反向传播 → 梯度裁剪 → 参数更新
   - 在开发集上评估 WER/CER
   - 保存最佳模型（按 dev WER）
5. 训练结束后加载最佳模型，在测试集上最终评估
6. 保存 metrics.json 和 test_predictions.csv

**关键配置参数**（以 wav2vec2 为例）：
- 训练轮数：5 epochs
- 批次大小：2（因为 GPU 显存只有 8 GB，SSL 模型很大）
- 学习率：0.001
- 最大训练音频长度：18 秒
- 最大评估音频长度：30 秒（长音频被跳过，但仅影响约 9 条/split）
- 梯度裁剪：1.0
- 优化器：AdamW

---

## 五、四个实验系统详情

### 系统 1：Log-mel 基线 (`mel_1h_hf_cuda`)
- **目的**：作为"没有预训练知识"的对照组
- **模型**：MelCTCModel（4层全连接 + CTC头）
- **参数量**：约 33 万（极小）
- **训练**：8 epochs, batch_size=8, lr=0.0003
- **结果**：WER=1.000, CER=1.000（完全失败）
- **原因分析**：仅有 1 小时数据，小模型无法学到有效的声学-文字映射。模型倾向于输出空白（空字符串），导致所有词都被视为删除错误，WER/CER 达到理论上限 100%。

### 系统 2：wav2vec2 最后一层 (`wav2vec2_1h_frozen_hf_cuda`)
- **目的**：测试冻结 SSL 编码器 + 最终层表征的效果
- **模型**：SSLCTCModel(`facebook/wav2vec2-base`, hidden_layer="last")
- **编码器参数量**：约 9500 万（冻结不训练）
- **CTC 头参数量**：约 120 万（可训练）
- **训练**：5 epochs, batch_size=2, lr=0.001
- **结果**：Dev WER=0.986, Test WER=0.985, Test CER=0.596
- **分析**：比纯基线好一点（CER 方面），但 WER 接近随机水平。wav2vec2 的最终层表征可能太"抽象"——它预训练时学到的信息更适合做对比学习，而不是直接做字符分类。

### 系统 3：wav2vec2 第6层 (`wav2vec2_1h_layer6_hf_cuda`)
- **目的**：**消融实验**——改变单一变量（hidden layer），看效果差异
- **模型**：SSLCTCModel(`facebook/wav2vec2-base`, hidden_layer="6")
- **训练**：与系统2完全相同（5 epochs, batch_size=2, lr=0.001）
- **结果**：Dev WER=0.659, Test WER=0.667, Test CER=0.262
- **关键发现**：**仅改变表征层，Test WER 从 0.985 大幅降到 0.667，CER 从 0.596 降到 0.262！** 这说明 wav2vec2 的中间层保留的语音信息比最终层更适合下游 ASR 任务。这是论文的核心发现之一。

### 系统 4：HuBERT 最后一层 (`hubert_1h_frozen_hf_cuda`)
- **目的**：测试另一种 SSL 预训练范式的效果
- **模型**：SSLCTCModel(`facebook/hubert-base-ls960`, hidden_layer="last")
- **训练**：5 epochs, batch_size=2, lr=0.001
- **结果**：Dev WER=0.656, Test WER=0.662, Test CER=0.236
- **分析**：**最佳系统**。HuBERT 的最终层比 wav2vec2 的最终层更适合下游 ASR 任务。这可能是因为 HuBERT 的预训练目标（聚类伪标签）更接近 ASR 的需求。

---

## 六、关键实验结果一览

| 系统 | Dev WER ↓ | Test WER ↓ | Test CER ↓ | RTF |
|------|-----------|------------|------------|-----|
| Log-mel 基线 | 1.000 | 1.000 | 1.000 | ~0.00004 |
| wav2vec2 最终层 | 0.986 | 0.985 | 0.596 | 0.0043 |
| **wav2vec2 第6层** | **0.659** | **0.667** | **0.262** | 0.0042 |
| **HuBERT 最终层** | **0.656** | **0.662** | **0.236** | 0.0044 |

**核心结论**：
1. ✅ 预训练 SSL 表征对低资源 ASR 确实有用（SSL 系统远超基线）
2. ✅ 表征层选择非常重要（wav2vec2 第6层远超最终层）
3. ✅ HuBERT 在冻结条件下优于 wav2vec2
4. ✅ 使用冻结编码器，仅训练小 CTC 头，在 8GB GPU 上完全可行
5. ⚠️ 冻结最终层表征 + 小 CTC 头并不能自动解决任务——WER 仍然较高（~66%），说明还需要语言模型或编码器微调

---

## 七、错误分析要点

（错误样例在 `results/error_examples.csv`）

### Log-mel 系统
- 输出几乎全为空字符串
- 说明模型学会了"输出空白"的策略——在 1 小时数据下，这是最小化 CTC loss 的局部最优解

### wav2vec2 最终层
- 严重退化，大量无意义输出（如 "bt swate se g ni int lo psa d dfs in nomas be tomn w sss"）
- 最终层表征可能丢失了太多语音细节

### HuBERT 和 wav2vec2 第6层
- 错误更"合理"：保留了语音结构，但拼写错误多
- 例如 "stephanos dedalos" → "stdeffenos det lse"（语音近似但拼写错）
- 短句/名字/生僻词尤其困难
- 这是**字符级贪婪 CTC 无语言模型**的预期表现

---

## 八、满足了哪些作业要求？

根据 `project_requirements.md` 的要求：

| 要求 | 完成情况 |
|------|---------|
| 使用语音自监督模型提取表征 | ✅ wav2vec2 和 HuBERT |
| 构建 ASR 系统（含对比实验） | ✅ 4 个系统，含 1 个非 SSL 基线 + 层消融实验 |
| 使用公开英文数据集 | ✅ LibriSpeech |
| 报告 WER/CER | ✅ 完整记录 |
| 报告 RTF | ✅ 已记录（~0.004 实时率，非常快） |
| 不同自监督模型的比较 | ✅ wav2vec2 vs HuBERT |
| 不同 hidden layer 的比较 | ✅ wav2vec2 最终层 vs 第6层 |
| 连续表征分析 | ✅ 专注于连续 hidden states |
| 错误案例分析 | ✅ 每个模型 top-10 最差预测 |
| 论文格式 | ✅ ICASSP 2026 LaTeX 模板 |

### 未覆盖的方向（论文中已说明为有意限制）：
- 离散语音单元（未使用量化器，没有 codebook/token rate/bitrate 分析）
- 编码器微调（仅做冻结实验）
- 语言模型融合
- 大范围数据规模变化（仅 1 小时点）
- BPE/子词建模

---

## 九、技术栈总结

| 层面 | 技术选型 |
|------|---------|
| 框架 | PyTorch 2.7.1 + CUDA 12.8 |
| 预训练模型 | Hugging Face Transformers |
| 音频处理 | torchaudio, soundfile |
| 评价指标 | 手写编辑距离（WER/CER） |
| 损失函数 | PyTorch 内置 CTC Loss |
| 解码 | 贪婪 CTC 解码（无 beam search） |
| 配置管理 | YAML |
| 报告 | LaTeX (ICASSP 2026 模板) |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB) |
| 数据 | LibriSpeech train-clean-100 的 ~1 小时子集 |

---

## 十、如何运行/复现

详细步骤在 `REPRODUCIBILITY.md` 中，这里概述：

1. **创建虚拟环境**：
   ```
   py -3.10 -m venv .venv_cuda
   pip install torch==2.7.1 torchaudio==2.7.1
   pip install transformers datasets soundfile jiwer pyyaml tqdm pytest
   ```

2. **准备数据**：
   ```
   python scripts/prepare_hf_librispeech_subset.py --train-hours 1 --dev-items 3000 --test-items 3000
   ```

3. **训练**（单个或批量）：
   ```
   python scripts/train_asr.py --config configs/hubert_1h_frozen_hf_cuda.yaml
   # 或批量运行全部
   python scripts/run_experiments.py --plan configs/experiment_plan.yaml
   ```

4. **汇总结果**：
   ```
   python scripts/summarize_results.py --outputs outputs --report-dir results
   ```

5. **编译论文**：
   ```
   cd report
   pdflatex -output-directory build main.tex
   ```

6. **运行测试**：
   ```
   pytest tests/test_metrics.py
   python scripts/smoke_test.py
   ```

TODO:
- 图表修复绘制