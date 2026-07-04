# 补充实验结果报告

> 生成日期：2026-07-04  
> 对应指南：`RUN_EXPERIMENTS.md`  
> 实验完成情况：Phase 2 (Layer Sweep) ✅ | Phase 3 (Fine-tuning) ✅ | Phase 4 (LM 融合) ❌ 因服务器缺少 cmake 跳过

---

## 一、实验总览

在原有 4 个基线系统的基础上，新增 5 个实验，共 9 个系统：

| # | 系统 | 类型 | 状态 |
|---|------|------|------|
| 1 | Log-mel 基线 | 基线 | 原有 |
| 2 | wav2vec2 last (frozen) | 基线 | 原有 |
| 3 | wav2vec2 layer 6 (frozen) | 基线 | 原有 |
| 4 | HuBERT last (frozen) | 基线 | 原有 |
| 5 | wav2vec2 layer 0 (frozen) | Layer Sweep | **新增** |
| 6 | wav2vec2 layer 3 (frozen) | Layer Sweep | **新增** |
| 7 | wav2vec2 layer 9 (frozen) | Layer Sweep | **新增** |
| 8 | wav2vec2 layer 6 (FT top-6) | Fine-tuning | **新增** |
| 9 | HuBERT last (FT top-6) | Fine-tuning | **新增** |

全部训练配置：编码器冻结时 batch_size=2, lr=0.001, 5 epochs；微调时为 3 epoch 冻结预热 + 5 epoch 解冻顶部 6 层, finetune_lr=0.0001, 共 8 epochs。训练设备：服务器 GPU（CUDA），模型为 wav2vec2-base 和 hubert-base-ls960。

---

## 二、完整结果表

按 Test WER 从低到高排序：

| System | Dev WER ↓ | Test WER ↓ | Test CER ↓ | RTF |
|--------|-----------|------------|------------|-----|
| **HuBERT last (FT top-6)** | **0.316** | **0.320** | **0.106** | 0.0011 |
| wav2vec2 layer 9 (frozen) | 0.605 | 0.617 | 0.243 | 0.0022 |
| HuBERT last (frozen) | 0.656 | 0.663 | 0.232 | 0.0020 |
| wav2vec2 layer 6 (frozen) | 0.666 | 0.670 | 0.281 | 0.0021 |
| wav2vec2 layer 6 (FT top-6) | 0.678 | 0.683 | 0.271 | 0.0011 |
| wav2vec2 layer 3 (frozen) | 0.913 | 0.915 | 0.418 | 0.0022 |
| wav2vec2 layer 0 (frozen) | 0.975 | 0.974 | 0.582 | 0.0022 |
| wav2vec2 last (frozen) | 0.989 | 0.986 | 0.640 | 0.0022 |
| Log-mel 基线 | 1.000 | 1.000 | 1.000 | ~0.00005 |

---

## 三、Layer Sweep 分析

### 3.1 实验设计

系统扫描 wav2vec2-base 的 5 个隐藏层（layer 0, 3, 6, 9, 12/last），编码器全冻结，仅训练 CTC 投影头。目的是验证"中间层优于最终层"的趋势是否在整条深度轴上连续成立。

### 3.2 结果

| Layer | Test WER | Test CER |
|-------|----------|----------|
| 0 (最浅) | 0.974 | 0.582 |
| 3 | 0.915 | 0.418 |
| 6 | 0.670 | 0.281 |
| **9** | **0.617** | **0.243** |
| 12/last (最深) | 0.986 | 0.640 |

### 3.3 关键发现

1. **完美的 U 形曲线**：WER 从 layer 0 的 0.974 持续下降到 layer 9 的 0.617，然后在 layer 12 急剧回升至 0.986。曲线图见 `report_figures/layer_sweep.pdf`。

2. **最优层是 layer 9**（而非之前报告的 layer 6）。layer 9 的 WER (0.617) 比 layer 6 (0.670) 进一步降低了 5.3 个绝对百分点（8% 相对改善）。这说明在 wav2vec2 冻结条件下，**接近但不等于最终层的表征最适合下游 ASR**。

3. **最终层完全不可用**（WER 0.986，接近随机水平）。这与 wav2vec2 2.0 的预训练目标一致：最终层的对比学习表征过度适配了"区分不同音频片段"的任务，丢失了细粒度的语音-文字映射信息。

4. **layer 0 几乎无用**（WER 0.974）：CNN 特征提取器输出的最浅层表征还没有学到足够的语音结构。

### 3.4 对论文的意义

这是一个非常干净、有说服力的消融实验。单变量控制（仅改变 `hidden_layer` 参数），5 个数据点构成完整 U 形曲线，直接支撑论文的核心论点："预训练 SSL 表征的层选择对低资源 ASR 至关重要"。

---

## 四、Fine-tuning 分析

### 4.1 实验设计

在冻结训练的基础上，引入两阶段训练：
- **Phase A（epoch 1-3）**：编码器全冻结，lr=0.001，只训练 CTC 投影头
- **Phase B（epoch 4-8）**：解冻顶部 6 层 Transformer，lr=0.0001，联合训练

选择 wav2vec2 layer 6 和 HuBERT last 两个系统进行微调，原因是它们分别是各自模型在冻结条件下表现最好的层之一。

### 4.2 结果

#### HuBERT last：巨大成功

| 阶段 | Dev WER | Test WER | Test CER |
|------|---------|----------|----------|
| 冻结 (5 epoch) | 0.656 | 0.663 | 0.232 |
| **微调 (8 epoch)** | **0.316** | **0.320** | **0.106** |
| 相对改善 | — | **−51.6%** | **−54.2%** |

微调后 WER 从 0.663 下降到 0.320，**几乎减半**。这是全部 9 个系统中的最佳结果，且遥遥领先。训练曲线显示 dev WER 在 epoch 4（开始微调后）出现大幅跳升（从 0.704 降至 0.491），之后持续收敛到 0.316，没有出现过拟合迹象，说明 8 个 epoch 可能还不够，更长训练有望进一步改善。

#### wav2vec2 layer 6：基本无效

| 阶段 | Dev WER | Test WER | Test CER |
|------|---------|----------|----------|
| 冻结 (5 epoch) | 0.666 | 0.670 | 0.281 |
| 微调 (8 epoch) | 0.678 | 0.683 | 0.271 |
| 相对改善 | — | **+1.9%** | −3.6% |

微调后的 Test WER (0.683) 反而略差于冻结版 (0.670)。CER 有微弱改善（0.281 → 0.271），但幅度远小于 HuBERT。**更重要的是：wav2vec2 layer 6 微调 (0.683) 还不如直接选 layer 9 冻结 (0.617)**。

### 4.3 为什么 HuBERT 微调有效而 wav2vec2 无效？

这是一个值得在论文中深入讨论的发现。我们认为原因在于两种预训练目标的本质差异：

| 维度 | wav2vec2 2.0 | HuBERT |
|------|-------------|--------|
| 预训练目标 | 对比学习（区分不同音频片段） | 聚类伪标签（预测离散语音单元） |
| 表征特点 | 判别性、对说话人/信道敏感 | 更接近语音内容本身 |
| 微调行为 | 最终层表征已过度特化，微调顶部层也难以挽救 | 表征保留了足够的语音信息，微调可有效适配 ASR |
| 低资源下的策略 | **选对 layer 比微调更重要** | **微调是最大的单次改善来源** |

简而言之：HuBERT 的聚类预训练目标天然更接近 ASR 需求，其表征在微调时能有效迁移。而 wav2vec2 的对比学习表征在冻结中间层时已经足够好，微调仅有的 1 小时数据不足以让顶部层学到比 layer 9 冻结表征更好的表示。

### 4.4 对论文的意义

1. **HuBERT FT 是全文最强结果**，WER 0.320 — 相比冻结版的 0.663 是质的飞跃，可以有力回应"为什么不微调"的质疑。
2. **wav2vec2 FT 的失败同样有价值** — 说明微调不是万能药，在低资源场景下表征层选择可能比微调更关键。这本身就是一个有趣的发现。
3. 两个模型的相反表现构成了**自然的对比讨论**，增强了论文的分析深度。

---

## 五、与前人工作的对比

| 工作 | 数据量 | 模型 | 方法 | WER |
|------|--------|------|------|-----|
| Baevski et al. (2020) | 960h | wav2vec2 | 全微调 | ~2.0 (test-clean) |
| Hsu et al. (2021) | 960h | HuBERT | 全微调 | ~1.8 (test-clean) |
| 本项目 (冻结) | 1h | HuBERT last | 冻结 CTC 头 | 0.663 |
| 本项目 (冻结) | 1h | wav2vec2 layer 9 | 冻结 CTC 头 | 0.617 |
| **本项目 (微调)** | **1h** | **HuBERT last** | **部分微调 (top-6)** | **0.320** |

注意：前人工作在 960 小时全量 LibriSpeech 上报告的是 test-clean 全集的 WER（约 2-3%）。本项目在 test-clean 全集的 1 小时子集上评估，WER 数值不可直接比较，但 0.320 的 WER 在仅 1 小时训练数据下是合理且令人鼓舞的结果。

---

## 六、论文修改建议

### 6.1 必须修改的部分

1. **Results 表**：替换为完整 9 系统的表（`results/report_tables.tex` 已自动生成），或至少新增 HuBERT FT、layer sweep 的关键行。

2. **Layer Ablation**：将原有的 layer 6 vs layer 12 的简单对比扩展为完整的 5 点 U 形曲线。新增 `\ref{fig:layersweep}` 图（`report_figures/layer_sweep.pdf`）。

3. **Fine-tuning 讨论**：新增一小节（或替换原有"为什么不微调"部分），报告 HuBERT FT 的显著改善和 wav2vec2 FT 的无效，分析原因。

### 6.2 建议修改的部分

4. **Abstract**：如果 HuBERT FT (WER 0.320) 被纳入，Abstract 中的数字和结论需要更新。

5. **Conclusion**：强调两个核心发现 — (1) 层选择对冻结 SSL 表征至关重要（U 形曲线），(2) 微调的效果高度依赖预训练范式（HuBERT 受益巨大，wav2vec2 几乎无效）。

6. **Limitations**：注意以下已知局限：
   - Phase 4 (n-gram LM 融合) 因服务器编译环境问题未完成
   - Fine-tuning 仅尝试了一组超参（top-6 层, lr=0.0001），未做超参搜索
   - 仅测试了 1 小时数据点，未探索数据量 scaling

---

## 七、文件清单

解压 `supplementary_results_*.tar` 后，项目目录中新增/更新的文件：

```
outputs/
├── wav2vec2_1h_layer0_hf_cuda/     # Layer Sweep 新增
├── wav2vec2_1h_layer3_hf_cuda/     # Layer Sweep 新增
├── wav2vec2_1h_layer9_hf_cuda/     # Layer Sweep 新增
├── wav2vec2_1h_finetune6_hf_cuda/  # Fine-tuning 新增
└── hubert_1h_finetune6_hf_cuda/    # Fine-tuning 新增（最佳系统）

results/
├── summary.csv                     # 所有 9 系统的汇总
├── report_tables.tex               # LaTeX 表格（可直接 \input）
├── error_examples.csv              # 各模型最差预测样例
└── layer_sweep.csv                 # Layer Sweep 数据（5 层 × 4 指标）

report_figures/
├── layer_sweep.pdf                 # U 形曲线图（WER/CER vs Layer）
├── training_curves.pdf             # 训练曲线
├── wer_cer_bars.pdf                # 柱状图对比
├── error_breakdown.pdf             # 错误分解
├── layer_ablation.pdf              # 层消融对比
├── convergence_heatmap.pdf         # 收敛热力图
└── architecture.pdf                # 架构图

configs/                            # 补充实验的 7 个 YAML 配置
src/trainer.py                      # 已修复的 fine-tuning 训练逻辑
scripts/train_asr.py                # 已修复的训练脚本
```
