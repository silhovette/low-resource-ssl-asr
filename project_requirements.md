# 方向一：基于语音自监督表征的低资源 ASR

## 任务描述

给定一段语音信号，系统需要输出对应的文本转写。该方向要求使用语音自监督模型提取语音表征，并构建 ASR 系统。系统可以使用连续 hidden states，也可以进一步构建离散语音单元。具体建模方式不限，可以采用 CTC、encoder-decoder、Transducer、Transformer、Conformer、RNN、语言模型融合或其他结构。

## 数据要求
推荐使用公开英文 ASR 数据集，例如 LibriSpeech。可根据方法设计自行决定数据预处理方式，包括采样率统一、文本规范化、子词建模、数据增强、特征缓存等。

## 方法要求

ASR 方向的系统应围绕以下问题展开：
- 如何有效利用预训练语音自监督表征完成低资源语音识别。
- 连续表征与离散语音单元对识别性能、模型复杂度和信息压缩效率有何影响。
- 不同自监督模型、不同表征层或不同单元设计是否会带来明显差异。
- 在有限标注数据条件下，系统如何取得更好的识别性能与泛化能力。
- 系统实现具有充分自由度，但应至少包含一个清晰的语音自监督表征使用方案，并与合理的对比系统或消融设置进行比较。
## 评价指标

ASR 方向以识别准确率为核心评价目标，主要指标包括：

| 指标 | 含义 |
| :--- | :--- |
| **WER** | Word Error Rate，词错误率 |
| **CER** | Character Error Rate，字符错误率 |

可根据系统设计进一步报告：

| 指标 | 含义 |
| :--- | :--- |
| **推理速度** | RTF 或平均推理耗时 |
| **token rate** | 使用离散单元时，平均每秒 token 数 |
| **bitrate** | 使用离散单元时，离散表示的信息率 |
| **codebook size** | 使用离散单元时，离散单元词表大小 |

## 建议分析方向

报告中可围绕以下方向展开分析：

* 不同自监督模型的比较。
* 不同 hidden layer 的比较。
* 连续表征与离散单元的比较。
* 不同 codebook size 的影响。
* token 去重、时长建模或 BPE 等策略的影响。
* 低资源条件下训练数据规模对性能的影响。
* 错误案例分析，例如替换、删除、插入错误。
* 识别性能与模型大小、训练成本、bitrate 之间的关系。

## 通用要求

无论选择 ASR 还是 TTS 方向，均需满足以下要求：
- 系统必须使用至少一种语音自监督模型或其表征。
- 系统必须面向明确的 ASR 或 TTS 下游任务。
- 实验必须包含可量化评价指标。
- 实验必须包含至少一个对比实验或消融实验。
- 方法设计应体现一定的独立思考，可以在表征选择、离散化方式、模型结构、训练目标或系统组合方式上进行创新。

## 参考资料

以下资料仅作为数据集、baseline、工具包和实现流程的参考入口，不限定最终实现方案。
1. https://www.openslr.org/12/
2. https://keithito.com/LJ-Speech-Dataset/
3. https://github.com/s3prl/s3prl
4. https://github.com/espnet/espnet
5. https://github.com/speechbrain/speechbrain
6. https://huggingface.co/docs/transformers/en/model_doc/wav2vec2
7. https://huggingface.co/docs/transformers/en/model_doc/hubert
8. https://huggingface.co/docs/transformers/en/model_doc/wavlm
9. https://github.com/sarulab-speech/UTMOSv2
10. https://cmsworkshops.com/ICASSP2026/papers/paper_kit.php

## 提交要求

项目论文须按照 ICASSP 2026 官方 LaTeX 模板进行撰写，格式规范应与正式学术投稿保持一致。

最终提交项目论文的 PDF 文件，以及可复现实验结果的代码等支撑材料。

项目论文应重点呈现方法设计、实验设置、核心结果与分析结论。

每小组提交一份即可，并在报告中备注好小组成员姓名、学号。