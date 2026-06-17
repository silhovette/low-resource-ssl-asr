课程大作业：基于语音自监督表征的 ASR/TTS 系统设计
作业目标
本课程大作业围绕语音自监督表征学习及其在智能语音系统中的应用展开。每组需在 ASR 或 TTS 两个方向中任选一个，设计并实现一个以语音自监督表征为核心的下游任务系统，并通过客观指标完成评测与分析。
本作业重点考察以下能力：
理解并使用语音自监督模型，例如 wav2vec 2.0、HuBERT、WavLM、XLS-R 等。
探索连续语音表征或离散语音单元在 ASR/TTS 任务中的作用。
围绕具体下游任务完成方法设计、系统实现与实验验证。
使用明确、可量化的指标评价系统性能。
以正式学术论文形式呈现方法、实验结果与分析结论。
本作业不限定具体模型结构和实现路线，可以自由选择下游模型、训练策略、离散化方法、特征层、损失函数和系统组合方式，但最终系统必须体现语音自监督表征与 ASR 或 TTS 下游任务的结合。
作业内容
每组从以下两个任务方向中任选一个完成。每组人数范围1-3人。
方向一：基于语音自监督表征的低资源 ASR
任务描述
给定一段语音信号，系统需要输出对应的文本转写。该方向要求使用语音自监督模型提取语音表征，并构建 ASR 系统。系统可以使用连续 hidden states，也可以进一步构建离散语音单元。具体建模方式不限，可以采用 CTC、encoder-decoder、Transducer、Transformer、Conformer、RNN、语言模型融合或其他结构。
数据要求
推荐使用公开英文 ASR 数据集，例如 LibriSpeech。可根据方法设计自行决定数据预处理方式，包括采样率统一、文本规范化、子词建模、数据增强、特征缓存等。
方法要求
ASR 方向的系统应围绕以下问题展开：
如何有效利用预训练语音自监督表征完成低资源语音识别。
连续表征与离散语音单元对识别性能、模型复杂度和信息压缩效率有何影响。
不同自监督模型、不同表征层或不同单元设计是否会带来明显差异。
在有限标注数据条件下，系统如何取得更好的识别性能与泛化能力。
系统实现具有充分自由度，但应至少包含一个清晰的语音自监督表征使用方案，并与合理的对比系统或消融设置进行比较。
评价指标
ASR 方向以识别准确率为核心评价目标，主要指标包括：
指标	含义
WER	Word Error Rate，词错误率
CER	Character Error Rate，字符错误率
可根据系统设计进一步报告：
指标	含义
推理速度	RTF 或平均推理耗时
token rate	使用离散单元时，平均每秒 token 数
bitrate	使用离散单元时，离散表示的信息率
codebook size	使用离散单元时，离散单元词表大小
建议分析方向
报告中可围绕以下方向展开分析：
不同自监督模型的比较。
不同 hidden layer 的比较。
连续表征与离散单元的比较。
不同 codebook size 的影响。
token 去重、时长建模或 BPE 等策略的影响。
低资源条件下训练数据规模对性能的影响。
错误案例分析，例如替换、删除、插入错误。
识别性能与模型大小、训练成本、bitrate 之间的关系。
方向二：基于语音自监督表征的 TTS 系统
任务描述
给定文本输入，系统需要生成对应语音。该方向要求将语音自监督表征或离散语音单元引入 TTS 系统中。系统可以采用 text-to-unit-to-waveform、text-to-representation-to-waveform、unit-based acoustic modeling、SSL-unit vocoder 或其他形式。
数据要求
推荐使用公开英文 TTS 数据集，例如 LJSpeech。可根据方法设计自行决定文本处理、音素转换、对齐方式、声学表示、vocoder 或波形生成模块。
方法要求
TTS 方向的系统应围绕以下问题展开：
如何将语音自监督表征引入文本到语音生成流程。
连续语音表征或离散语音单元是否有助于提升合成语音的可懂度、自然度或建模效率。
不同单元设计、不同表征层或不同声学建模方式对合成质量有何影响。
如何平衡合成语音质量、模型复杂度、推理效率和表征压缩率。
系统实现具有充分自由度，可以自由选择文本前端、声学模型、unit predictor、vocoder、diffusion-based generator、GAN-based generator、flow-based model 或其他波形生成方法。
评价指标
TTS 方向应同时关注合成语音的可懂度和音质/自然度。
可懂度指标包括：
指标	含义
ASR-WER	使用 ASR 系统评估合成语音的词错误率
ASR-CER	使用 ASR 系统评估合成语音的字符错误率
音质与自然度指标包括：
指标	含义
MCD	Mel Cepstral Distortion
F0 RMSE	基频误差
UTMOS / MOS predictor	自动语音质量或自然度预测
人工 MOS	人工主观自然度评分
可根据系统设计进一步报告：
指标	含义
参数量	模型总参数量或可训练参数量
训练成本	训练时间、GPU 型号或 GPU hours
推理速度	RTF 或平均合成耗时
token rate	使用离散单元时，平均每秒 token 数
bitrate	使用离散单元时，离散表示的信息率
codebook size	使用离散单元时，离散单元词表大小
建议分析方向
报告中可围绕以下方向展开分析：
不同自监督模型或不同层表征的比较。
不同离散单元数量的影响。
text-to-unit 与 text-to-mel 等方案的比较。
合成语音可懂度与自然度之间的关系。
bitrate、音质和模型复杂度之间的关系。
合成失败案例分析，例如发音错误、韵律异常、噪声、漏词、重复词等。
vocoder 或波形生成模块对最终结果的影响。
通用要求
无论选择 ASR 还是 TTS 方向，均需满足以下要求：
系统必须使用至少一种语音自监督模型或其表征。
系统必须面向明确的 ASR 或 TTS 下游任务。
实验必须包含可量化评价指标。
实验必须包含至少一个对比实验或消融实验。
方法设计应体现一定的独立思考，可以在表征选择、离散化方式、模型结构、训练目标或系统组合方式上进行创新。
参考资料
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
提交要求
项目论文须按照 ICASSP 2026 官方 LaTeX 模板进行撰写，格式规范应与正式学术投稿保持一致。
最终提交项目论文的 PDF 文件，以及可复现实验结果的代码等支撑材料。
项目论文应重点呈现方法设计、实验设置、核心结果与分析结论。
每小组提交一份即可，并在报告中备注好小组成员姓名、学号。

