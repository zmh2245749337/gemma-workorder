# Gemma 3 轻量大模型推理优化与质量评测

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zmh2245749337/gemma-inference-eval/blob/main/notebooks/Gemma_Inference_Eval_Colab.ipynb)

这是一个面向学习和面试复盘的可复现项目，使用 `google/gemma-3-1b-it` 研究四个问题：

1. 大语言模型如何从 logits 逐 token 生成文本；
2. KV Cache 是否能在相同输出下减少重复计算；
3. 16 位与 4 位权重量化在 Colab GPU 上如何影响显存、首 token 延迟和吞吐量；
4. 性能优化后，模型在固定任务集上的基本质量是否发生变化。

项目不沿用旧说明中的历史性能数字。所有可写进简历的数据都必须由当前代码重新运行，并保留硬件、软件版本、参数、原始输出和时间戳。

## 为什么这样重做

旧项目的架构分析和 Top-K 思路可以保留，但缺少源代码、实验日志和统一基准，而且历史材料中存在互相冲突的吞吐量。这个版本把项目收敛为一条可以验证的主线：

> 手写生成循环 → 验证与 Hugging Face 贪心解码一致 → 比较有无 KV Cache → 比较不同精度 → 在固定题集上检查质量 → 保存完整实验记录

## 目录

```text
configs/                 固定中文任务集
docs/                    零基础原理、实验设计与面试复盘
notebooks/               Colab 运行入口
scripts/                 可直接运行的实验入口
src/gemma_eval/          模型加载、手写解码、基准与评测代码
tests/                   不下载模型即可运行的单元测试
reports/                 实验结果说明；真实运行结果默认不提交
requirements-colab.txt   Colab 依赖
```

## Colab 准备

Gemma 在 Hugging Face 上需要先同意模型许可。打开模型页面、登录账户并接受条款，然后在 Colab Secrets 中添加 `HF_TOKEN`。不要把 Token 写入代码、截图或 GitHub。

```bash
pip install -r requirements-colab.txt
```

仓库发布后，可以点击 README 顶部的徽章在 Colab 中打开 `notebooks/Gemma_Inference_Eval_Colab.ipynb` 并逐格运行。

确认 GPU：

```bash
nvidia-smi
```

## 1. 验证手写解码

下面的命令使用贪心解码，比较手写生成循环与 `model.generate` 的 token 是否完全一致：

```bash
python scripts/run_manual_decode.py \
  --prompt "请用三句话解释什么是KV Cache。" \
  --max-new-tokens 48 \
  --greedy \
  --check-hf-parity
```

再运行采样版本：

```bash
python scripts/run_manual_decode.py \
  --prompt "写一个关于机器人学习的短故事。" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --seed 42
```

## 2. 比较 KV Cache

同一次命令会在相同提示词、输出长度和精度下分别测试启用与关闭 KV Cache：

```bash
python scripts/run_benchmark.py \
  --precision fp16 \
  --max-new-tokens 32 \
  --warmups 1 \
  --repeats 3 \
  --output reports/benchmark.csv
```

主要指标：

- `ttft_ms`：完成 prefill 并选出第一个新 token 的时间；
- `decode_tps`：第一个 token 之后的生成吞吐量；
- `total_tps`：包含 prefill 的端到端吞吐量；
- `peak_memory_mb`：本次生成期间 PyTorch 记录的峰值 GPU 显存；
- `model_footprint_mb`：模型加载后的内存占用估计。

## 3. 比较 16 位与 4 位量化

分别运行两次。每次只加载一种精度，避免同时驻留两个模型造成显存干扰：

```bash
python scripts/run_benchmark.py --precision fp16 --output reports/benchmark.csv
python scripts/run_benchmark.py --precision 4bit --output reports/benchmark.csv
```

量化不保证一定更快。4 位通常会降低显存，但在小模型、短输出或特定 GPU 上，反量化开销可能抵消速度收益。因此结论必须来自当前硬件实测。

## 4. 固定题集质量检查

```bash
python scripts/run_quality_eval.py \
  --precision fp16 \
  --dataset configs/prompts_zh.jsonl \
  --output reports/quality_fp16.jsonl

python scripts/run_quality_eval.py \
  --precision 4bit \
  --dataset configs/prompts_zh.jsonl \
  --output reports/quality_4bit.jsonl
```

当前题集只检查可客观判断的关键词、抽取和格式遵循任务，不把它包装成通用大模型能力评测。开放式文本质量仍需人工盲评或引入正式评测集。

## 简历数据的使用边界

完成实验前只能写“实现了什么”，不能写“提升了多少”。完成后还要满足：

- 至少 1 次预热、3 次正式重复；
- 保存 GPU、PyTorch、Transformers、量化方式和随机种子；
- 对比实验使用相同提示词、生成长度和停止条件；
- 同时报告性能和任务通过率，不只挑最好看的指标；
- 明确结果只适用于测试硬件与当前模型版本。

## 当前阶段

- [x] 手写 greedy、Top-K、Top-P 采样逻辑
- [x] KV Cache / 无 Cache 统一基准入口
- [x] FP16/BF16/8-bit/4-bit 模型加载入口
- [x] 固定中文题集与可追溯结果格式
- [ ] 在 Colab T4 上生成第一份真实实验记录
- [ ] 根据真实结果绘制图表并撰写结论
- [ ] 扩充到 40–50 条评测样本并进行人工抽检
