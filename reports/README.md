# 实验结果目录

运行脚本后会在这里产生 CSV、JSON 或 JSONL。临时结果默认被 `.gitignore` 忽略，防止不同硬件的实验相互覆盖。

当前公开证据分为三组：

- [`workorder_experiment_20260813/`](workorder_experiment_20260813/)：QLoRA 工单结构化、Base-vs-QLoRA 指标和合并权重白盒复核；
- [`core_alignment_20260812/`](core_alignment_20260812/)：官方 Gemma 与独立 Decoder 的原始数值对齐记录；
- [`t4_20260811/`](t4_20260811/)：KV Cache、FP16/4bit、吞吐和显存实验的原始数据、图表与报告。

新增公开结果时，只提交经过复核的报告，并至少包含：

- Colab GPU 型号和运行时间；
- PyTorch、Transformers、bitsandbytes 版本；
- 模型 ID、精度、提示词 token 数和生成 token 数；
- 预热次数、重复次数、随机种子；
- TTFT、decode TPS、端到端 TPS、峰值显存；
- 固定题集通过率与失败样例；
- 结论的适用范围和限制。
