# 实验结果目录

运行脚本后会在这里产生 CSV、JSON 或 JSONL。结果文件默认被 `.gitignore` 忽略，防止不同硬件的临时实验相互覆盖。

准备公开结果时，建议只提交一份经过复核的报告，并至少包含：

- Colab GPU 型号和运行时间；
- PyTorch、Transformers、bitsandbytes 版本；
- 模型 ID、精度、提示词 token 数和生成 token 数；
- 预热次数、重复次数、随机种子；
- TTFT、decode TPS、端到端 TPS、峰值显存；
- 固定题集通过率与失败样例；
- 结论的适用范围和限制。
