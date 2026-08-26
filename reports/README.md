# 模型评测结果

运行 `scripts/evaluate_tool_use.py` 后，Base 与 QLoRA 的完整 JSON 报告会写到这里，但默认不提交，避免仓库被逐样本原始输出撑大。

三份评测完成后运行：

```powershell
.\run_local.ps1 -Mode analyze
```

脚本会重新关联固定数据集，计算参数 Slot 指标、决策混淆矩阵和错误分类，并生成可提交的 [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)。

正式更新 README 或简历前，请至少核对：

- 两次评测使用同一模型版本、测试集、Prompt、精度和贪心解码设置；
- 报告中的 Adapter 路径正确；
- 测试集共 440 条，其中 40 条为 manifest 明示的受控缺参样本；
- 随机抽查原始输出，确认指标提升不是解析器放宽造成的；
- 记录 GPU、依赖版本、训练参数和失败案例。

建议文件名：

```text
tool_use_base_4bit.json
tool_use_qlora_4bit.json
tool_use_qlora_challenge_4bit.json
```
