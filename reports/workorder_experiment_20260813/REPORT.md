# Gemma-WorkOrder：QLoRA 结构化工单实验报告

## 目的与范围

在既有 Gemma 3 1B 白盒 Decoder、官方权重对齐与推理实验的基础上，验证小模型经 QLoRA 后能否稳定完成一个**受约束的工单结构化任务**：将中文设备故障描述转换为固定 JSON Schema，选择有限的本地只读查询工具，并只生成需人工确认的工单草稿。

这不是工业故障诊断、自动派单或维修决策系统。故障码含义来自本地 SQLite 参考表；模型输出需经过 JSON、字段和工具白名单校验。

## 数据与设置

- 基础模型：`google/gemma-3-1b-it`
- 硬件：Google Colab Tesla T4（15 GB 显存）
- 数据：90 条固定随机种子（42）的受控自建样本；训练/验证/测试为 62 / 14 / 14
- 微调：4bit NF4 QLoRA，`r=16`、`alpha=32`、`dropout=0.05`，目标层为 `q/k/v/o` 及 FFN 投影层
- 训练：3 epoch，batch size 1，gradient accumulation 8，learning rate `2e-4`
- 解码：贪心生成（`do_sample=False`），最大生成 256 token

训练耗时约 109.6 秒；最终 train loss 为 0.0227，validation loss 为 0.0200。损失仅用于训练过程监控，以下任务指标才用于 Base-vs-QLoRA 比较。

## 结果

| 指标 | Base Gemma（4bit） | QLoRA Gemma（4bit） |
| --- | ---: | ---: |
| 测试样本数 | 14 | 14 |
| JSON 合法率 | 28.57% | **100.00%** |
| 字段精确率 | 60.00% | **98.57%** |
| 受控动作路由准确率 | 14.29% | **100.00%** |
| 完整记录精确率 | 0.00% | **85.71%** |

指标定义：

- JSON 合法率：输出可解析为 JSON 且通过字段/工具白名单校验的比例；
- 字段精确率：固定 Schema 的字段级精确匹配；
- 受控动作路由准确率：当前测试集中的 `query_fault_code` 或 `ask_clarification` 是否与受控标签一致；系统虽支持 `query_asset_history` 路径，但本指标不用于证明该路径已得到充分验证；
- 完整记录精确率：同一条样本的全部字段均精确匹配的比例。

## 白盒复核

将 QLoRA Adapter 合并为 Hugging Face checkpoint 后，再载入本项目独立实现的 Gemma Decoder，与 Transformers 官方实现比较层输出和最终 logits：

- Layer 0（Sliding Attention）、Layer 5（Global Attention）、Layer 25（Sliding Attention）的 max/mean absolute error 均为 `0.0`；
- 三层 cosine similarity 分别为 `1.000000119`、`0.999999940`、`1.000000000`；
- 最后一 token logits 的 max/mean absolute error 均为 `0.0`；
- 官方实现与自研 Decoder 的 top-1 token ID 均为 `57137`。

这项复核只证明：本次合并后的 checkpoint 可被自研 Decoder 正确加载、映射并得到一致计算结果；它不等价于该模型在开放场景的泛化能力评估。

## 结果边界

- 数据由受控模板生成，测试集只有 14 条；
- 路由指标主要覆盖故障码查询和缺失信息澄清，不是复杂多工具 Agent 评测；
- 指标只说明当前窄 Schema 上的任务适配效果；
- 未使用企业生产数据，也未验证真实工业诊断能力。

## 可复现入口

完整命令见 [`notebooks/Gemma_WorkOrder_Run_All_Colab.ipynb`](../../notebooks/Gemma_WorkOrder_Run_All_Colab.ipynb)。在 Colab 中配置有模型访问权的 `HF_TOKEN` 并选择 T4 GPU 后运行即可复现流程。
