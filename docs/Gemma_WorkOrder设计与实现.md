# Gemma-WorkOrder：设计与实现

> 状态：核心闭环已实现并在 Tesla T4 上完成 Base-vs-QLoRA 实验与合并权重白盒复核。  
> 范围：受控自建数据上的工单结构化、本地只读查询与人工确认草稿，不是工业诊断或生产派单系统。

## 1. 项目定位

Gemma-WorkOrder 是一个轻量级本地大模型应用原型：维修人员输入非结构化中文故障描述，Gemma 3-1B 输出固定工单 JSON，系统识别缺失字段、执行有限的本地只读查询，并生成待人工确认的工单草稿。

项目同时保留白盒 Gemma Decoder，用于验证 QLoRA Adapter 合并后的权重是否仍能被独立实现正确加载和计算。两部分的关系是：

```text
应用主线：工单数据 → QLoRA → JSON校验 → 受控动作 → SQLite查询 → 人工确认草稿
验证支线：Adapter合并 → 自研Decoder加载 → 中间层/logits对齐
```

生产实现本身不要求复现 Decoder；白盒部分用于模型结构理解、权重映射和推理正确性验证。

## 2. 用户流程

示例输入：

```text
A区3号风机今天上午频繁停机，面板显示E07，
重启后运行十分钟再次停止，暂时没有更换零件。
```

模型目标输出：

```json
{
  "asset_name": "A区3号风机",
  "asset_id": null,
  "asset_type": "风机",
  "fault_code": "E07",
  "symptom": "重启后运行十分钟再次停止",
  "occurred_at": "今天上午",
  "actions_taken": ["重启"],
  "parts_replaced": [],
  "missing_fields": [],
  "next_action": "query_fault_code"
}
```

系统随后执行 JSON 与字段校验，将动作映射为白名单工具调用，从本地 SQLite 查询故障码参考信息，并返回 `draft_requires_human_confirmation` 草稿。

## 3. 代码结构

| 模块 | 实际职责 |
| --- | --- |
| `workorder.py` | Schema、规则基线、JSON提取、字段校验、动作路由、SQLite查询、草稿生成 |
| `workorder_data.py` | SFT Prompt、JSONL读取、规范化答案和任务指标 |
| `workorder_inference.py` | 将 Gemma 生成结果接入同一套 Schema 与工具边界 |
| `train_workorder_qlora.py` | 4bit QLoRA 训练和 Adapter 保存 |
| `evaluate_workorder.py` | Base/Adapter 的同集结构化任务评测 |
| `merge_workorder_adapter.py` | Adapter 合并为 Hugging Face checkpoint |
| `serve_workorder.py` | 本地 FastAPI 演示入口 |
| `gemma3_core.py` | 独立 Gemma Decoder、权重映射和 Hybrid KV Cache |
| `run_core_alignment.py` | Transformers 与自研 Decoder 的层输出/logits 对齐 |

## 4. 数据与输出契约

当前实验使用 90 条固定随机种子生成的受控自建数据：

| 划分 | 样本数 |
| --- | ---: |
| 训练集 | 62 |
| 验证集 | 14 |
| 测试集 | 14 |

样本覆盖三个设备类型、故障码是否存在、发生时间是否存在和若干症状表达。当前测试集主要验证两类动作：

- `query_fault_code`：存在故障码时查询本地参考表；
- `ask_clarification`：关键信息缺失时生成补充问题。

系统 Schema 也允许 `query_asset_history`，但当前 100% 动作路由指标不用于证明该路径已在多样化测试集上充分验证。

数据属于可复现的功能验证集。由于模板数量和测试集规模有限，不能用来声称真实业务泛化能力。

## 5. QLoRA 训练

训练基于成熟框架完成，不在自研 Decoder 中反向传播：

- 基础模型：`google/gemma-3-1b-it`；
- 量化：bitsandbytes 4bit NF4；
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`；
- 目标模块：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`；
- 训练：3 epoch、batch size 1、gradient accumulation 8、learning rate `2e-4`；
- 监督方式：Prompt token 的 label 设为 `-100`，只对规范化 JSON 答案计算损失。

训练耗时约 109.6 秒，最终 train loss 为 `0.0227`，validation loss 为 `0.0200`。

## 6. 结构化输出与安全边界

模型生成内容属于不可信输入，必须依次通过：

```text
模型文本
  → 提取单个JSON对象
  → 拒绝未知字段
  → 校验列表字段类型
  → 校验next_action枚举
  → 映射白名单工具
  → 执行本地只读查询或生成追问
  → 生成待人工确认草稿
```

边界包括：

- 不把模型生成的故障解释直接视为事实；
- 故障码含义来自本地参考表，并携带手册引用；
- 不提供数据库写工具；
- 不自动提交工单；
- 不生成或执行维修决策；
- Schema 校验失败时拒绝继续调用工具。

## 7. Base-vs-QLoRA 评测

相同 14 条测试样本、4bit 加载和贪心生成下：

| 指标 | Base Gemma | QLoRA Gemma |
| --- | ---: | ---: |
| JSON 合法率 | 28.57% | 100.00% |
| 字段精确率 | 60.00% | 98.57% |
| 受控动作路由准确率 | 14.29% | 100.00% |
| 完整记录精确率 | 0.00% | 85.71% |

这些结果说明 QLoRA 显著改善了当前窄 Schema 的输出稳定性和字段匹配；它们不证明开放场景推理、工业诊断或复杂 Agent 工具规划能力。

## 8. Adapter 合并与白盒验证

1. 使用 PEFT `merge_and_unload` 将 Adapter 合并进基础模型；
2. 保存标准 Hugging Face checkpoint；
3. 复用权重映射加载进独立 PyTorch Gemma Decoder；
4. 同时运行 Transformers 与自研前向链路；
5. 比较 Layer 0、5、25 和最后 token 的完整 logits。

本次运行结果：三层输出以及最后 token logits 的最大/平均绝对误差均为 `0.0`，top-1 token ID 均为 `57137`。

该结果验证的是固定环境和输入下的权重加载与计算一致性，不外推为所有输入都能位级一致。

## 9. 本地接口

`serve_workorder.py` 提供两个模式：

- `baseline`：无需下载模型，使用透明规则基线演示 Schema、工具和草稿链路；
- `gemma`：加载 Gemma 模型，将生成结果接入同一安全边界。

接口包括：

```text
GET  /api/health
POST /api/workorders/parse
```

接口是本地原型入口，不代表生产部署、鉴权、并发治理或企业系统集成已经完成。

## 10. 复现与证据

- 一键 Colab：[`notebooks/Gemma_WorkOrder_Run_All_Colab.ipynb`](../notebooks/Gemma_WorkOrder_Run_All_Colab.ipynb)
- 真实实验报告：[`reports/workorder_experiment_20260813/REPORT.md`](../reports/workorder_experiment_20260813/REPORT.md)
- 规则基线演示输出：[`reports/workorder_phase_a_demo.json`](../reports/workorder_phase_a_demo.json)
- 白盒实现说明：[`docs/Gemma3核心架构复现.md`](Gemma3核心架构复现.md)

## 11. 后续工作

若继续扩展，优先增加人工复核的多样化测试集、未知设备/故障码负例、`query_asset_history` 路由样本和端到端 API 延迟记录，而不是直接扩大功能或宣称生产可用。
