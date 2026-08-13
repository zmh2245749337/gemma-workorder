# Gemma-WorkOrder：设备故障工单结构化与本地工具路由升级路线

> 状态：规划中，本文描述下一阶段实现方案，不代表相关功能已经完成。  
> 原则：保留现有白盒 Decoder、官方权重对齐、手写生成、KV Cache 与量化实验；新增能力必须有代码、运行记录和可复现实验后才能写入简历。

## 1. 项目最终定位

将当前“Gemma 3 核心架构复现与高效推理”扩展为：

> 面向设备运维记录的轻量级本地推理原型：把维修人员输入的非结构化中文故障描述转换为标准工单，识别缺失字段，并选择本地资产、故障码或维修手册查询工具；专业事实来自本地可信数据，模型不独立进行高风险故障诊断。

最终名称暂定：

**Gemma-WorkOrder：基于 Gemma 3-1B 的设备故障工单结构化与本地工具路由系统**

## 2. 为什么和现有项目能够串联

```text
现有：Decoder白盒复现 → 官方权重数值对齐 → 手写解码
      → Hybrid KV Cache → FP16/4-bit实验

新增：工单Schema与数据管线 → QLoRA任务适配 → Adapter合并
      → 自研Decoder加载合并权重 → JSON校验与工具路由
      → 本地SQLite查询 → FastAPI工单接口
```

现有模块在新场景中的职责：

| 现有能力 | 场景作用 |
|---|---|
| 自研 Gemma Decoder | 提供可观察、可逐层验证的 FP16 推理后端 |
| 官方权重对齐 | 建立 Transformers 与自研实现的正确性基线 |
| 手写生成 | 控制 greedy/采样和结构化输出实验 |
| Hybrid KV Cache | 支持多轮信息补充并研究缓存开销 |
| 4-bit 实验 | 研究有限显存下的本地部署取舍 |
| 现有报告体系 | 保存环境、参数、原始输出和可复现实验证据 |

业务原型本身不要求从零复现 Decoder。复现部分的价值是结构理解、权重验证和推理实验；生产部署仍应优先考虑 Transformers、vLLM 等成熟实现。

## 3. 真实功能边界

### 3.1 第一版必须支持

1. **工单字段抽取**：设备编号/名称、故障码、现象、时间、已采取操作、更换部件；
2. **缺失信息识别**：指出设备型号、时间、运行参数等缺失字段；
3. **本地工具路由**：在有限工具集合中选择正确工具并生成参数；
4. **工单草稿生成**：结合模型抽取结果和数据库查询结果生成待确认草稿；
5. **结构化校验**：JSON解析、字段类型、工具白名单和必填参数校验。

### 3.2 第一版明确不做

- 不宣称独立完成工业故障诊断；
- 不自动决定维修方案或执行维修操作；
- 不替代专业运维人员；
- 不声称使用真实企业生产数据；
- 不声称已经在企业内网生产部署；
- 不把自研 eager PyTorch Decoder 宣称为高性能推理引擎。

## 4. 用户流程

示例输入：

```text
A区3号风机今天上午频繁停机，面板显示E07，
重启后运行十分钟再次停止，暂时没有更换零件。
```

模型首先输出：

```json
{
  "asset_name": "A区3号风机",
  "asset_id": null,
  "fault_code": "E07",
  "symptom": "重启后运行十分钟再次停止",
  "occurred_at": "今天上午",
  "actions_taken": ["重启"],
  "parts_replaced": [],
  "missing_fields": ["设备编号", "设备型号", "运行电流"],
  "next_action": "query_fault_code"
}
```

随后生成工具调用：

```json
{
  "name": "query_fault_code",
  "arguments": {
    "code": "E07",
    "asset_type": "风机"
  }
}
```

系统从本地可信数据查询故障码解释，再生成待人工确认的标准工单。若必要参数缺失，模型只生成追问，不调用工具。

## 5. 本地工具与可信数据

第一版工具保持少而明确：

| 工具 | 输入 | 数据来源 | 输出 |
|---|---|---|---|
| `query_asset` | asset_id/name | SQLite资产表 | 设备类型、型号、位置 |
| `query_asset_history` | asset_id | SQLite维修记录 | 最近故障与维护摘要 |
| `query_fault_code` | code/type | JSON或SQLite故障码表 | 故障码含义与手册来源 |
| `query_manual` | model/keyword | 本地手册条目 | 带来源的相关条目 |
| `draft_work_order` | structured fields | 纯本地模板 | 待确认工单草稿 |

模型只负责理解、抽取和路由；设备事实和专业说明必须来自本地表或手册条目。

## 6. 数据 Schema

推荐训练样本：

```json
{
  "sample_id": "wo_0001",
  "input_text": "3号风机上午出现E07，重启十分钟后再次停机",
  "output": {
    "asset_name": "3号风机",
    "asset_id": null,
    "fault_code": "E07",
    "symptom": "重启十分钟后再次停机",
    "occurred_at": "上午",
    "actions_taken": ["重启"],
    "parts_replaced": [],
    "missing_fields": ["设备编号", "设备型号"],
    "next_action": "query_fault_code"
  }
}
```

数据至少覆盖：

- 字段完整与字段缺失；
- 口语、省略、错别字和无关信息；
- 同一事实的多种中文表达；
- 多故障码、无故障码和未知故障码；
- 不需要调用工具的普通记录；
- 工具参数不足、应当追问的记录；
- 容易混淆的工具选择负例。

建议先构造 500～1000 条可控样本完成可行性验证，再根据错误类型扩充。训练集可模板生成和模型辅助改写，测试集必须人工复核，并在文档中明确属于自建数据。

## 7. QLoRA 训练阶段

训练使用成熟框架，不在自研 Decoder 中反向传播：

- Transformers；
- PEFT；
- TRL/SFTTrainer；
- bitsandbytes 4-bit NF4；
- LoRA Adapter；
- Colab Tesla T4。

训练产物：

```text
artifacts/workorder_lora/
├─ adapter_config.json
├─ adapter_model.safetensors
├─ training_args.json
└─ metrics.json
```

必须保存：随机种子、数据版本、模型版本、训练参数、运行环境、Loss曲线和失败样例。

## 8. Adapter 合并与白盒推理验证

1. 使用 PEFT `merge_and_unload` 将 Adapter 合并到基础模型；
2. 保存标准 Hugging Face 合并权重；
3. 复用当前权重映射，将合并权重加载进自研 Gemma Decoder；
4. 在固定工单输入上比较 Transformers 与自研 Decoder；
5. 记录中间层、logits、top-1 与生成 token；
6. 若不一致，定位权重命名、dtype、chat template、位置编码或生成参数差异。

注意：现有自研 Decoder 首先验证 FP16 合并权重。4-bit 部署可以继续使用成熟 bitsandbytes 路径；除非真实实现自研量化权重计算，否则不宣称自研 Decoder 支持4-bit内核。

## 9. 结构化输出与安全校验

推理输出必须经过：

```text
模型文本 → JSON提取 → Pydantic/JSON Schema校验
        → 工具名白名单 → 参数完整性校验
        → 可执行工具调用或生成追问
```

校验失败时最多进行一次格式修复；仍失败则返回人工处理，不循环调用模型。

`draft_work_order` 只生成草稿。任何写入外部系统的动作都需要人工确认，第一版不实现自动提交。

## 10. API 与演示界面

建议接口：

```text
POST /api/workorders/parse       结构化故障描述
POST /api/tools/route            生成并校验工具调用
POST /api/tools/execute          执行本地只读查询
POST /api/workorders/draft       生成待确认工单
GET  /api/health                 模型和数据库状态
```

最小演示页面显示：原始描述、结构化字段、缺失信息、工具调用、本地查询来源、工单草稿和推理耗时。

## 11. 评估设计

### 11.1 任务效果

- JSON有效率；
- 字段级 Precision/Recall/F1；
- 工具选择准确率；
- 工具参数完全匹配率；
- 缺失字段识别F1；
- 无需工具场景识别率。

### 11.2 模型对比

- Base Gemma 与 QLoRA Gemma；
- Transformers 与自研 Decoder；
- FP16 与4-bit成熟加载路径；
- greedy 与有限采样策略。

### 11.3 系统指标

- 峰值显存；
- TTFT；
- decode tokens/s；
- 单条工单端到端延迟；
- 工具查询与模型推理耗时拆分。

所有数字必须由脚本生成并保留原始记录。自建小型测试集结果不得表述为通用模型能力。

## 12. 分阶段实施与停止点

### Phase A：场景骨架，不训练（后续第一步）

- 定义Schema、工具协议和SQLite样例数据；
- 构造50条人工复核可行性集；
- 用Base Gemma跑通结构化输出；
- 实现JSON校验和错误分析。

**验收后再进入Phase B。** 若1B模型连窄Schema都无法稳定学习，则收缩字段，不继续扩大场景。

### Phase B：QLoRA任务适配

- 构造500～1000条训练数据；
- Colab T4完成QLoRA；
- 对比Base与微调模型；
- 保存Adapter、训练配置和报告。

### Phase C：合并权重与自研Decoder验证

- 合并Adapter；
- 加载到自研Decoder；
- 完成固定样例数值与生成对齐；
- 复用现有KV Cache和性能脚本。

### Phase D：本地应用原型

- FastAPI接口；
- SQLite工具查询；
- 工单草稿页面；
- FP16/4-bit部署对比；
- README截图、Demo和完整报告。

每个Phase完成后单独提交，保证中途停止时仓库仍然完整、可解释。

## 13. 简历关键词解锁条件

| 关键词 | 写入简历前必须满足 |
|---|---|
| 工单结构化 | Schema、解析代码与真实Demo已存在 |
| QLoRA/PEFT | 实际训练完成，Adapter和参数已保存 |
| 工具路由 | 至少三个真实工具可选择并通过Schema校验 |
| 自研Decoder加载微调权重 | 合并权重已成功加载并有对齐报告 |
| FastAPI | 接口能启动并完成端到端请求 |
| 本地部署 | 在本地或Colab完成实际运行，不等同生产部署 |
| 4-bit | 存在真实显存/速度/任务效果对比 |

## 14. 预期简历主线（全部完成后）

> 基于PyTorch独立实现Gemma 3-1B文本Decoder并完成官方权重数值对齐；面向设备运维记录构建自建中文工单数据，使用QLoRA适配字段抽取、缺失信息识别和本地工具路由；将合并权重加载回自研Decoder，结合KV Cache、4-bit成熟推理路径与FastAPI完成本地工单原型，并从结构化任务效果、显存和吞吐三个层面验证取舍。

该段只是目标口径。未完成对应Phase前不得直接复制到简历。

