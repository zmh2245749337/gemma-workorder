# Gemma Tool-Use Post-Training

> 面向任务型 Agent 的 Gemma 3-1B 多轮状态跟踪、工具决策后训练与安全执行实验。

这个项目只解决一个明确问题：**让 Gemma 3-1B 在多轮中文对话中维护结构化状态，判断应该调用工具、追问缺失信息还是不调用工具，并在代码侧安全执行。**

它不是另一个大而全的 Agent 应用。规划、RAG、MCP、多 Agent 协作由主项目承担；本项目聚焦 Agent 内部更靠近模型的一层：数据构建、QLoRA 后训练、结构化评测和工具执行边界。

## 一条主线

```text
最近 6 轮对话 + 上一轮状态 + 工具说明
                    │
                    ▼
              Gemma 3-1B
                    │
                    ▼
 belief_state + decision + tool_call + missing_slots
                    │
                    ▼
        Schema 校验 + 工具白名单 + 风险策略
          │              │               │
       只读查询        参数不足         打车请求
       自动执行        继续追问         人工确认
```

模型负责“理解和建议”，确定性代码负责“校验和授权”：

- `call_tool`：参数完整时提出白名单工具调用；
- `ask_user`：地铁/打车缺少出发地或目的地时先追问；
- `no_tool`：致谢、确认等不需要外部能力的轮次不误调工具；
- 只读工具通过校验后查询本地 SQLite；
- `request_taxi` 被注册为副作用工具，即使模型选择正确也只能停在人工确认，不会真实派车。

## 数据与任务

数据来自 [CrossWOZ](https://github.com/thu-coai/CrossWOZ) 官方 train/val/test 划分。构建脚本没有把多轮对话拆成互不相干的单句，而是保留：

- 最近 6 轮对话历史；
- 上一轮累计 `belief_state`；
- 当前用户输入；
- CrossWOZ 动态 `user_state` 对应的目标状态；
- 下一系统轮 `sys_state_init` 对应的只读查询参数。

当前提交的数据规模：

| 划分 | 样本数 | 说明 |
| --- | ---: | --- |
| train | 1,968 | 分层抽样的真实轮次 + 160 条缺参安全样本 |
| validation | 440 | 400 条真实轮次 + 40 条缺参安全样本 |
| test | 440 | 400 条真实轮次 + 40 条缺参安全样本 |
| challenge | 80 | 指代、修正、跨领域、拒绝误调用等难例切片 |

CrossWOZ 几乎总会提供完整路线参数，无法直接检验模型是否会在缺参时停下来。因此项目从各自数据划分中的真实地点值构造了少量、显式标记为 `controlled_required_slot_ablation` 的安全样本；具体数量和标签来源写在 [`data/tool_use/manifest.json`](data/tool_use/manifest.json)，不会把受控构造样本冒充自然对话。

## 实验结果

在 RTX 3060 Laptop 6GB 上冻结 Gemma 3-1B 基座，以 4bit QLoRA 训练 1 个 epoch（1,968 条训练样本、440 条验证样本），再在互不重叠的 440 条 test 和 80 条 challenge 上使用贪心解码评测。

| 指标 | Base（test） | QLoRA（test） | QLoRA（challenge） |
| --- | ---: | ---: | ---: |
| 严格 JSON 合法率 | 0.00% | **96.36%** | **93.75%** |
| Schema 合法率 | 2.50% | **98.41%** | **98.75%** |
| 状态 Slot F1 | 1.71% | **89.17%** | **89.59%** |
| 决策准确率 | 2.50% | **91.82%** | **91.25%** |
| 工具选择准确率 | 1.14% | **86.93%** | **81.58%** |
| 参数 Slot F1 | 0.00% | **68.63%** | **59.61%** |
| 参数完全匹配率 | 0.00% | **31.82%** | **23.68%** |
| 追问决策准确率 | 0.00% | **100.00%** | **100.00%** |
| 非工具轮误调用率 ↓ | 0.00%¹ | 11.36% | **9.52%** |

¹ Base 的 0% 来自几乎无法产生合法结构和工具调用，并不代表它具备更好的拒绝误调用能力。

结果表明，后训练对结构化输出、状态维护和决策路由的提升能够延续到 challenge 难例；同时参数精确生成仍是明确短板。完整对比、决策混淆矩阵、状态长度分桶和失败样例见 [`reports/EXPERIMENT_REPORT.md`](reports/EXPERIMENT_REPORT.md)。项目保留参数完全匹配率，不以较宽松的 Slot F1 替代严格指标，也不因 10 条 test / 2 条 challenge 路线样本量过小而主推必填参数准确率。

## 输出契约与安全边界

模型必须只输出一个 JSON 对象：

```json
{
  "belief_state": [
    {
      "goal_id": 1,
      "domain": "地铁",
      "constraints": {"出发地": "故宫"},
      "requested_fields": []
    }
  ],
  "decision": "ask_user",
  "tool_call": null,
  "missing_slots": ["目的地"]
}
```

运行时不直接相信这段 JSON。[`tool_use.py`](src/gemma_eval/tool_use.py) 会依次检查字段、枚举、工具名、必填参数和风险级别。风险不是模型预测出来的，而是固定在工具注册表中；这样模型无法通过输出一句“这是安全操作”绕过人工确认。

## 本地运行

环境要求：Python 3.10+；训练和 4bit 推理需要 NVIDIA GPU。首次运行前，需要在 Hugging Face 接受 [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) 许可，并执行一次 `hf auth login`，或将 `.env.example` 复制为 `.env` 后填入只读 Token。

`nvidia-smi` 能看到显卡不等于当前 Python 安装了 CUDA 版 PyTorch。训练前先确认下面命令输出 `True`；若为 `False`，需要按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 为当前 Python 安装与驱动匹配的 CUDA 版本。

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

本机已经配置 `C:\anaconda\envs\gemma-workorder` 时，可以不激活 Conda，直接用统一入口：

```powershell
.\run_local.ps1 -Mode check
.\run_local.ps1 -Mode smoke
.\run_local.ps1 -Mode train
.\run_local.ps1 -Mode eval-base
.\run_local.ps1 -Mode eval-qlora
.\run_local.ps1 -Mode eval-challenge
.\run_local.ps1 -Mode analyze
.\run_local.ps1 -Mode demo
```

正式使用按 `train → eval-base → eval-qlora → eval-challenge → analyze → demo` 的顺序运行即可。已有三份原始报告时，`analyze` 只做离线统计，不加载模型。

仓库已经提交处理后的数据，无需为了训练重新下载 CrossWOZ：

```powershell
# 1. Base 模型基线；可先加 --limit 20 做小规模检查
python scripts/evaluate_tool_use.py --precision 4bit `
  --output reports/tool_use_base_4bit.json

# 2. 4bit QLoRA；先用 --max-train-samples 32 验证本机环境，再跑完整训练
python scripts/train_tool_use_qlora.py --max-train-samples 32 `
  --max-validation-samples 32 --epochs 1 `
  --output-dir artifacts/tool_use_smoke_adapter
python scripts/train_tool_use_qlora.py `
  --output-dir artifacts/tool_use_qlora_adapter

# 3. 在同一测试集上评测 Adapter
python scripts/evaluate_tool_use.py `
  --adapter artifacts/tool_use_qlora_adapter `
  --output reports/tool_use_qlora_4bit.json

# 4. 在独立难例切片上评测
python scripts/evaluate_tool_use.py `
  --adapter artifacts/tool_use_qlora_adapter `
  --dataset data/tool_use/challenge.jsonl `
  --output reports/tool_use_qlora_challenge_4bit.json

# 5. 选一个 challenge 样本跑完整的“模型—策略—本地工具”链路
python scripts/run_tool_use_demo.py `
  --adapter artifacts/tool_use_qlora_adapter --index 0

# 6. 从保存的原始输出生成可提交的实验与错误分析报告
python scripts/analyze_tool_use_results.py
```

如需改变抽样规模，单独下载 CrossWOZ 后运行：

```powershell
python scripts/build_tool_use_dataset.py `
  --crosswoz-dir C:\path\to\CrossWOZ\data\crosswoz `
  --output-dir data\tool_use
```

## 评测口径

[`evaluate_tool_use.py`](scripts/evaluate_tool_use.py) 同时报告：

- 严格 JSON 合法率与 Schema 合法率；
- 状态联合准确率（JGA）与槽位 Precision / Recall / F1；
- `call_tool / ask_user / no_tool` 决策准确率；
- 工具选择准确率与参数完全匹配率；
- 参数 Slot Precision / Recall / F1 与少量路线工具的必填参数准确率；
- 不该调用工具时的误调用率。

Base 和 QLoRA 使用同一数据、同一 Prompt、同一贪心解码设置。README 数字来自本机保存的完整输出，并由分析脚本重新关联固定数据集计算；原始逐样本 JSON 默认不提交，避免仓库体积膨胀。

## 目录

```text
data/tool_use/                  固定划分、challenge 切片与本地参考数据
src/gemma_eval/tool_use.py      Schema、工具注册表、策略与本地执行
src/gemma_eval/tool_use_data.py Prompt、JSONL 校验与指标
scripts/build_tool_use_dataset.py
scripts/train_tool_use_qlora.py
scripts/evaluate_tool_use.py
scripts/analyze_tool_use_results.py
scripts/run_tool_use_demo.py
tests/test_tool_use.py          主链路测试
docs/                           设计、学习说明与模型原理附录
```

`decoding.py`、`gemma3_core.py` 及其测试是早期学习 Gemma 自回归生成、KV Cache 和 Decoder 结构时留下的原理附录，不属于当前求职主线，也不参与上述模型指标。

## 项目边界

- 本地工具使用 CrossWOZ 裁剪数据，只用于可复现实验，不代表实时商户信息；
- 打车工具仅模拟“等待确认”，没有接入真实派车系统；
- 受控缺参样本用于安全能力测试，不代表 CrossWOZ 原生分布；
- 该项目验证的是小模型的上下文状态与工具决策能力，不宣称实现通用 Agent；
- Adapter、基础模型和 Token 均不提交到 Git。

详细设计见 [`docs/系统设计与求职定位.md`](docs/系统设计与求职定位.md)，从零熟悉与面试准备顺序见 [`docs/项目学习与面试手册.md`](docs/项目学习与面试手册.md)。
