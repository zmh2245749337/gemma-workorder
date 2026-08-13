# Gemma-WorkOrder：Colab 实跑操作

这份操作用于产出可以如实写入项目复盘的 QLoRA 训练、结构化评测和 Adapter 合并结果。数据集是仓库内生成的 **90 条受控自建样本**，因此结果只能说明该受控任务上的表现，不能写成企业生产数据或通用设备诊断能力。

## 0. 运行前边界

- 需要 Hugging Face 已获 `google/gemma-3-1b-it` 访问权的 Token；Token 放入 Colab Secrets 的 `HF_TOKEN`，不要提交到 GitHub。
- 在 Colab 里选择 GPU 运行时；T4 已足够跑小规模 QLoRA。
- 训练输出写在 `artifacts/`，该目录默认不提交。请只将**训练参数、日志、评测报告中的真实数字**摘进 README/复盘，不上传 Token 或模型权重。

## 1. 准备环境

新建 Colab Notebook，依次运行：

```python
!git clone https://github.com/zmh2245749337/gemma-workorder.git
%cd gemma-workorder
!pip -q install -r requirements-colab.txt
```

在左侧钥匙图标的 Secrets 中创建 `HF_TOKEN`，打开“Notebook access”，然后运行：

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!nvidia-smi
```

如果最后一行没有 GPU 信息，停止运行，重新在“运行时类型”中选择 GPU 后再继续。

## 2. 生成并检查受控数据

```python
!python scripts/build_workorder_dataset.py --output-dir data/workorder --samples 90 --seed 42
!head -n 1 data/workorder/train.jsonl
```

输入为中文故障描述，输出是固定 JSON Schema。这个设计的目的不是让模型给出维修意见，而是训练它做受约束字段抽取和下一步**只读**查询选择。

## 3. 先跑 Base Gemma 评测（对照组）

```python
!python scripts/evaluate_workorder.py \
  --model-id google/gemma-3-1b-it \
  --precision 4bit \
  --dataset data/workorder/test.jsonl \
  --output reports/workorder_base_4bit.json
```

记下输出中的：`json_valid_rate`、`field_exact_rate`、`tool_accuracy`、`record_exact_rate`。它们是后续微调前后的可比指标。

## 4. 运行 QLoRA

```python
!python scripts/train_workorder_qlora.py \
  --model-id google/gemma-3-1b-it \
  --train data/workorder/train.jsonl \
  --validation data/workorder/validation.jsonl \
  --output-dir artifacts/workorder_qlora_adapter \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --max-length 1024 \
  --seed 42
```

训练结束后确认文件存在：

```python
!ls artifacts/workorder_qlora_adapter
!cat artifacts/workorder_qlora_adapter/training_metrics.json
```

## 5. 用 Adapter 评测

```python
!python scripts/evaluate_workorder.py \
  --model-id google/gemma-3-1b-it \
  --adapter artifacts/workorder_qlora_adapter \
  --precision 4bit \
  --dataset data/workorder/test.jsonl \
  --output reports/workorder_qlora_4bit.json
```

比较 `reports/workorder_base_4bit.json` 与 `reports/workorder_qlora_4bit.json`。如果测试集太小、样本模式过于相似，指标会偏乐观；在复盘中必须写清楚“受控自建集”，并保留失败样本，不挑着只展示成功案例。

## 6. 合并 Adapter，并做白盒复核

```python
!python scripts/merge_workorder_adapter.py \
  --model-id google/gemma-3-1b-it \
  --adapter artifacts/workorder_qlora_adapter \
  --output-dir artifacts/workorder_merged

!python scripts/run_core_alignment.py \
  --model-id artifacts/workorder_merged \
  --output reports/workorder_merged_core_alignment.json
```

这里的白盒复核沿用项目已有的 Gemma Decoder 和官方权重映射：它验证合并后的 checkpoint 是否还能被自研核心加载并做数值对齐。它不是 QLoRA 效果指标，不能替代第 5 步的任务评测。

## 7. 本地接口演示

在本地或 Colab 中先跑不依赖模型的安全基线：

```python
!python scripts/serve_workorder.py --mode baseline --host 0.0.0.0 --port 8000
```

另开一个终端/单元请求：

```bash
curl -X POST http://127.0.0.1:8000/api/workorders/parse \
  -H "Content-Type: application/json" \
  -d '{"text":"A区3号风机今天上午频繁停机，面板显示E07，重启十分钟后再次停止"}'
```

接口只会生成 `draft_requires_human_confirmation` 草稿。故障码释义来自本地 SQLite 参考表；系统没有写入、自动派单、自动维修或诊断决策能力。

## 8. 最后记录什么

建议记录一张表：基础模型/QLoRA、精度（FP16 或 4bit）、四项任务指标、一次成功 JSON、一次失败 JSON、T4 显存与速度。简历只写你亲自跑出的数字，并在面试中主动说明数据集规模、边界和失败案例。
