# 数据说明

本目录数据由 [THU-CoAI/CrossWOZ](https://github.com/thu-coai/CrossWOZ) 官方 train/val/test 划分派生。CrossWOZ 采用 Apache License 2.0；原始数据的版权和许可归原项目所有。本仓库的 MIT License 不替代原始数据许可。

- `train.jsonl`、`validation.jsonl`、`test.jsonl`：固定随机种子产生的模型训练与评测数据；
- `challenge.jsonl`：指代、修正、跨领域、缺参和不误调用等难例切片；
- `reference/`：当前固定划分提到的景点、酒店、餐馆和地铁数据子集；
- `manifest.json`：样本数量、路由分布和标签来源。

每条 `source.type == "controlled_required_slot_ablation"` 的记录都是显式构造的缺参安全样本，不属于 CrossWOZ 的自然话语分布。其地点值来自同一 split 内的真实记录，train/validation/test 之间没有跨 split 复制对话样本。
