# Gemma 3 核心架构复现说明

这部分代码的目的不是重新训练 10 亿参数，也不是替代 Hugging Face 的高性能实现，而是回答一个更扎实的问题：**模型内部的一层到底怎样把输入变成输出，为什么增量生成可以复用 KV Cache？**

## 1. Gemma 3 1B 的结构参数

| 项目 | 数值 | 含义 |
| --- | ---: | --- |
| Decoder 层数 | 26 | 串联 26 个 Transformer Block |
| 隐藏维度 | 1152 | 每个 token 的主干向量长度 |
| FFN 中间维度 | 6912 | GeGLU 的扩展维度 |
| Query Heads | 4 | 有 4 组查询头 |
| KV Heads | 1 | 所有 Query Head 共享一组 K/V，即 MQA |
| Head Dim | 256 | 每个注意力头的维度 |
| Sliding Window | 512 | 局部层只看最近 512 个 token |
| 层模式 | 5:1 | 5 层局部注意力后接 1 层全局注意力 |

这些默认值写在 `Gemma3CoreConfig` 中；实际对齐时也会从官方模型配置读取，避免把代码绑死在某个版本。

## 2. 一层 Decoder 的数据流

```text
hidden states
    │
    ├─ RMSNorm ─ MQA Attention ─ RMSNorm ─┐
    │                                     │ residual add
    └─────────────────────────────────────┘
                         │
                         ├─ RMSNorm ─ GeGLU ─ RMSNorm ─┐
                         │                              │ residual add
                         └──────────────────────────────┘
```

Gemma 3 的 Block 不是简单的“两个 Pre-Norm”：注意力与 FFN 的输出后还各有一次 RMSNorm，因此代码中保留了四个归一化层。

### RMSNorm

先用均方根归一化，再乘以 `1 + weight`。参数以 0 为中心初始化，这一点与直接乘 `weight` 的普通写法不同。计算归一化时提升到 FP32，最后转回输入精度。

### RoPE

RoPE 将绝对位置变成 Query/Key 向量中的旋转。Gemma 3 的局部层与全局层使用不同的基频：1B 模型局部层为 10,000，全局层为 1,000,000。

### Multi-Query Attention

4 个 Query Head 只配 1 个 KV Head。计算注意力前，K/V 在逻辑上扩展到 4 个头；实际 KV Cache 只需要保存原始的一组 K/V，从而降低显存占用。

### Sliding Window + Global Attention

局部层只允许 Query 看到最近 512 个位置；每第 6 层使用全局注意力补充远距离信息。因此自定义 Cache 对局部层裁剪历史，对全局层保留完整历史，这就是项目里的 Hybrid KV Cache。

### GeGLU

FFN 有 gate、up、down 三个线性层。gate 分支经过近似 tanh 的 GELU 后，与 up 分支逐元素相乘，再通过 down 投影回隐藏维度。

## 3. 怎样证明实现不是“看起来差不多”

仓库采用三层证据：

1. **结构测试**：检查 5:1 层模式、MQA 投影形状、局部 Mask 和权重共享。
2. **行为测试**：小模型上比较完整前向与 `prefill + 单 token decode`，验证 KV Cache 不改变结果。
3. **官方权重对齐**：把 `google/gemma-3-1b-it` 的权重复制到独立实现，比较第 0、5、25 层和最终 logits 的最大误差、平均误差、余弦相似度与 top-1 token。

Colab 命令：

```bash
python scripts/run_core_alignment.py \
  --precision fp16 \
  --layers 0 5 25 \
  --output reports/core_alignment.json
```

第 0、25 层覆盖局部注意力，第 5 层覆盖全局注意力。只有实际生成的 `reports/core_alignment.json` 才是可写入简历的数值证据。

### 已验证结果

2026-08-12 在 Tesla T4、FP16 环境完成一次官方权重对齐。第 0、5、25 层以及最后一个 token 的 logits 最大/平均绝对误差均为 `0.0`，官方实现与独立实现的 top-1 token ID 均为 `57137`。原始记录位于 [`reports/core_alignment_20260812/core_alignment.json`](../reports/core_alignment_20260812/core_alignment.json)。该结果证明本次记录的输入和环境下两条前向链路一致，不代表未经测试的所有输入与环境。

## 4. 面试时必须能讲清的边界

- 这是**推理内核复现与权重对齐**，不是从零预训练模型。
- 官方权重来自 `google/gemma-3-1b-it`，自己实现的是前向计算、Mask、位置编码和 Cache 数据流。
- eager PyTorch 版本优先可读性和可验证性，不会比 FlashAttention/编译内核更快。
- 旧的 20 题评测只是部署配置的回归检查，不代表通用模型能力；项目主线已经改为结构复现和推理优化。

## 5. 官方资料

- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [Google Gemma 3 Model Card](https://ai.google.dev/gemma/docs/core/model_card_3)
- [Hugging Face Gemma 3 文档](https://huggingface.co/docs/transformers/main/en/model_doc/gemma3)
- [Transformers Gemma 3 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py)
