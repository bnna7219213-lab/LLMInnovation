# Phase 1 — 3B 基线 Transformer

完整的 Decoder-only Transformer（LLaMA/Mistral 风格），是后续所有文本任务的 backbone。

测试：`node --test test/transformer3b.test.js`

## TransformerModel（`src/models/text/transformer3b.js`）

```js
const { TransformerModel, CONFIG_3B } = require('./src/models/text/transformer3b.js');

// 生产配置：3B 参数
const model = new TransformerModel(CONFIG_3B);

// 测试用微型配置（验证同一代码路径）
const tiny = new TransformerModel({
  vocabSize: 256, dModel: 16, numLayers: 2,
  numHeads: 4, numKVHeads: 2, dFF: 32, maxSeqLen: 64,
});

// 前向：输入 token ids → logits
const logits = model.forward([1, 2, 3, 4], 4); // [seqLen, vocabSize]

// 生成：greedy / 温度采样
const tokens = model.generate([1, 2], 20, 0);      // temperature=0 → greedy
const sampled = model.generate([1, 2], 20, 0.8);   // 采样
```

**API**：`forward(inputIds, seqLen)`、`generate(promptIds, maxNewTokens, temperature)`、`getParams()`、`countParams()`

## CONFIG_3B

| 参数 | 值 |
|---|---|
| vocabSize | 32768 |
| dModel | 2560 |
| numLayers | 32 |
| numHeads | 32 |
| numKVHeads | 8（GQA） |
| dFF | 6912 |
| maxSeqLen | 2048 |
| ropeBase | 10000 |

## 架构要点

```
Embedding (vocabSize × dModel)
  ↓
× numLayers:  x = x + MHA(RMSNorm(x))   ← Pre-Norm residual
              x = x + SwiGLU(RMSNorm(x))
  ↓
Final RMSNorm → Linear (tie embedding: Woutput = Wembedᵀ)
```

- **Tie embedding**：输出投影复用 `Wembed` 的转置，不额外存储，省 ~256M 参数
- **GQA**：32 个 Q head 共享 8 个 KV head（每组 4 个 Q 共享 1 组 KV），KV cache 减 4x
- **RoPE**：在 attention 输入上应用旋转位置编码

## 关键实现细节（坑）

1. **因果掩码预分配**：构造函数一次性分配 `maxSeqLen × maxSeqLen`，`_getCausalMask(seqLen)` 截取子视图。**不要**逐层分配掩码（会导致索引错位）。
2. **RoPE 应用位置**：在 `multiHeadAttention` 之前对 `xNorm` 应用（本实现简化，实际应在 Q/K 投影后分别应用）。
3. **vocabSize ≥ 256**：BPE 初始词表 256，模型 vocabSize 必须 ≥ 256。
4. **token_id 越界检查**：`forward` 会校验 `0 ≤ tid < vocabSize`。

## 测试要点

- 微型配置参数量断言（不能直接用 3B 配置跑测试，内存爆炸）
- OOV token（256 ≥ vocabSize=256 时边界）
- 温度多样性：temperature>0 时多次采样输出不同
