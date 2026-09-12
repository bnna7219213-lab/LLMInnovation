# Phase 3 — Mamba + Jamba 混合架构

非 Transformer 架构自研：选择性状态空间模型（Mamba）+ Transformer 混合（Jamba）。

测试：`node --test test/mamba.test.js`

## Selective Scan（`src/models/mamba/selective_scan.js`）

Mamba 的核心递归：

```
h_t = A_t · h_{t-1} + B_t · x_t
y_t = C_t · h_t
```

其中 A、B、C 都是**数据相关**（选择性），区别于固定参数的普通 SSM。

```js
const { selectiveScanSequential, selectiveScanParallel } =
  require('./src/models/mamba/selective_scan.js');

// 顺序扫描（推理，O(n)）
const out = selectiveScanSequential(x, A, B, C, seqLen, dState);

// 并行扫描（训练，O(n log n)）——当前实现内部回退到顺序扫描保证正确性
const out2 = selectiveScanParallel(x, A, B, C, seqLen, dState);
```

**注意**：`selectiveScanParallel` 目前是简化版，内部回退到 `selectiveScanSequential`（注释标注 warp-scan 为 future work）。seqLen 需为 2 的幂（内部会向上取整）。

## MambaBlock（`src/models/mamba/mamba_block.js`）

```js
const { MambaBlock } = require('./src/models/mamba/mamba_block.js');

const block = new MambaBlock(dModel = 512, dState = 16, dConv = 4);
const out = block.forward(x, seqLen);   // [seqLen, dModel]
block.countParams();
```

**数据流**：

```
x → 1D Conv (causal, kernel=dConv) → SiLU
  → 投影出 Δ, A, B, C, x_proj
  → Δ: softplus（保证正数）
  → A: -|A|（保证衰减）
  → B_scaled = Δ·B·x_proj, C_scaled = Δ·C, A_scaled = exp(Δ·A)
  → Selective Scan
  → 输出投影 → [seqLen, dModel]
```

**对比 Transformer**：

| 维度 | Transformer | Mamba |
|---|---|---|
| 时间复杂度 | O(n²) | O(n) |
| 推理显存 | O(n)（KV cache） | O(1)（无 KV cache） |
| 长序列 1M tokens | 不可行 | 可行 |

## JambaModel（`src/models/hybrid/jamba.js`）

Mamba 层与 Transformer 层交替：

```js
const { JambaModel } = require('./src/models/hybrid/jamba.js');

const model = new JambaModel({
  vocabSize: 32768, dModel: 512, numLayers: 8,
  numHeads: 8, numKVHeads: 4, dFF: 1024, maxSeqLen: 2048,
  dState: 16, dConv: 4,
});

const logits = model.forward([1,2,3,4], 4);
const tokens = model.generate([1,2], 20, 0);

// 层类型统计
model.countLayerTypes();  // { mamba: 4, transformer: 4 }
model.countParams();
```

**层交替规则**：偶数层 = Mamba，奇数层 = Transformer（Layer 0, 2, 4... 是 Mamba）。

**API**：`forward(inputIds, seqLen)`、`generate(promptIds, maxNewTokens, temperature)`、`countLayerTypes()`、`countParams()`

## 选型理由（plan.md 第 4 章）

- **Mamba**：O(n) 时间、O(1) 推理显存，长序列极优
- **Jamba**：交替结构互补——Mamba 处理全局依赖，Transformer 处理精确局部语义

## 测试要点

- MambaBlock 前向无 NaN、维度正确
- Jamba `countLayerTypes` 返回 mamba/transformer 数量相等
- 温度多样性测试需足够高的 temperature
