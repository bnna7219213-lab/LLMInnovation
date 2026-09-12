# Phase 6 — VLM + 长上下文 + 最终集成

多模态融合（视觉 + 语言）、100B MoE 配置、KV Cache 推理优化、评估基准。

测试：`node --test test/vlm.test.js`

## Vision Encoder（`src/models/vlm/vision_encoder.js`）

ViT 风格：图像 → patch → 视觉 token 序列。

```js
const { VisionEncoder } = require('./src/models/vlm/vision_encoder.js');

const enc = new VisionEncoder({
  imageSize: 32, patchSize: 8, channels: 3,
  dModel: 16, numLayers: 2, numHeads: 2,
});

// 前向：[H×W×C] → [numPatches, dModel]
const visTokens = enc.forward(image);

enc.numPatches;   // (imageSize/patchSize)²
enc.countParams();
```

**流程**：Patchify（图像切块）→ Patch Embedding（线性投影）→ 位置编码 → Transformer Encoder（full attention，非因果）→ Final norm。

**约束**：`imageSize` 必须被 `patchSize` 整除。

## VLM（`src/models/vlm/vlm.js`）

LLaVA 风格：视觉 token + 文本 token 拼接后走 LLM backbone。

```js
const { VLM } = require('./src/models/vlm/vlm.js');

const vlm = new VLM({
  vision: { imageSize: 16, patchSize: 8, channels: 3, dModel: 8, numLayers: 1, numHeads: 2 },
  vocabSize: 16, dModel: 8, numLayers: 2, numHeads: 2, numKVHeads: 2, dFF: 32,
});

// 联合前向：[image, tokenIds] → 文本位置 logits
const logits = vlm.forward(image, [1, 2, 3, 4]);   // [seqLen, vocabSize]

// 生成
const tokens = vlm.generate(image, [1, 2], maxNewTokens = 20, temperature = 0);
```

**架构**：

```
图像 → VisionEncoder → 视觉 token [numPatches, dVis]
        ↓ linear projection (dVis → dModel)
视觉 token + 文本 token embedding → concat → LLM backbone → 文本 logits
```

**API**：`encodeImage(image)`、`embedText(tokenIds)`、`forward(image, tokenIds)`、`generate(image, promptIds, maxNewTokens, temperature)`、`countParams()`

## 100B MoE（`src/models/text/transformer100b.js`）

稀疏 Mixture-of-Experts 配置 + 路由。

```js
const { MoERouter, CONFIG_100B, estimateParams } =
  require('./src/models/text/transformer100b.js');

// MoE 路由：top-k 专家选择
const router = new MoERouter({ dModel: 8192, numExperts: 8, topK: 2 });
const { expertIdx, weights } = router.route(x); // 每个 token 选 2 个专家

// 100B 配置
CONFIG_100B.numLayers;   // 64
CONFIG_100B.moe.numExperts; // 8

// 参数量估算
estimateParams();        // ~113.8B
```

**CONFIG_100B**（Mixtral 8x22B 风格）：

| 参数 | 值 |
|---|---|
| vocabSize | 128000 |
| dModel | 8192 |
| numLayers | 64 |
| numHeads | 64 |
| numKVHeads | 8（GQA） |
| dFF | 8192（专家内） |
| numExperts | 8 |
| topK | 2 |
| maxSeqLen | 32768 |

**优势**：总参数 ~114B，但每 token 只激活 ~14B（计算效率高），专家并行跨多 GPU。

**坑**：`Float32Array.map()` 返回的是 Float32Array，对象会被强制转 NaN——路由排序需用 `Array.from(probs).map(...)`。

## KV Cache（`src/inference/kv_cache.js`）

Autoregressive 推理核心优化：缓存历史 K/V，只算新 token。

```js
const { KVCache, quantizeInt8, dequantizeInt8 } = require('./src/inference/kv_cache.js');

const cache = new KVCache({
  numLayers: 32, numKVHeads: 8, dK: 128, maxSeqLen: 4096,
  windowSize: 0,      // 滑动窗口（0 = 无限）
  quantize: false,    // int8 量化
});

// 存储新 token 的 KV
cache.store(layerIdx, keys, values);

// 读取（考虑滑动窗口）
const { keys, values, len, offset } = cache.retrieve(layerIdx);

cache.clear();
cache.memoryBytes();       // 当前占用
cache.peakMemoryBytes();   // 峰值

// int8 对称量化
const { data, scale } = quantizeInt8(x);   // Float32Array → { Int8Array, scale }
const restored = dequantizeInt8(data, scale); // 反量化（有精度损失）
```

**内存估算**（3B, GQA=8, dK=128）：每层 KV = 2 × 8 × 128 × maxSeqLen × 2 bytes ≈ 16.8MB，32 层 ≈ 537MB。

**量化公式**：对称量化 `scale = max|x|/127`，`q = round(clip(x/scale, ±127))`，反量化 `x' = q·scale`。

**注意**：量化 `clip(x/scale, ±127)` 与反量化 `q·scale` 严格对称（之前 `/127` 导致尺度错配，已修复）。

## 评估基准（`src/eval/benchmarks.js`）

```js
const { perplexity, clozeAccuracy, repetitionRate, diversity, runBenchmarks } =
  require('./src/eval/benchmarks.js');

// 困惑度：exp(-avg log P(token|context))
const ppl = perplexity(tokenIds, forwardFn, vocabSize);

// 完形填空 top-1 准确率
const acc = clozeAccuracy(contexts, targets, forwardFn);

// n-gram 重复率（越低越好）
const rep = repetitionRate(generated, n = 2);

// 样本间多样性（越高越好）
const div = diversity(samples, n = 3);

// 基准运行器（计时）
const results = runBenchmarks([{ name: 'ppl', fn: () => perplexity(...) }]);
```

## 长上下文（NTK RoPE）

RoPE 外推由 `src/nn/rope.js` 的 `applyRoPE` 支持：

```js
// scale=8 → 8x context 外推（位置可到 16384）
applyRoPE(x, nVec, dModel, positions, base = 10000, scale = 8, alpha = 1);
```

NTK 插值：`base' = base · scale^((scale-1)/alpha)`，数学形式简洁（旋转矩阵在 2D 子空间操作）。

## 测试要点

- MoE 路由权重归一化、专家索引互异
- KVCache 滑动窗口 offset 正确
- 量化-反量化误差有界
- perplexity 均匀分布 ≈ vocabSize
- NTK scale 外推无 NaN、不同 scale 编码不同
