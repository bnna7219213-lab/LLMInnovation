# Phase 0 — 基础设施

自研 LLM 的地基：分词器、神经网络原语、训练基建。所有后续 Phase 都依赖这一层。

测试：`node --test test/tokenizer.test.js test/nn.test.js`（41 个用例）

## 1. BPE 分词器

### Vocab（`src/tokenizer/vocab.js`）

维护 `token_id → 字节序列` 的映射。初始词表固定为 256 个 UTF-8 单字节 token。

```js
const { Vocab, BASE_BYTES } = require('./src/tokenizer/vocab.js');

const vocab = new Vocab();          // 初始 256 字节 token
console.log(vocab.size);            // 256
vocab.add(new Int32Array([104, 105])); // 添加合并结果 "hi"
vocab.getBytes(256);                // Int32Array [104, 105]
vocab.byteToId(104);                // 104（单字节直接映射）

// 序列化
const json = vocab.toJSON();
const restored = Vocab.fromJSON(json);
```

**API**：`size`（getter）、`add(bytes)`、`getBytes(tokenId)`、`byteToId(byte)`、`toJSON()`、`static fromJSON(data)`

**约束**：仅支持初始词表大小 256，传其他值会抛错。

### BPETokenizer（`src/tokenizer/bpe.js`）

字节对编码，训练 merges 规则后用于编码/解码。

```js
const { BPETokenizer } = require('./src/tokenizer/bpe.js');

// 训练
const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
tokenizer.train(['hello world', 'hello there', 'world hello']);

// 编码/解码
const ids = tokenizer.encode('hello');
const text = tokenizer.decode(ids);   // 'hello'

// 序列化
const json = tokenizer.toJSON();
const restored = BPETokenizer.fromJSON(json);
```

**API**：`train(corpus, targetSize)`、`encode(text)`、`decode(ids)`、`toJSON()`、`static fromJSON(data)`

**注意**：`encode` 使用贪心最长匹配（Trie 思路），merges 按字节序列长度降序匹配。

## 2. 神经网络原语

### RMSNorm（`src/nn/rmsnorm.js`）

```js
const { rmsNorm, rmsNormBatch } = require('./src/nn/rmsnorm.js');

const gamma = new Float32Array(8).fill(1);
const x = new Float32Array(16); // 2 个 8 维向量
const out = rmsNorm(x, gamma, 1e-6);          // 单向量
const out2 = rmsNormBatch(x, 2, 8, gamma);     // 批量 [rows, dModel]
```

公式：`output = x / sqrt(mean(x²) + eps) * gamma`

### SwiGLU（`src/nn/swiglu.js`）

```js
const { swiglu } = require('./src/nn/swiglu.js');
// swiglu(x, nVec, dModel, dFF, W1, W2, W3)
// W1/W2: [dFF, dModel], W3: [dModel, dFF]
```

### RoPE + NTK（`src/nn/rope.js`）

```js
const { applyRoPE } = require('./src/nn/rope.js');
// applyRoPE(x, nVec, dModel, positions, base=10000, scale=1, alpha=1)
const out = applyRoPE(x, 4, 16, [0,1,2,3], 10000, 8, 1); // scale=8 → 8x 外推
```

NTK 插值：`base' = base * scale^((scale-1)/alpha)`。dModel 必须为偶数。

### Attention（`src/nn/attention.js`）

```js
const { multiHeadAttention, matmul } = require('./src/nn/attention.js');
// multiHeadAttention(x, nSeq, dModel, numHeads, dK, numKVHeads, WQ, WK, WV, WO, mask)
// GQA: numKVHeads < numHeads（须整除）
// mask: [nSeq, nSeq]，非 0 表示遮盖（-∞）
```

## 3. 优化器与调度器

### AdamW（`src/nn/optimizer.js`）

```js
const { AdamW } = require('./src/nn/optimizer.js');

const opt = new AdamW({ learningRate: 3e-4, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1 });
const state = opt.createState(param);

// 单参数更新（step +1）
opt.stepUpdate(param, grad, state);

// 批量更新（所有参数共享同一个 step，只递增 1 次）★ 推荐
opt.update([{ param, grad }, { param: p2, grad: g2 }], [state1, state2]);
```

**坑**：不要循环调用 `stepUpdate` 更新多个参数——会导致 step 被多次递增，偏差修正错误。多参数请用 `update()`。

### LRScheduler（`src/training/scheduler.js`）

```js
const { LRScheduler } = require('./src/training/scheduler.js');

const sched = new LRScheduler({
  maxLR: 3e-4, minLR: 0, warmupSteps: 2000,
  totalSteps: 100000, schedule: 'cosine', // 'cosine' | 'linear' | 'step'
});
const lr = sched.getLR(5000);
```

## 4. Checkpoint 与分布式

### Checkpoint（`src/training/checkpoint.js`）

```js
const { saveCheckpoint, loadCheckpoint } = require('./src/training/checkpoint.js');

const params = { 'Wembed': model.Wembed, 'layer0.WQ': model.layers[0].WQ };
saveCheckpoint(params, optimizerState, { step: 100, loss: 2.3, learningRate: 3e-4 }, './ckpt.json');
const { params: loaded } = loadCheckpoint('./ckpt.json');
```

### 分布式（`src/training/distributed.js`）

```js
const { DistributedContext, allReduce, allGather, reduceScatter, broadcast } = require('./src/training/distributed.js');

const ctx = new DistributedContext(numDevices, deviceId);
ctx.allReduce(grad);      // 梯度求和平均
ctx.ifMaster(() => save()); // 仅主设备执行
```

单进程模式下通信为纯模拟（直接返回/分片），预留多进程扩展接口。

## Phase 0 测试要点

- tokenizer：小语料无法到达目标词表，用范围断言而非精确断言
- optimizer：`update()` step 只递增 1 次
- rope：NTK scale 验证 freq 维度对
