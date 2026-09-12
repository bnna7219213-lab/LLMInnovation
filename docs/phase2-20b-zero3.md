# Phase 2 — 20B 模型 + ZeRO-3 + 分布式

把 3B 架构放大到 20B，并引入显存优化的训练基础设施。

测试：`node --test test/zero3.test.js`

## 20B 模型（`src/models/text/transformer20b.js`）

```js
const { TransformerModel, CONFIG_20B, createTransformer20B } = require('./src/models/text/transformer20b.js');

const model = createTransformer20B();   // 等价于 new TransformerModel(CONFIG_20B)
```

**CONFIG_20B**：vocabSize=32768, dModel=6144, numLayers=48, numHeads=24, numKVHeads=8, dFF=16384, maxSeqLen=8192

> 与 3B 完全相同的架构（复用 `TransformerModel`），仅配置不同。总参数 ~20B。

## ZeRO Stage-3（`src/training/zero3.js`）

参数分片到所有 GPU，需要时 AllGather，用完立即释放。

```js
const { ZeRO3Shard, ZeRO3Context } = require('./src/training/zero3.js');

// 低级 API：分片参数管理
const shard = new ZeRO3Shard(numShards = 4, shardIdx = 0);
shard.registerParam('layer0.WQ', model.layers[0].WQ); // 只存 1/4
const full = shard.gatherParam('layer0.WQ');           // 拼回完整（模拟 AllGather）
shard.freeParam('layer0.WQ');                          // 用完释放
shard.reduceScatterGrad('layer0.WQ', fullGrad);        // 梯度分片

// 高级 API：训练上下文管理器
const zero3 = new ZeRO3Context(4, 0);
zero3.registerParam('layer0.WQ', model.layers[0].WQ);
zero3.gatherParams(['layer0.WQ', 'layer0.WK']);   // 前向：收集
// ... 计算 ...
zero3.freeParams(['layer0.WQ', 'layer0.WK']);     // 释放
zero3.reduceScatterGrad('layer0.WQ', grad);       // 反向：分片梯度
```

**核心思想**：

| 模式 | 每 GPU 内存 |
|---|---|
| 传统训练 | 参数 + 梯度 + 优化器状态 = 完整 × 3（约 8x 参数量含 fp16 master） |
| ZeRO-3 | 每项都 1/N 分片 |

**API**：
- `ZeRO3Shard`：`registerParam(name, fullParam)`、`gatherParam(name)`、`freeParam(name)`、`setGradShard(name, shard)`、`reduceScatterGrad(name, fullGrad)`、`registerOptState(name, fullState)`、`toJSON()`
- `ZeRO3Context`：`registerParam`、`gatherParams(names)`、`freeParams(names)`、`freeAllGathered()`、`reduceScatterGrad`、`getShard()`

**注意**：单机模拟——`gatherParam` 只填自己的 shard，其他 shard 为 0（真实多设备需跨进程通信）。

## 分布式通信（`src/training/distributed.js`）

```js
const { allReduce, allGather, reduceScatter, broadcast, DistributedContext } =
  require('./src/training/distributed.js');

// AllReduce：梯度求和平均
allReduce(grad, numDevices);
// AllGather：收集所有设备数据
allGather(localData, numDevices, deviceId);
// ReduceScatter：先 reduce 再分片
reduceScatter(localData, numDevices, deviceId);
// Broadcast：广播
broadcast(data, sourceDevice, numDevices);
```

单进程模式下为纯模拟（直接返回或简单分片），预留多进程（Cluster/Child Process）扩展。

## 100B 训练瓶颈（plan.md 第 3 章）

- **显存**：ZeRO-3 + Activation Recomputation
- **通信**：分层 AllReduce 同步梯度
- 20B 训练需要 8×A100 或 4×H100；100B 需要 MoE 稀疏激活（见 Phase 6）
