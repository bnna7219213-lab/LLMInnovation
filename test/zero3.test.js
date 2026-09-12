/**
 * zero3.test.js — Phase 2 ZeRO-3 + 分布式测试
 *
 * 测试内容:
 *   1. ZeRO3Shard: 注册/收集/释放参数
 *   2. ZeRO3Context: 训练流程模拟
 *   3. DistributedContext: AllReduce/AllGather/ReduceScatter/Broadcast
 *   4. Transformer20B: 配置正确性
 */

const { ZeRO3Shard, ZeRO3Context } = require('../src/training/zero3.js');
const { allReduce, allGather, reduceScatter, broadcast, DistributedContext } = require('../src/training/distributed.js');
const { CONFIG_20B, TransformerModel } = require('../src/models/text/transformer20b.js');
const assert = require('assert');
const { describe, it } = require('node:test');

// ============ ZeRO3Shard ============

describe('ZeRO3Shard', () => {
  it('注册参数: 分片正确', () => {
    const shard = new ZeRO3Shard(4, 1); // 4 个 shard, 当前是第 1 个 (0-indexed)
    const param = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]); // 8 元素
    shard.registerParam('test', param);

    const names = shard.getParamNames();
    assert.deepStrictEqual(names, ['test']);
  });

  it('gatherParam: 收集完整参数', () => {
    const shard = new ZeRO3Shard(4, 0);
    const param = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    shard.registerParam('test', param);

    const full = shard.gatherParam('test');
    assert.strictEqual(full.length, 8);
    // shard 0 存 [1,2], 其他位置为 0 (模拟)
    assert.strictEqual(full[0], 1);
    assert.strictEqual(full[1], 2);
    assert.strictEqual(full[2], 0); // 其他 shard 的模拟值
  });

  it('freeParam: 释放完整参数', () => {
    const shard = new ZeRO3Shard(2, 0);
    const param = new Float32Array([1, 2, 3, 4]);
    shard.registerParam('test', param);

    shard.gatherParam('test');
    shard.freeParam('test');

    // 再次 gather 会重建
    const full = shard.gatherParam('test');
    assert.strictEqual(full.length, 4);
  });

  it('reduceScatterGrad: 梯度分片正确', () => {
    const shard = new ZeRO3Shard(4, 1);
    const param = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    shard.registerParam('test', param);

    const fullGrad = new Float32Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const gradShard = shard.reduceScatterGrad('test', fullGrad);

    // shard 1 (0-indexed) 应存 [30, 40] (索引 2-3)
    assert.strictEqual(gradShard.length, 2);
    assert.strictEqual(gradShard[0], 30);
    assert.strictEqual(gradShard[1], 40);
  });

  it('toJSON: 序列化正确', () => {
    const shard = new ZeRO3Shard(2, 0);
    shard.registerParam('a', new Float32Array([1, 2, 3, 4]));
    const json = shard.toJSON();
    assert.strictEqual(json.numShards, 2);
    assert.strictEqual(json.shardIdx, 0);
    assert.ok(json.params['a']);
  });
});

// ============ ZeRO3Context ============

describe('ZeRO3Context', () => {
  it('完整训练流程模拟: gather → 计算 → free → reduceScatter', () => {
    const ctx = new ZeRO3Context(4, 0);

    ctx.registerParam('W1', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    ctx.registerParam('W2', new Float32Array([9, 10, 11, 12, 13, 14, 15, 16]));

    // 前向: 收集需要的参数
    ctx.gatherParams(['W1']);
    const fullW1 = ctx.shard.gatherParam('W1');
    assert.strictEqual(fullW1.length, 8);

    // 用完释放
    ctx.freeParams(['W1']);

    // 反向: 梯度分片
    const grad = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    ctx.reduceScatterGrad('W1', grad);

    const gradShard = ctx.shard.getGradShard('W1');
    assert.ok(gradShard.length > 0);
  });

  it('freeAllGathered: 释放所有已收集参数', () => {
    const ctx = new ZeRO3Context(2, 0);
    ctx.registerParam('a', new Float32Array([1, 2, 3, 4]));
    ctx.registerParam('b', new Float32Array([5, 6, 7, 8]));

    ctx.gatherParams(['a', 'b']);
    ctx.freeAllGathered();
    // 不抛错即为通过
  });
});

// ============ DistributedContext ============

describe('DistributedContext', () => {
  it('AllReduce: 单设备返回原数据', () => {
    const data = new Float32Array([1, 2, 3]);
    const result = allReduce(data, 1);
    assert.deepStrictEqual(result, data);
  });

  it('AllReduce: 多设备平均 (模拟)', () => {
    const data = new Float32Array([4, 8, 12]);
    const result = allReduce(data, 4);
    // 模拟: 假设所有设备数据相同, 平均 = data / numDevices
    assert.strictEqual(result[0], 1);
    assert.strictEqual(result[1], 2);
    assert.strictEqual(result[2], 3);
  });

  it('AllGather: 单设备返回原数据', () => {
    const data = new Float32Array([1, 2]);
    const result = allGather(data, 1, 0);
    assert.deepStrictEqual(result, data);
  });

  it('ReduceScatter: 分片正确', () => {
    const data = new Float32Array([10, 20, 30, 40]);
    const result = reduceScatter(data, 2, 1);
    // 2 设备, device 1: 取后半 [30, 40] / 2 = [15, 20]
    assert.strictEqual(result.length, 2);
    assert.strictEqual(result[0], 15);
    assert.strictEqual(result[1], 20);
  });

  it('DistributedContext: isMaster 正确', () => {
    const ctx0 = new DistributedContext(4, 0);
    const ctx1 = new DistributedContext(4, 1);
    assert.strictEqual(ctx0.isMaster, true);
    assert.strictEqual(ctx1.isMaster, false);
  });

  it('ifMaster: 只在主设备执行', () => {
    const ctx = new DistributedContext(2, 0);
    let called = false;
    ctx.ifMaster(() => { called = true; });
    assert.strictEqual(called, true);

    const ctx2 = new DistributedContext(2, 1);
    let called2 = false;
    ctx2.ifMaster(() => { called2 = true; });
    assert.strictEqual(called2, false);
  });
});

// ============ Transformer20B ============

describe('Transformer20B', () => {
  it('CONFIG_20B 参数正确', () => {
    assert.strictEqual(CONFIG_20B.vocabSize, 32768);
    assert.strictEqual(CONFIG_20B.dModel, 6144);
    assert.strictEqual(CONFIG_20B.numLayers, 48);
    assert.strictEqual(CONFIG_20B.numHeads, 24);
    assert.strictEqual(CONFIG_20B.numKVHeads, 8);
    assert.strictEqual(CONFIG_20B.dFF, 16384);
    assert.strictEqual(CONFIG_20B.maxSeqLen, 8192);
  });

  it('dModel / numHeads = dK 正确', () => {
    assert.strictEqual(CONFIG_20B.dModel / CONFIG_20B.numHeads, 256);
  });

  it('numHeads % numKVHeads === 0', () => {
    assert.strictEqual(CONFIG_20B.numHeads % CONFIG_20B.numKVHeads, 0);
  });
});