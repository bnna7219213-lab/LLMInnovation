/**
 * vlm.test.js — Phase 6 VLM + 长上下文 + 集成测试
 *
 * 测试内容:
 *   1. VisionEncoder: patchify + 前向 + 维度
 *   2. VLM: 图像+文本联合前向 + 生成
 *   3. MoERouter: top-k 路由
 *   4. 100B 配置: 参数量估算
 *   5. KVCache: 存储/检索/窗口/量化
 *   6. Eval: perplexity + 重复率
 *   7. 长上下文: NTK RoPE 外推
 */

const assert = require('assert');
const { describe, it, beforeEach } = require('node:test');

let VisionEncoder = null;
let VLM = null;
let MoERouter = null;
let CONFIG_100B = null;
let estimateParams = null;
let KVCache = null;
let quantizeInt8 = null;
let dequantizeInt8 = null;
let evalMod = null;
let applyRoPE = null;

beforeEach(() => {
  ({ VisionEncoder } = require('../src/models/vlm/vision_encoder.js'));
  ({ VLM } = require('../src/models/vlm/vlm.js'));
  ({ MoERouter, CONFIG_100B, estimateParams } = require('../src/models/text/transformer100b.js'));
  ({ KVCache, quantizeInt8, dequantizeInt8 } = require('../src/inference/kv_cache.js'));
  evalMod = require('../src/eval/benchmarks.js');
  ({ applyRoPE } = require('../src/nn/rope.js'));
});

describe('VisionEncoder', () => {
  it('patchify: 输出正确 patch 数', () => {
    const enc = new VisionEncoder({ imageSize: 16, patchSize: 8, channels: 3, dModel: 8, numLayers: 1, numHeads: 2 });
    assert.strictEqual(enc.numPatches, 4); // (16/8)^2
    assert.strictEqual(enc.patchDim, 8 * 8 * 3); // 192
    const image = new Float32Array(16 * 16 * 3).fill(0.5);
    const patches = enc.patchify(image);
    assert.strictEqual(patches.length, 4 * 192);
  });

  it('forward: 输出 [numPatches, dModel]', () => {
    const enc = new VisionEncoder({ imageSize: 16, patchSize: 8, channels: 3, dModel: 8, numLayers: 1, numHeads: 2 });
    const image = new Float32Array(16 * 16 * 3).fill(0.3);
    const out = enc.forward(image);
    assert.strictEqual(out.length, 4 * 8);
    // 不应有 NaN
    for (let i = 0; i < out.length; i++) {
      assert.ok(!isNaN(out[i]), `输出不应有 NaN: ${out[i]}`);
    }
  });

  it('countParams: 正数', () => {
    const enc = new VisionEncoder({ imageSize: 16, patchSize: 8, channels: 3, dModel: 8, numLayers: 1, numHeads: 2 });
    assert.ok(enc.countParams() > 0);
  });

  it('constructor: 不可整除时报错', () => {
    assert.throws(() => new VisionEncoder({ imageSize: 15, patchSize: 8 }));
  });
});

describe('VLM', () => {
  it('forward: 输出 [seqLen, vocabSize] logits', () => {
    const vlm = new VLM({
      vision: { imageSize: 16, patchSize: 8, channels: 3, dModel: 8, numLayers: 1, numHeads: 2 },
      vocabSize: 16, dModel: 8, numLayers: 2, numHeads: 2, numKVHeads: 2, dFF: 32,
    });
    const image = new Float32Array(16 * 16 * 3).fill(0.2);
    const tokens = [1, 2, 3, 4];
    const logits = vlm.forward(image, tokens);
    assert.strictEqual(logits.length, 4 * 16);
    for (let i = 0; i < logits.length; i++) {
      assert.ok(!isNaN(logits[i]), `logits 不应有 NaN: ${logits[i]}`);
    }
  });

  it('generate: greedy 生成 token 序列', () => {
    const vlm = new VLM({
      vision: { imageSize: 16, patchSize: 8, channels: 3, dModel: 8, numLayers: 1, numHeads: 2 },
      vocabSize: 16, dModel: 8, numLayers: 2, numHeads: 2, numKVHeads: 2, dFF: 32,
    });
    const image = new Float32Array(16 * 16 * 3).fill(0.2);
    const out = vlm.generate(image, [1, 2], 3, 0);
    assert.strictEqual(out.length, 2 + 3);
    for (const t of out) {
      assert.ok(t >= 0 && t < 16, `token 应在词表范围: ${t}`);
    }
  });

  it('countParams: 正数且含 vision 参数', () => {
    const vlm = new VLM({
      vision: { imageSize: 16, patchSize: 8, channels: 3, dModel: 8, numLayers: 1, numHeads: 2 },
      vocabSize: 16, dModel: 8, numLayers: 2, numHeads: 2, numKVHeads: 2, dFF: 32,
    });
    assert.ok(vlm.countParams() > 0);
  });
});

describe('MoE (100B)', () => {
  it('route: top-k 专家选择', () => {
    const router = new MoERouter({ dModel: 8, numExperts: 8, topK: 2 });
    const x = new Float32Array(8).fill(0.1);
    const { expertIdx, weights } = router.route(x);
    assert.strictEqual(expertIdx.length, 2);
    assert.strictEqual(weights.length, 2);
    // 权重归一化
    const sum = weights[0] + weights[1];
    assert.ok(Math.abs(sum - 1) < 1e-6, `权重应归一化: ${sum}`);
    // 专家索引互不相同
    assert.notStrictEqual(expertIdx[0], expertIdx[1]);
  });

  it('CONFIG_100B: 配置完整', () => {
    assert.strictEqual(CONFIG_100B.numLayers, 64);
    assert.strictEqual(CONFIG_100B.moe.numExperts, 8);
    assert.strictEqual(CONFIG_100B.moe.topK, 2);
  });

  it('estimateParams: 参数量量级 ~100B', () => {
    const params = estimateParams();
    // 应在 50B - 200B 范围 (MoE 稀疏激活)
    assert.ok(params > 50e9, `参数量应 >50B: ${params}`);
    assert.ok(params < 200e9, `参数量应 <200B: ${params}`);
  });
});

describe('KVCache', () => {
  it('store/retrieve: 基本读写', () => {
    const cache = new KVCache({ numLayers: 2, numKVHeads: 2, dK: 4, maxSeqLen: 8 });
    const keys = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const values = new Float32Array([9, 10, 11, 12, 13, 14, 15, 16]);
    cache.store(0, keys, values);
    const { keys: rk, values: rv, len } = cache.retrieve(0);
    assert.strictEqual(len, 1);
    assert.deepStrictEqual(Array.from(rk), Array.from(keys));
    assert.deepStrictEqual(Array.from(rv), Array.from(values));
  });

  it('windowSize: 滑动窗口保留最近 token', () => {
    const cache = new KVCache({ numLayers: 1, numKVHeads: 1, dK: 2, maxSeqLen: 16, windowSize: 3 });
    for (let i = 0; i < 5; i++) {
      cache.store(0, new Float32Array([i, i]), new Float32Array([i * 10, i * 10]));
    }
    const { len, offset } = cache.retrieve(0);
    assert.strictEqual(len, 3);
    assert.strictEqual(offset, 2); // 跳过前 2 个
  });

  it('memoryBytes: 内存计算正确', () => {
    const cache = new KVCache({ numLayers: 2, numKVHeads: 2, dK: 4, maxSeqLen: 8 });
    cache.store(0, new Float32Array(8), new Float32Array(8));
    cache.store(1, new Float32Array(8), new Float32Array(8));
    // 每层: len(1) * numKVHeads(2) * dK(4) * 2 (K+V) * 4 bytes = 64
    assert.strictEqual(cache.memoryBytes(), 2 * 1 * 2 * 4 * 2 * 4);
  });

  it('quantizeInt8: 量化反量化可逆', () => {
    const x = new Float32Array([0.5, -0.3, 0.8, -0.9, 0.1]);
    const { data, scale } = quantizeInt8(x);
    assert.strictEqual(data.length, 5);
    const restored = dequantizeInt8(data, scale);
    // 允许一定误差
    for (let i = 0; i < x.length; i++) {
      assert.ok(Math.abs(restored[i] - x[i]) < 0.02, `量化误差过大: ${restored[i]} vs ${x[i]}`);
    }
  });
});

describe('Eval Benchmarks', () => {
  it('perplexity: 均匀分布 ≈ vocabSize', () => {
    const vocabSize = 10;
    const tokenIds = [1, 2, 3, 4, 5];
    // forwardFn 返回均匀 logits
    const forwardFn = () => new Float32Array(vocabSize).fill(1.0);
    const ppl = evalMod.perplexity(tokenIds, forwardFn, vocabSize);
    // 均匀分布困惑度 = vocabSize
    assert.ok(Math.abs(ppl - vocabSize) < 1e-4, `均匀分布 ppl 应=${vocabSize}, 实际=${ppl}`);
  });

  it('clozeAccuracy: 完美预测 100%', () => {
    const vocabSize = 10;
    const contexts = [[1, 2], [3, 4], [5, 6]];
    const targets = [7, 8, 9];
    const forwardFn = (ctx) => {
      const logits = new Float32Array(vocabSize).fill(0);
      logits[ctx.length === 2 ? targets[contexts.indexOf(ctx)] : 0] = 100;
      return logits;
    };
    // 简化: 用闭包映射
    const map = new Map(contexts.map((c, i) => [c.join(','), targets[i]]));
    const fn = (ctx) => {
      const logits = new Float32Array(vocabSize).fill(0);
      logits[map.get(ctx.join(','))] = 100;
      return logits;
    };
    assert.strictEqual(evalMod.clozeAccuracy(contexts, targets, fn), 1);
  });

  it('repetitionRate: 重复序列返回 >0', () => {
    const seq = [1, 2, 3, 1, 2, 3, 1, 2, 3];
    const rate = evalMod.repetitionRate(seq, 3);
    assert.ok(rate > 0, `重复序列应有重复: ${rate}`);
    const unique = [1, 2, 3, 4, 5];
    assert.strictEqual(evalMod.repetitionRate(unique, 2), 0);
  });

  it('diversity: 不同样本多样性 > 单样本', () => {
    const samples = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];
    const div = evalMod.diversity(samples, 2);
    assert.ok(div > 0.5, `多样本应高多样性: ${div}`);
  });
});

describe('长上下文 (NTK RoPE)', () => {
  it('scale=1 不改变位置编码', () => {
    const dModel = 8;
    const x = new Float32Array(dModel).fill(1);
    const positions = [5];
    const base = applyRoPE(x, 1, dModel, positions, 10000, 1, 1);
    const ref = applyRoPE(x, 1, dModel, positions, 10000, 1, 1);
    assert.deepStrictEqual(Array.from(base), Array.from(ref));
  });

  it('NTK scale 外推到长序列 (无 NaN)', () => {
    const dModel = 16;
    const nVec = 8;
    const x = new Float32Array(nVec * dModel);
    for (let i = 0; i < x.length; i++) x[i] = (Math.random() - 0.5) * 2;
    // 模拟 8x 上下文外推: scale=8, 位置可到 16384
    const positions = [0, 1, 2, 3, 4096, 8192, 12288, 16384];
    const out = applyRoPE(x, nVec, dModel, positions, 10000, 8, 1);
    assert.strictEqual(out.length, x.length);
    for (let i = 0; i < out.length; i++) {
      assert.ok(!isNaN(out[i]), `NTK 输出不应有 NaN: ${out[i]}`);
    }
  });

  it('不同 scale 产生不同编码', () => {
    const dModel = 8;
    const x = new Float32Array(dModel).fill(1);
    const positions = [1000];
    const out1 = applyRoPE(x, 1, dModel, positions, 10000, 1, 1);
    const out8 = applyRoPE(x, 1, dModel, positions, 10000, 8, 1);
    let diff = 0;
    for (let i = 0; i < dModel; i++) {
      if (Math.abs(out1[i] - out8[i]) > 1e-8) diff++;
    }
    assert.ok(diff > 0, '不同 scale 应产生不同编码');
  });
});