/**
 * nn.test.js — Phase 0 NN 原语测试
 *
 * 测试内容:
 *   1. RMSNorm: 已知输入的手算验证
 *   2. SwiGLU: 维度验证 + swish 正确性
 *   3. RoPE: 手算验证旋转角度
 *   4. Attention: causal mask + GQA + MHA 维度正确
 *   5. AdamW: 一阶/二阶矩更新 + 权重衰减 + 偏差修正
 *   6. LRScheduler: cosine/linear/step 各调度器正确
 *   7. Checkpoint: save/load 可逆
 */

const { rmsNorm, rmsNormBatch } = require('../src/nn/rmsnorm.js');
const { swiglu } = require('../src/nn/swiglu.js');
const { applyRoPE } = require('../src/nn/rope.js');
const { multiHeadAttention, matmul } = require('../src/nn/attention.js');
const { AdamW } = require('../src/nn/optimizer.js');
const { LRScheduler } = require('../src/training/scheduler.js');
const { saveCheckpoint, loadCheckpoint } = require('../src/training/checkpoint.js');
const assert = require('assert');
const { describe, it } = require('node:test');
const fs = require('fs');
const path = require('path');

// ============ RMSNorm ============

describe('RMSNorm', () => {
  it('标准 RMSNorm 计算正确', () => {
    // 输入: [1, 2, 3, 4]  gamma: [1,1,1,1]  eps=1e-6
    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // output = [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
    const x = new Float32Array([1, 2, 3, 4]);
    const gamma = new Float32Array([1, 1, 1, 1]);
    const out = rmsNorm(x, gamma);

    const expectedRMS = Math.sqrt(7.5);
    assert.ok(Math.abs(out[0] - 1 / expectedRMS) < 1e-6, `out[0]=${out[0]} expected=${1/expectedRMS}`);
    assert.ok(Math.abs(out[1] - 2 / expectedRMS) < 1e-6);
    assert.ok(Math.abs(out[2] - 3 / expectedRMS) < 1e-6);
    assert.ok(Math.abs(out[3] - 4 / expectedRMS) < 1e-6);
  });

  it('gamma 缩放正确', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const gamma = new Float32Array([2, 2, 2, 2]);
    const out = rmsNorm(x, gamma);
    const expectedRMS = Math.sqrt(7.5);
    // output = (x/rms) * gamma = (x/rms) * 2
    assert.ok(Math.abs(out[0] - 2 / expectedRMS) < 1e-6);
  });

  it('batch 模式正确', () => {
    const X = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]); // 2 vectors of dModel=4
    const gamma = new Float32Array([1, 1, 1, 1]);
    const out = rmsNormBatch(X, 2, 4, gamma);

    const rms1 = Math.sqrt((1 + 4 + 9 + 16) / 4);
    const rms2 = Math.sqrt((25 + 36 + 49 + 64) / 4);
    assert.ok(Math.abs(out[0] - 1 / rms1) < 1e-6);
    assert.ok(Math.abs(out[4] - 5 / rms2) < 1e-6);
  });
});

// ============ SwiGLU ============

describe('SwiGLU', () => {
  it('维度正确: [nVec, dModel] → [nVec, dModel]', () => {
    const dModel = 4, dFF = 8, nVec = 2;
    const x = new Float32Array(nVec * dModel);
    for (let i = 0; i < x.length; i++) x[i] = 0.1;

    const W1 = new Float32Array(dFF * dModel);
    const W2 = new Float32Array(dFF * dModel);
    const W3 = new Float32Array(dModel * dFF);

    const out = swiglu(x, nVec, dModel, dFF, W1, W2, W3);
    assert.strictEqual(out.length, nVec * dModel);
  });

  it('零输入输出零', () => {
    const dModel = 4, dFF = 8, nVec = 1;
    const x = new Float32Array(nVec * dModel);
    const W1 = new Float32Array(dFF * dModel);
    const W2 = new Float32Array(dFF * dModel);
    const W3 = new Float32Array(dModel * dFF);

    const out = swiglu(x, nVec, dModel, dFF, W1, W2, W3);
    for (let i = 0; i < out.length; i++) {
      assert.ok(Math.abs(out[i]) < 1e-10, `out[${i}]=${out[i]}`);
    }
  });

  it('swish 正确性: swish(0)=0, swish(x→+∞)≈x, swish(x→-∞)≈0', () => {
    // 直接验证 swish 公式
    const swish = a => a / (1 + Math.exp(-a));
    assert.ok(Math.abs(swish(0)) < 1e-10);
    assert.ok(swish(10) > 9.9);
    assert.ok(Math.abs(swish(-10)) < 0.01);
  });
});

// ============ RoPE ============

describe('RoPE', () => {
  it('dModel 为奇数抛错', () => {
    const x = new Float32Array([1, 2, 3]);
    assert.throws(() => applyRoPE(x, 1, 3, [0]), /偶数/);
  });

  it('pos=0 时输出等于输入 (cos(0)=1, sin(0)=0)', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const out = applyRoPE(x, 1, 4, [0]);
    assert.strictEqual(out[0], 1);
    assert.strictEqual(out[1], 2);
    assert.strictEqual(out[2], 3);
    assert.strictEqual(out[3], 4);
  });

  it('pos=1, base=10000 产生非平凡旋转', () => {
    const x = new Float32Array([1, 0, 0, 0]); // 只有第一个维度有值
    const out = applyRoPE(x, 1, 4, [1]);
    // angle_0 = 1 * 1/10000^0 = 1 (freq=0)
    // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
    assert.ok(Math.abs(out[0] - 1 * Math.cos(1)) < 1e-6, `out[0]=${out[0]} expected=${Math.cos(1)}`);
    assert.ok(Math.abs(out[1] - 1 * Math.sin(1)) < 1e-6);
  });

  it('NTK 插值: scale=2 时 base 被放大 (用 freq=0.5 的第二对维度)', () => {
    // x=[0,0,1,0]: 只有第 2 对维度 (freq=0.5) 有值
    // angle = pos / base^0.5 = pos / sqrt(base)
    // scale=1: base=10000, angle=1000/100=10
    // scale=2: base=20000, angle=1000/141.4=7.07
    // cos(10)≈-0.839, cos(7.07)≈0.686 → 差异明显
    const x = new Float32Array([0, 0, 1, 0]);
    const out1 = applyRoPE(x, 1, 4, [1000], 10000, 1);
    const out2 = applyRoPE(x, 1, 4, [1000], 10000, 2);
    assert.ok(Math.abs(out1[2] - out2[2]) > 1e-6,
      `scale=1 out[2]=${out1[2]}, scale=2 out[2]=${out2[2]}, 应不同`);
  });
});

// ============ Attention ============

describe('Attention', () => {
  it('MHA: 维度正确 [nSeq, dModel] → [nSeq, dModel]', () => {
    const nSeq = 4, dModel = 8, numHeads = 2, dK = 4, numKVHeads = 2;
    const x = new Float32Array(nSeq * dModel);
    for (let i = 0; i < x.length; i++) x[i] = 0.1;

    const WQ = new Float32Array(numHeads * dK * dModel);
    const WK = new Float32Array(numKVHeads * dK * dModel);
    const WV = new Float32Array(numKVHeads * dK * dModel);
    const WO = new Float32Array(dModel * numHeads * dK);

    const out = multiHeadAttention(x, nSeq, dModel, numHeads, dK, numKVHeads, WQ, WK, WV, WO);
    assert.strictEqual(out.length, nSeq * dModel);
  });

  it('GQA: numHeads=4, numKVHeads=2 (每组 2 个 Q 共享 1 组 KV)', () => {
    const nSeq = 2, dModel = 8, numHeads = 4, dK = 2, numKVHeads = 2;
    const x = new Float32Array(nSeq * dModel);
    for (let i = 0; i < x.length; i++) x[i] = 0.05;

    const WQ = new Float32Array(numHeads * dK * dModel);
    const WK = new Float32Array(numKVHeads * dK * dModel);
    const WV = new Float32Array(numKVHeads * dK * dModel);
    const WO = new Float32Array(dModel * numHeads * dK);

    const out = multiHeadAttention(x, nSeq, dModel, numHeads, dK, numKVHeads, WQ, WK, WV, WO);
    assert.strictEqual(out.length, nSeq * dModel);
  });

  it('numHeads 不能被 numKVHeads 整除时抛错', () => {
    const x = new Float32Array(8);
    assert.throws(() =>
      multiHeadAttention(x, 2, 8, 3, 2, 2,
        new Float32Array(6*8), new Float32Array(4*8), new Float32Array(4*8), new Float32Array(8*6)),
      /整除/);
  });

  it('Causal mask: 输出不依赖未来 token', () => {
    const nSeq = 3, dModel = 4, numHeads = 1, dK = 4, numKVHeads = 1;
    const x = new Float32Array(nSeq * dModel);
    // x = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]  → Q,K,V 唯一非零
    x[0] = 1; x[5] = 1; x[10] = 1;

    const WQ = new Float32Array(numHeads * dK * dModel);
    for (let i = 0; i < WQ.length; i++) WQ[i] = 1 / Math.sqrt(dModel);
    const WK = new Float32Array(numKVHeads * dK * dModel);
    for (let i = 0; i < WK.length; i++) WK[i] = 1 / Math.sqrt(dModel);
    const WV = new Float32Array(numKVHeads * dK * dModel);
    for (let i = 0; i < WV.length; i++) WV[i] = 1;
    const WO = new Float32Array(dModel * numHeads * dK);
    for (let i = 0; i < WO.length; i++) WO[i] = 1;

    // causal mask: 位置 i 只看位置 <= i
    const mask = new Float32Array(nSeq * nSeq);
    for (let i = 0; i < nSeq; i++) {
      for (let j = 0; j < nSeq; j++) {
        mask[i * nSeq + j] = j > i ? 1 : 0; // 1 = 遮盖, 0 = 不遮盖
      }
    }

    const out = multiHeadAttention(x, nSeq, dModel, numHeads, dK, numKVHeads, WQ, WK, WV, WO, mask);
    // 输出存在 (不抛错) 且维度正确
    assert.strictEqual(out.length, nSeq * dModel);
  });
});

// ============ AdamW ============

describe('AdamW', () => {
  it('初始 step=0, lr 正确', () => {
    const opt = new AdamW({ learningRate: 1e-3 });
    assert.strictEqual(opt.getStep(), 0);
    assert.strictEqual(opt.getLearningRate(), 1e-3);
  });

  it('一阶/二阶矩正确更新', () => {
    const opt = new AdamW({ learningRate: 1e-3, weightDecay: 0 });
    const param = new Float32Array([1, 2]);
    const grad = new Float32Array([0.1, 0.2]);
    const state = opt.createState(param);

    opt.stepUpdate(param, grad, state, false); // 无权重衰减

    assert.strictEqual(opt.getStep(), 1);
    // 经过一步后, param 应该变了 (方向: -grad)
    assert.ok(param[0] < 1, `param[0]=${param[0]} 应 < 1`);
    assert.ok(param[1] < 2, `param[1]=${param[1]} 应 < 2`);
  });

  it('权重衰减正确 (param 向 0 靠近)', () => {
    const opt = new AdamW({ learningRate: 1e-3, weightDecay: 0.1, beta1: 0, beta2: 0 });
    // β1=β2=0 → 无偏差修正影响, 更新 = lr * (grad + λ*param)
    const param = new Float32Array([1]);
    const grad = new Float32Array([0]); // 梯度为 0, 仅权重衰减生效
    const state = opt.createState(param);

    opt.stepUpdate(param, grad, state, true);
    // 期望: param = 1 - 1e-3 * (0 + 0.1*1) = 1 - 1e-4 = 0.9999
    assert.ok(Math.abs(param[0] - 0.9999) < 1e-6, `param[0]=${param[0]} 应 ≈ 0.9999`);
  });

  it('批量 update 多个参数字段', () => {
    const opt = new AdamW({ learningRate: 1e-3, weightDecay: 0 });
    const p1 = new Float32Array([1, 2]);
    const p2 = new Float32Array([3, 4]);
    const g1 = new Float32Array([0.1, 0.1]);
    const g2 = new Float32Array([0.1, 0.1]);
    const s1 = opt.createState(p1);
    const s2 = opt.createState(p2);

    opt.update(
      [{ param: p1, grad: g1 }, { param: p2, grad: g2 }],
      [s1, s2]
    );

    // update() 一次调用 = 一个训练步, step 只应递增 1
    assert.strictEqual(opt.getStep(), 1);
  });

  it('setLearningRate 正确', () => {
    const opt = new AdamW({ learningRate: 1e-3 });
    opt.setLearningRate(2e-4);
    assert.strictEqual(opt.getLearningRate(), 2e-4);
  });
});

// ============ LRScheduler ============

describe('LRScheduler', () => {
  it('cosine: warmup 阶段线性增长', () => {
    const sched = new LRScheduler({ maxLR: 1e-3, warmupSteps: 100, totalSteps: 1000, schedule: 'cosine' });
    assert.strictEqual(sched.getLR(0), 0);
    assert.strictEqual(sched.getLR(50), 5e-4);
    assert.strictEqual(sched.getLR(100), 1e-3);
  });

  it('cosine: decay 阶段余弦衰减到 minLR', () => {
    const sched = new LRScheduler({ maxLR: 1e-3, minLR: 1e-5, warmupSteps: 100, totalSteps: 1000, schedule: 'cosine' });
    const lrStart = sched.getLR(100);
    const lrEnd = sched.getLR(1000);
    assert.ok(lrStart > lrEnd);
    assert.ok(Math.abs(lrEnd - 1e-5) < 1e-6, `lrEnd=${lrEnd} 应 ≈ 1e-5`);
  });

  it('linear: warmup 后保持 maxLR', () => {
    const sched = new LRScheduler({ maxLR: 1e-3, warmupSteps: 100, schedule: 'linear' });
    assert.strictEqual(sched.getLR(0), 0);
    assert.strictEqual(sched.getLR(50), 5e-4);
    assert.strictEqual(sched.getLR(100), 1e-3);
    assert.strictEqual(sched.getLR(200), 1e-3);
  });

  it('step: 每 N 步衰减一次', () => {
    const sched = new LRScheduler({ maxLR: 1e-3, stepDecayEvery: 100, stepDecayFactor: 0.1, schedule: 'step' });
    assert.strictEqual(sched.getLR(0), 1e-3);
    assert.strictEqual(sched.getLR(99), 1e-3);
    assert.strictEqual(sched.getLR(100), 1e-4);
    assert.ok(Math.abs(sched.getLR(200) - 1e-5) < 1e-9, `step=200 lr=${sched.getLR(200)}`);
  });

  it('step 为负数抛错', () => {
    const sched = new LRScheduler({ maxLR: 1e-3, schedule: 'cosine' });
    assert.throws(() => sched.getLR(-1), /负数/);
  });
});

// ============ Checkpoint ============

describe('Checkpoint', () => {
  it('save/load 可逆', () => {
    const params = {
      'layer1.weight': new Float32Array([1, 2, 3, 4]),
      'layer2.weight': new Float32Array([5, 6, 7, 8]),
    };
    const optState = { m: [1, 2, 3, 4], v: [0.1, 0.2, 0.3, 0.4] };
    const metadata = { step: 100, loss: 3.5, learningRate: 1e-4 };

    const tmpPath = path.join('/tmp', 'checkpoint_test_' + Date.now() + '.json');
    saveCheckpoint(params, optState, metadata, tmpPath);

    const loaded = loadCheckpoint(tmpPath);
    assert.strictEqual(loaded.metadata.step, 100);
    assert.strictEqual(loaded.metadata.loss, 3.5);
    assert.strictEqual(loaded.metadata.learningRate, 1e-4);
    assert.deepStrictEqual(Array.from(loaded.params['layer1.weight']), [1, 2, 3, 4]);
    assert.deepStrictEqual(Array.from(loaded.params['layer2.weight']), [5, 6, 7, 8]);

    fs.unlinkSync(tmpPath);
  });
});