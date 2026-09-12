/**
 * text23d.test.js — Phase 5 Text-to-3D 测试
 *
 * 测试内容:
 *   1. Triplane: 查询 + SDF + 体积渲染
 *   2. SDS: 单步训练 + 梯度信号
 *   3. Text23D: 完整生成管道
 */

const assert = require('assert');
const { describe, it, beforeEach } = require('node:test');

let Triplane = null;
let SDS = null;
let Text23D = null;
let UNet = null;
let generateSchedule = null;

beforeEach(() => {
  ({ Triplane } = require('../src/models/3d/triplane.js'));
  ({ SDS } = require('../src/models/3d/sds.js'));
  ({ Text23D } = require('../src/models/3d/text23d.js'));
  ({ UNet } = require('../src/models/diffusion/unet.js'));
  ({ generateSchedule } = require('../src/models/diffusion/diffusion_schedule.js'));
});

describe('Triplane', () => {
  it('constructor: 参数量正确', () => {
    const tp = new Triplane({ planeSize: 8, latentChannels: 4, mlpHidden: 8, numMLPLayers: 1 });
    // 3 planes * 8*8*4 = 768 + MLP: 4*3*8 + 8 = 112 + 8*4 + 4 = 40 => total ~920
    const params = tp.countParams();
    assert.ok(params > 0, '参数应为正数');
    assert.strictEqual(tp.planeSize, 8);
    assert.strictEqual(tp.latentChannels, 4);
  });

  it('query: 输出维度正确', () => {
    const tp = new Triplane({ planeSize: 8, latentChannels: 4, mlpHidden: 4, numMLPLayers: 1 });
    const feat = tp.query(0, 0, 0);
    assert.strictEqual(feat.length, 4 * 3); // latentCh * 3
  });

  it('forward: 输出 SDF + RGB', () => {
    const tp = new Triplane({ planeSize: 8, latentChannels: 4, mlpHidden: 4, numMLPLayers: 1 });
    const feat = tp.query(0, 0, 0);
    const out = tp.forward(feat);
    assert.strictEqual(out.length, 4); // [sdf, r, g, b]
    // RGB 应在 [0, 1]
    for (let i = 1; i <= 3; i++) {
      assert.ok(out[i] >= 0 && out[i] <= 1, `RGB 应在 [0,1], 实际: ${out[i]}`);
    }
  });

  it('querySDF: 完整查询返回 [sdf, r, g, b]', () => {
    const tp = new Triplane({ planeSize: 8, latentChannels: 4, mlpHidden: 4, numMLPLayers: 1 });
    const out = tp.querySDF(0, 0, 0);
    assert.strictEqual(out.length, 4);
    // 不应有 NaN
    for (let i = 0; i < 4; i++) {
      assert.ok(!isNaN(out[i]), `输出不应有 NaN: ${out[i]}`);
    }
  });

  it('volumeRender: 渲染结果有效', () => {
    const tp = new Triplane({ planeSize: 8, latentChannels: 4, mlpHidden: 4, numMLPLayers: 1 });
    const result = tp.volumeRender(new Float32Array([0, 0, 3]), new Float32Array([0, 0, -1]), 8);
    assert.ok(result.color.length === 3);
    assert.ok(!isNaN(result.color[0]), 'color 不应有 NaN');
    assert.ok(!isNaN(result.depth), 'depth 不应有 NaN');
    // RGB 应在 [0, 1]
    for (let i = 0; i < 3; i++) {
      assert.ok(result.color[i] >= 0 && result.color[i] <= 1,
        `color[${i}] 应在 [0,1], 实际: ${result.color[i]}`);
    }
  });

  it('renderCamera: 输出图像尺寸正确', () => {
    const tp = new Triplane({ planeSize: 8, latentChannels: 4, mlpHidden: 4, numMLPLayers: 1 });
    const img = tp.renderCamera(4, 4, Math.PI / 4, 3, 4);
    assert.strictEqual(img.length, 4 * 4 * 3);
    let nanCount = 0;
    for (let i = 0; i < img.length; i++) if (isNaN(img[i])) nanCount++;
    assert.strictEqual(nanCount, 0, '图像不应有 NaN');
  });

  it('getIsoSurface2D: 返回二值化等值面', () => {
    const tp = new Triplane({ planeSize: 8, latentChannels: 4, mlpHidden: 4, numMLPLayers: 1 });
    const surf = tp.getIsoSurface2D(0, 8);
    assert.strictEqual(surf.length, 8 * 8);
    // 只应包含 0 和 1
    for (let i = 0; i < surf.length; i++) {
      assert.ok(surf[i] === 0 || surf[i] === 1, '等值面应二值化');
    }
  });
});

describe('SDS', () => {
  it('constructor: 默认配置正确', () => {
    const sds = new SDS({});
    assert.strictEqual(sds.tMin, 50);
    assert.strictEqual(sds.tMax, 1000);
    assert.strictEqual(sds.imgSize, 32);
  });

  it('_weight: 权重函数返回值正确', () => {
    const sds = new SDS({});
    const sched = generateSchedule('cosine', 1000, { s: 0.008 });
    const w1 = sds._weight(10, sched.alpha, sched.alphaBar);
    const w2 = sds._weight(500, sched.alpha, sched.alphaBar);
    const w3 = sds._weight(999, sched.alpha, sched.alphaBar);
    assert.ok(w1 > 0, '权重应为正');
    assert.ok(w2 > 0, '权重应为正');
    assert.ok(w1 !== w2, '不同时间步权重应不同');
  });

  it('step: 返回正确字段', () => {
    const tp = new Triplane({ planeSize: 4, latentChannels: 2, mlpHidden: 2, numMLPLayers: 1 });
    const sds = new SDS({ imgSize: 4, numSamples: 4 });
    const sched = generateSchedule('cosine', 1000, { s: 0.008 });

    const textEmb = new Float32Array(16 * 8);
    for (let i = 0; i < textEmb.length; i++) textEmb[i] = (Math.random() - 0.5) * 0.1;

    const epsilon = new Float32Array(4 * 4 * 3);
    for (let i = 0; i < epsilon.length; i++) epsilon[i] = (Math.random() - 0.5) * 0.1;

    const result = sds.step(tp, 500, textEmb, sched.alpha, sched.alphaBar, [0, 0], epsilon);
    assert.ok(result.zt, '应返回 zt');
    assert.ok(result.epsilon, '应返回 epsilon');
    assert.ok(result.x0, '应返回 x0');
    assert.ok(!isNaN(result.sqrtAlphaBar), 'sqrtAlphaBar 不应为 NaN');
  });
});

describe('Text23D', () => {
  it('constructor: 创建成功', () => {
    const t23d = new Text23D({
      unetConfig: {
        latentChannels: 3, latentSize: 8,
        textEmbedDim: 8, textTokenLen: 8,
        blockChannels: [16, 32], numHeads: 1,
      },
      triplaneConfig: { planeSize: 4, latentChannels: 4, mlpHidden: 4, numMLPLayers: 1 },
      sdsConfig: { tMin: 50, tMax: 200, imgSize: 4, numSamples: 4, fov: Math.PI / 4 },
    });
    assert.ok(t23d.unet);
    assert.ok(t23d.schedule);
  });

  it('encodeText: 输出维度正确', () => {
    const t23d = new Text23D({
      unetConfig: { latentChannels: 3, latentSize: 8, textEmbedDim: 8, textTokenLen: 8,
        blockChannels: [16, 32], numHeads: 1 },
      triplaneConfig: { planeSize: 4, latentChannels: 4 },
      sdsConfig: { imgSize: 4, numSamples: 4 },
    });
    const emb = t23d.encodeText('cat', 8, 8);
    assert.strictEqual(emb.length, 8 * 8);
  });

  it('render: 输出图像尺寸正确', () => {
    const t23d = new Text23D({
      unetConfig: { latentChannels: 3, latentSize: 8, textEmbedDim: 8, textTokenLen: 8,
        blockChannels: [16, 32], numHeads: 1 },
      triplaneConfig: { planeSize: 4, latentChannels: 4, mlpHidden: 4 },
      sdsConfig: { imgSize: 4, numSamples: 4, fov: Math.PI / 4 },
    });
    const tp = new Triplane({ planeSize: 4, latentChannels: 4, mlpHidden: 4 });
    const img = t23d.render(tp, 4, 4);
    assert.strictEqual(img.length, 4 * 4 * 3);
  });

  it('generate: 完整生成管道', () => {
    const t23d = new Text23D({
      unetConfig: {
        latentChannels: 3, latentSize: 4,
        textEmbedDim: 8, textTokenLen: 8,
        blockChannels: [16], numHeads: 1,
      },
      triplaneConfig: { planeSize: 4, latentChannels: 4, mlpHidden: 4, numMLPLayers: 1 },
      sdsConfig: { tMin: 50, tMax: 200, imgSize: 4, numSamples: 4, fov: Math.PI / 4 },
    });

    let lastLoss = 0;
    const result = t23d.generate('a sphere', 3, 0.01, 1, (step, loss) => {
      lastLoss = loss;
    });

    assert.ok(result.triplane, '应返回 triplane');
    assert.ok(!isNaN(lastLoss), 'loss 不应为 NaN');
  });
});