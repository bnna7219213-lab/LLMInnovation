/**
 * diffusion.test.js — Phase 4 Diffusion 测试
 *
 * 测试内容:
 *   1. Diffusion Schedule: 公式正确性 + 边界条件
 *   2. VAE: 编码-解码可逆 + 重参数化
 *   3. UNet: 维度正确 + 时间步嵌入
 *   4. Text2Image: 完整生成管道
 */

const assert = require('assert');
const { describe, it, beforeEach } = require('node:test');
const { generateSchedule, addNoise, reverseStep } = require('../src/models/diffusion/diffusion_schedule.js');
const { VAE } = require('../src/models/diffusion/vae.js');
const { UNet } = require('../src/models/diffusion/unet.js');
const { Text2Image } = require('../src/models/diffusion/text2image.js');

// ============ Diffusion Schedule ============

describe('DiffusionSchedule', () => {
  it('cosine 调度: alphaBar 从接近 1 衰减到接近 0', () => {
    const { alphaBar } = generateSchedule('cosine', 1000);
    assert.ok(alphaBar[0] > 0.99, `alphaBar[0]=${alphaBar[0]} 应接近 1`);
    assert.ok(alphaBar[999] < 0.01, `alphaBar[999]=${alphaBar[999]} 应接近 0`);
    // 单调递减
    for (let t = 1; t < 1000; t++) {
      assert.ok(alphaBar[t] <= alphaBar[t - 1], `alphaBar 在 t=${t} 应递减`);
    }
  });

  it('linear 调度: beta 线性增长', () => {
    const { betas, alphaBar } = generateSchedule('linear', 1000, { betaMin: 0.0001, betaMax: 0.02 });
    // beta 应线性增长
    assert.ok(betas[0] > 0.00005 && betas[0] < 0.001, `betas[0]=${betas[0]}`);
    assert.ok(betas[999] > 0.01 && betas[999] < 0.03, `betas[999]=${betas[999]}`);
    // alphaBar 从 1 衰减到接近 0
    assert.ok(alphaBar[0] > 0.99);
    assert.ok(alphaBar[999] < 1, `alphaBar[999]=${alphaBar[999]}`);
  });

  it('cosine 调度的 beta 从 alphaBar 反推正确', () => {
    const { betas, alpha, alphaBar } = generateSchedule('cosine', 1000);
    // α_t = 1 - β_t
    for (let t = 0; t < 1000; t++) {
      assert.ok(Math.abs(alpha[t] - (1 - betas[t])) < 1e-6,
        `α_t=${alpha[t]} ≠ 1-β_t=${1 - betas[t]}`);
    }
    // ᾱ_t = α_1 * α_2 * ... * α_t
    let prod = 1;
    for (let t = 0; t < 100; t++) {
      prod *= alpha[t];
      assert.ok(Math.abs(prod - alphaBar[t]) < 1e-4,
        `ᾱ_${t}=${alphaBar[t]} ≠ prod=${prod}`);
    }
  });

  it('addNoise: 公式 x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε', () => {
    const { alphaBar } = generateSchedule('cosine', 1000);
    const nPixels = 32;
    const x0 = new Float32Array(nPixels);
    for (let i = 0; i < nPixels; i++) x0[i] = 1.0; // 全 1
    const noise = new Float32Array(nPixels);
    for (let i = 0; i < nPixels; i++) noise[i] = 0.5; // 全 0.5
    const tArray = [500];

    const xt = addNoise(x0, tArray, noise, alphaBar); // 返回 Float32Array[]
    const sqrtAb = Math.sqrt(alphaBar[500]);
    const sqrtOneMinusAb = Math.sqrt(1 - alphaBar[500]);
    const expected = sqrtAb * 1.0 + sqrtOneMinusAb * 0.5;

    assert.ok(Math.abs(xt[0][0] - expected) < 1e-6,
      `xt[0][0]=${xt[0][0]} ≠ expected=${expected}`);
  });

  it('reverseStep: 在 t=0 时无新噪声', () => {
    const { alpha, alphaBar, sigma } = generateSchedule('cosine', 1000);
    const nPixels = 16;
    const x_t = new Float32Array(nPixels).fill(1);
    const epsilonPred = new Float32Array(nPixels).fill(0);
    const z = new Float32Array(nPixels).fill(0); // t=0 时无新噪声

    const x_tm1 = reverseStep(x_t, 0, epsilonPred, alpha, alphaBar, sigma, z, nPixels);
    // 当 ε=0, z=0 时, x_{t-1} 应稳定
    for (let i = 0; i < nPixels; i++) {
      assert.ok(!isNaN(x_tm1[i]), `x_tm1[${i}]=NaN`);
      assert.ok(isFinite(x_tm1[i]), `x_tm1[${i}]=无穷`);
    }
  });
});

// ============ VAE ============

describe('VAE', () => {
  let vae;
  beforeEach(() => {
    vae = new VAE({
      inputChannels: 3,
      latentChannels: 4,
      inputSize: 16,
      latentSize: 4,
      blockChannels: [16, 32],
    });
  });

  it('构造器: 配置正确', () => {
    assert.strictEqual(vae.inputChannels, 3);
    assert.strictEqual(vae.latentChannels, 4);
    assert.strictEqual(vae.inputSize, 16);
    assert.strictEqual(vae.latentSize, 4);
  });

  it('encode: 输出维度正确', () => {
    const batch = 1;
    const nPixels = vae.inputSize * vae.inputSize * vae.inputChannels;
    const x = new Float32Array(nPixels);
    for (let i = 0; i < nPixels; i++) x[i] = 0.1;

    const { mu, sigma } = vae.encode(x, batch);
    const expectedLen = vae.latentSize * vae.latentSize * vae.latentChannels;
    assert.strictEqual(mu.length, expectedLen);
    assert.strictEqual(sigma.length, expectedLen);
  });

  it('reparameterize: z = μ + σ * ε', () => {
    const mu = new Float32Array([1, 2, 3]);
    const sigma = new Float32Array([0.5, 0.5, 0.5]);
    const eps = new Float32Array([1, 1, 1]);

    const z = vae.reparameterize(mu, sigma, eps);
    assert.strictEqual(z[0], 1 + 0.5 * 1); // 1.5
    assert.strictEqual(z[1], 2 + 0.5 * 1); // 2.5
    assert.strictEqual(z[2], 3 + 0.5 * 1); // 3.5
  });

  it('getLatentShape: 正确', () => {
    const shape = vae.getLatentShape();
    assert.deepStrictEqual(shape, [4, 4, 4]);
  });
});

// ============ UNet ============

describe('UNet', () => {
  let unet;
  beforeEach(() => {
    unet = new UNet({
      latentChannels: 4,
      latentSize: 8,
      textEmbedDim: 32,
      textTokenLen: 8,
      blockChannels: [16, 32],
      numHeads: 2,
    });
  });

  it('构造器: 配置正确', () => {
    assert.strictEqual(unet.latentChannels, 4);
    assert.strictEqual(unet.latentSize, 8);
    assert.strictEqual(unet.textEmbedDim, 32);
    assert.strictEqual(unet.textTokenLen, 8);
  });

  it('forward: 输出维度正确', () => {
    const batch = 1;
    const nPixels = unet.latentSize * unet.latentSize * unet.latentChannels;
    const x = new Float32Array(nPixels);
    for (let i = 0; i < nPixels; i++) x[i] = 0.1;
    const t = [500];
    const textEmb = new Float32Array(unet.textTokenLen * unet.textEmbedDim);
    for (let i = 0; i < textEmb.length; i++) textEmb[i] = 0.01;

    const out = unet.forward([x], t, [textEmb], batch);
    assert.strictEqual(out[0].length, nPixels);
  });

  it('forward: 确定性 (相同输入相同输出)', () => {
    const nPixels = unet.latentSize * unet.latentSize * unet.latentChannels;
    const x = new Float32Array(nPixels);
    for (let i = 0; i < nPixels; i++) x[i] = 0.05;
    const t = [100];
    const textEmb = new Float32Array(unet.textTokenLen * unet.textEmbedDim).fill(0.01);

    const out1 = unet.forward([x], t, [textEmb], 1)[0];
    const out2 = unet.forward([x], t, [textEmb], 1)[0];
    for (let i = 0; i < out1.length; i++) {
      assert.strictEqual(out1[i], out2[i]);
    }
  });

  it('timeEmbed: 正弦位置编码, 维度正确', () => {
    const emb = unet._timeEmbed(100, unet.timeEmbedDim);
    assert.strictEqual(emb.length, unet.timeEmbedDim);
    for (let i = 0; i < emb.length; i++) {
      assert.ok(emb[i] >= -1 && emb[i] <= 1, `emb[${i}]=${emb[i]} 应在 [-1,1]`);
    }
  });
});

// ============ Text2Image ============

describe('Text2Image', () => {
  let t2i;
  beforeEach(() => {
    t2i = new Text2Image({
      T: 1000,
      schedule: 'cosine',
      textEmbedDim: 32,
      textTokenLen: 8,
      vaeConfig: {
        inputChannels: 3,
        latentChannels: 4,
        inputSize: 16,
        latentSize: 4,
        blockChannels: [16, 32],
      },
      numHeads: 2,
    });
  });

  it('encodeText: 输出维度正确', () => {
    const emb = t2i.encodeText('hello world', 8, 32);
    assert.strictEqual(emb.length, 8 * 32);
  });

  it('generate: 输出维度正确 [inputSize, inputSize, 3]', () => {
    // 用少量步数测试 (5 步)
    const image = t2i.generate('a cat', 42, 5);
    const expectedLen = 16 * 16 * 3;
    assert.strictEqual(image.length, expectedLen);
  });

  it('generate: 确定性 (相同 seed 相同输出)', () => {
    const img1 = t2i.generate('a cat', 42, 5);
    const img2 = t2i.generate('a cat', 42, 5);
    for (let i = 0; i < img1.length; i++) {
      assert.strictEqual(img1[i], img2[i]);
    }
  });

  it('generate: 不同 seed 不同输出', () => {
    const img1 = t2i.generate('a cat', 1, 5);
    const img2 = t2i.generate('a cat', 1000000, 5); // 远不同的 seed
    let diff = 0;
    for (let i = 0; i < img1.length; i++) {
      if (Math.abs(img1[i] - img2[i]) > 1e-8) diff++;
    }
    assert.ok(diff > 0, `不同 seed 应产生不同输出, 实际 diff=${diff}`);
  });
});