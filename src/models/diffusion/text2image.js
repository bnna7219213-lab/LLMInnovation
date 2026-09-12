/**
 * text2image.js — Text-to-Image 生成管道
 *
 * 流程:
 *   1. 文本编码: text → textEmbedding (LLM 或 CLIP)
 *   2. 噪声初始化: z_T = N(0, I)
 *   3. 反向扩散: for t = T-1 downto 0:
 *      ε_θ = UNet(z_t, t, textEmbedding)
 *      z_{t-1} = reverseStep(z_t, t, ε_θ, ...)
 *   4. VAE 解码: z_0 → image
 *   5. 后处理: [-1,1] → [0,255]
 *
 * 支持调度:
 *   - DDPM (1000 步, 高质量)
 *   - DDIM (50 步, 中等质量, 更快)
 *   - LCM (4-8 步, 快速推理)
 */

const { VAE } = require('./vae.js');
const { UNet } = require('./unet.js');
const { generateSchedule, addNoise, reverseStep, ddimStep } = require('./diffusion_schedule.js');

class Text2Image {
  constructor(config) {
    this.config = config || {};
    this.T = this.config.T || 1000;

    // 初始化组件
    const vaeConfig = this.config.vaeConfig || {
      inputChannels: 3,
      latentChannels: 4,
      inputSize: 64,
      latentSize: 8,
      blockChannels: [32, 64, 128],
    };

    const unetConfig = {
      latentChannels: vaeConfig.latentChannels,
      latentSize: vaeConfig.latentSize,
      textEmbedDim: this.config.textEmbedDim || 64,
      textTokenLen: this.config.textTokenLen || 16,
      blockChannels: vaeConfig.blockChannels,
      numHeads: this.config.numHeads || 4,
    };

    this.vae = new VAE(vaeConfig);
    this.unet = new UNet(unetConfig);
    this.schedule = generateSchedule(this.config.schedule || 'cosine', this.T,
      { betaMin: 0.0001, betaMax: 0.02, s: 0.008 });
  }

  /**
   * 文本编码: 简单哈希编码 (实际应使用 LLM/CLIP)
   * @param {string} prompt
   * @param {number} tokenLen
   * @param {number} embedDim
   * @returns {Float32Array} [tokenLen, embedDim]
   */
  encodeText(prompt, tokenLen = 16, embedDim = 64) {
    const emb = new Float32Array(tokenLen * embedDim);
    // 简单的字符级编码: 每个字符的 ASCII 映射到嵌入
    for (let i = 0; i < tokenLen && i < prompt.length; i++) {
      const ch = prompt.charCodeAt(i);
      for (let j = 0; j < embedDim; j++) {
        // 使用余弦编码
        emb[i * embedDim + j] = Math.cos(ch / embedDim * (j + 1));
      }
    }
    return emb;
  }

  /**
   * 高斯噪声生成
   */
  /**
   * DDPM 采样
   * @param {string} prompt
   * @param {number} seed
   * @param {number} numSteps - 采样步数
   * @returns {Float32Array} [inputSize, inputSize, 3]
   */
  generate(prompt, seed = 42, numSteps = null) {
    // 设置随机种子
    if (seed !== null) {
      this._setSeed(seed);
    }

    const latentShape = this.vae.getLatentShape();
    const nPixels = latentShape[0] * latentShape[1] * latentShape[2];
    const textEmb = this.encodeText(prompt, this.config.textTokenLen || 16,
      this.config.textEmbedDim || 64);

    // 初始噪声
    let z = this._randn(nPixels);

    // 采样步数
    if (!numSteps) numSteps = this.T;
    const step = Math.floor(this.T / numSteps);
    const timesteps = [];
    for (let t = this.T - 1; t >= 0; t -= step) {
      timesteps.push(t);
    }
    // 确保包含 t=0
    if (!timesteps.includes(0)) timesteps.push(0);

    for (let i = 0; i < timesteps.length; i++) {
      const t = timesteps[i];
      if (t < 0) continue;

      // UNet 预测噪声
      const zArr = [z]; // batch=1
      const tArr = [t];
      const textArr = [textEmb];
      const epsilonPred = this.unet.forward(zArr, tArr, textArr, 1)[0];

      if (i === timesteps.length - 1) {
        // 最后一步: t=0, 无新噪声
        const zNew = reverseStep(z, t, epsilonPred,
          this.schedule.alpha, this.schedule.alphaBar, this.schedule.sigma,
          new Float32Array(nPixels), nPixels);
        z = zNew;
      } else {
        const zNew = reverseStep(z, t, epsilonPred,
          this.schedule.alpha, this.schedule.alphaBar, this.schedule.sigma,
          this._randn(nPixels), nPixels);
        z = zNew;
      }
    }

    // VAE 解码 (z 已是 Float32Array[batch*nPixels], batch=1)
    const decoded = this.vae.decode(z, 1);

    return decoded;
  }

  /**
   * DDIM 采样 (更快, 确定性)
   */
  generateDDIM(prompt, seed = 42, numSteps = 50) {
    if (seed !== null) this._setSeed(seed);

    const latentShape = this.vae.getLatentShape();
    const nPixels = latentShape[0] * latentShape[1] * latentShape[2];
    const textEmb = this.encodeText(prompt, this.config.textTokenLen || 16,
      this.config.textEmbedDim || 64);

    let z = this._randn(nPixels);
    const step = Math.floor(this.T / numSteps);
    const timesteps = [];
    for (let t = this.T - 1; t >= 0; t -= step) {
      timesteps.push(t);
    }
    if (!timesteps.includes(0)) timesteps.push(0);

    for (let i = 0; i < timesteps.length - 1; i++) {
      const t = timesteps[i];
      const tPrev = timesteps[i + 1];

      const epsilonPred = this.unet.forward([z], [t], [textEmb], 1)[0];

      // x0_pred
      const sqrtAbT = Math.sqrt(this.schedule.alphaBar[t]);
      const sqrtOneMinusAbT = Math.sqrt(1 - this.schedule.alphaBar[t]);
      const x0Pred = new Float32Array(nPixels);
      for (let i2 = 0; i2 < nPixels; i2++) {
        x0Pred[i2] = (z[i2] - sqrtOneMinusAbT * epsilonPred[i2]) / sqrtAbT;
        x0Pred[i2] = Math.max(-1, Math.min(1, x0Pred[i2]));
      }

      // DDIM step
      const abTm1 = this.schedule.alphaBar[tPrev];
      const sqrtAbTm1 = Math.sqrt(abTm1);
      const sigma = Math.sqrt(Math.max(0, 1 - abTm1 - (abTm1 / this.schedule.alphaBar[t]) * (1 - this.schedule.alphaBar[t])));

      for (let i2 = 0; i2 < nPixels; i2++) {
        z[i2] = sqrtAbTm1 * x0Pred[i2] + sigma * epsilonPred[i2];
      }
    }

    const decoded = this.vae.decode(z, 1);
    return decoded;
  }

  /**
   * 训练一步: 去噪训练
   * @param {Float32Array} images - [batch, inputSize, inputSize, 3]
   * @param {string} prompt
   * @returns {number} 损失
   */
  trainStep(images, prompt) {
    const batch = 1;
    const nPixels = this.vae.getLatentShape().reduce((a, b) => a * b, 1);

    // VAE 编码
    const { mu, sigma } = this.vae.encode(images, batch);
    const eps = this._randn(mu.length);
    const z0 = this.vae.reparameterize(mu, sigma, eps);

    // 随机时间步
    const t = Math.floor(Math.random() * this.T);
    const noise = this._randn(nPixels);

    // 前向加噪 (addNoise 返回 Float32Array[])
    const noiseFlat = this._randn(nPixels);
    const z0Flat = z0; // 已经是 [nPixels]
    const xt = addNoise(new Float32Array([
      ...z0Flat,
    ]), [t], noiseFlat, this.schedule.alphaBar)[0];

    // UNet 预测
    const textEmb = this.encodeText(prompt);
    const epsilonPred = this.unet.forward([xt], [t], [textEmb], batch)[0];

    // 损失: MSE
    let loss = 0;
    for (let i = 0; i < nPixels; i++) {
      const diff = epsilonPred[i] - noise[i];
      loss += diff * diff;
    }
    loss /= nPixels;

    return loss;
  }

  _setSeed(seed) {
    this._seed = seed;
  }

  /** 伪随机数生成器 (LCG), 受 seed 控制 */
  _rand() {
    this._seed = (this._seed * 1103515245 + 12345) & 0x7fffffff;
    return this._seed / 0x7fffffff;
  }

  /** 高斯噪声 (受 seed 控制) */
  _randn(n) {
    const arr = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const u1 = Math.max(1e-10, this._rand());
      const u2 = this._rand();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      arr[i] = z;
    }
    return arr;
  }

  countParams() {
    return this.vae ? 0 : 0; // VAE 没有 countParams
    // 实际应统计 VAE + UNet 参数
  }
}

module.exports = { Text2Image };