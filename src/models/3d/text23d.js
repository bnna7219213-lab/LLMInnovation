/**
 * text23d.js — Text-to-3D 生成管道
 *
 * 流程:
 *   1. 文本编码: text → textEmbedding (LLM 或 CLIP)
 *   2. 初始化 triplane (随机或小噪声)
 *   3. 多视角 SDS 优化:
 *      for each step:
 *        - 随机相机视角
 *        - 渲染 2D 图像 x_0
 *        - 前向加噪 z_t = √ᾱ * x_0 + √(1-ᾱ) * ε
 *        - UNet 预测噪声 ε_pred = UNet(z_t, t, text)
 *        - 损失: L = 0.5 * w(t) * ||ε_pred - ε||²
 *        - 梯度: ∂L/∂θ (通过有限差分或随机投影)
 *        - 更新 triplane 参数
 *   4. 输出: triplane 模型 (支持任意视角渲染)
 *
 * 参数量估算 (以默认配置为例):
 *   - Triplane: 3 * 16 * 16 * 8 = 6144
 *   - UNet: ~50K (Tiny UNet)
 *   - 文本嵌入: 8 * 16 = 128
 *   - 总参数量: ~56K (CPU 可行)
 */

class Text23D {
  constructor(config) {
    this.config = config || {};

    // 从 text2image.js 导入依赖
    const { UNet } = require('../diffusion/unet.js');
    const { generateSchedule } = require('../diffusion/diffusion_schedule.js');

    // UNet 配置
    const unetConfig = this.config.unetConfig || {
      latentChannels: 3,
      latentSize: 32,
      textEmbedDim: 64,
      textTokenLen: 16,
      blockChannels: [32, 64],
      numHeads: 2,
    };

    // 扩散调度
    this.T = 1000;
    this.schedule = generateSchedule('cosine', this.T, { s: 0.008 });

    // UNet
    this.unet = new UNet(unetConfig);

    // Triplane 配置
    const triplaneConfig = this.config.triplaneConfig || {
      planeSize: 16,
      latentChannels: 8,
      mlpHidden: 16,
      numMLPLayers: 1,
    };
    this.triplaneConfig = triplaneConfig;

    // SDS 配置
    this.sdsConfig = this.config.sdsConfig || {
      tMin: 50,
      tMax: 500,
      imgSize: 32,
      numSamples: 8,
      fov: Math.PI / 4,
      distance: 3,
    };
  }

  /**
   * 文本编码 (简化版: 字符频率编码)
   * @param {string} prompt
   * @returns {Float32Array} [textTokenLen, textEmbedDim]
   */
  encodeText(prompt, tokenLen = 16, embedDim = 64) {
    const emb = new Float32Array(tokenLen * embedDim);
    for (let t = 0; t < tokenLen; t++) {
      const charCode = t < prompt.length ? prompt.charCodeAt(t) : 0;
      for (let d = 0; d < embedDim; d++) {
        // 简单频率编码 (类似 position encoding)
        emb[t * embedDim + d] = Math.sin(charCode / (Math.pow(10000, d / embedDim)));
      }
    }
    return emb;
  }

  /**
   * 生成 3D 模型
   * @param {string} prompt
   * @param {number} numSteps - 优化步数
   * @param {number} lr - 学习率
   * @param {number} seed - 随机种子
   * @param {Function} progressCallback - 进度回调 (step, loss)
   * @returns {{loss: number, triplane: Triplane}} 3D 模型
   */
  generate(prompt, numSteps = 50, lr = 0.01, seed = 1, progressCallback) {
    // 设置种子
    if (seed > 0) {
      // 简化: 用 Math.random()
      this._seed = seed;
    }

    const textEmb = this.encodeText(prompt);
    const triplane = this._createTriplane();

    // 加载 SDS
    const { SDS } = require('./sds.js');
    const sds = new SDS(this.sdsConfig);

    let loss = 0;
    for (let step = 0; step < numSteps; step++) {
      const result = sds.trainStep(triplane, this.unet, textEmb,
        this.schedule.alpha, this.schedule.alphaBar);

      loss = result.loss;

      // 有限差分梯度更新 (采样几个像素)
      const numPixelsPerStep = 4;
      for (let p = 0; p < numPixelsPerStep; p++) {
        const planeIdx = Math.floor(Math.random() * 3);
        const pixelIdx = Math.floor(Math.random() * triplane.planeSize * triplane.planeSize);
        const channel = Math.floor(Math.random() * triplane.latentChannels);
        const grad = sds.finiteDiffGradient(triplane, planeIdx, pixelIdx, channel,
          0.01, this.unet, textEmb, this.schedule.alpha, this.schedule.alphaBar);
        // 更新
        const planes = [triplane.planeXY, triplane.planeXZ, triplane.planeYZ];
        const target = planes[planeIdx];
        const offset = pixelIdx * triplane.latentChannels + channel;
        target[offset] -= lr * grad;
      }

      if (progressCallback) progressCallback(step, loss);
    }

    return { loss, triplane };
  }

  /**
   * 创建新的 triplane
   * @returns {Triplane}
   */
  _createTriplane() {
    const { Triplane } = require('./triplane.js');
    return new Triplane(this.triplaneConfig);
  }

  /**
   * 渲染当前 3D 模型
   * @param {Triplane} triplane
   * @param {number} imgSize
   * @param {number} numSamples
   * @returns {Float32Array} [imgSize, imgSize, 3]
   */
  render(triplane, imgSize = 32, numSamples = 16) {
    return triplane.renderCamera(imgSize, imgSize, this.sdsConfig.fov,
      this.sdsConfig.distance, numSamples);
  }
}

module.exports = { Text23D };