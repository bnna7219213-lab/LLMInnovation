/**
 * vision_encoder.js — ViT 风格视觉编码器
 *
 * 将图像编码为视觉 token 序列 (供 VLM 使用)
 *
 * 流程:
 *   1. Patchify: 图像 [H, W, C] → N 个 patch [P, P, C] (P = patchSize)
 *   2. Patch Embedding: 每个 patch 线性投影到 dModel 维
 *   3. Positional Encoding: 可学习位置嵌入
 *   4. Transformer Encoder: numLayers 层 self-attention + FFN
 *   5. 输出: [numPatches, dModel] 视觉 token
 *
 * 参考 ViT-B/16: patchSize=16, dModel=768, numLayers=12, numHeads=12
 * 224×224 图像 → 196 个 patch
 *
 * 微型配置 (测试用):
 *   imageSize=32, patchSize=8, dModel=16, numLayers=2, numHeads=2
 *   → 16 个 patch, 每个 16 维
 */

const { rmsNormBatch } = require('../../nn/rmsnorm.js');
const { swiglu } = require('../../nn/swiglu.js');
const { multiHeadAttention, matmul } = require('../../nn/attention.js');

class VisionEncoder {
  /**
   * @param {object} config
   * @param {number} config.imageSize - 图像边长 (假设方形)
   * @param {number} config.patchSize - patch 边长
   * @param {number} config.channels - 输入通道 (默认 3)
   * @param {number} config.dModel - 隐藏维度
   * @param {number} config.numLayers - encoder 层数
   * @param {number} config.numHeads - 注意力头数
   */
  constructor(config) {
    this.imageSize = config.imageSize || 32;
    this.patchSize = config.patchSize || 8;
    this.channels = config.channels || 3;
    this.dModel = config.dModel || 16;
    this.numLayers = config.numLayers || 2;
    this.numHeads = config.numHeads || 2;

    this.numPatchesPerSide = this.imageSize / this.patchSize;
    if (this.numPatchesPerSide !== Math.floor(this.numPatchesPerSide)) {
      throw new Error(`imageSize ${this.imageSize} 必须被 patchSize ${this.patchSize} 整除`);
    }
    this.numPatches = this.numPatchesPerSide * this.numPatchesPerSide;
    this.patchDim = this.patchSize * this.patchSize * this.channels;

    if (this.dModel % this.numHeads !== 0) {
      throw new Error(`dModel ${this.dModel} 必须被 numHeads ${this.numHeads} 整除`);
    }
    this.dK = this.dModel / this.numHeads;

    this._initWeights();
  }

  _initWeights() {
    const std = 0.02;
    // Patch embedding: [patchDim, dModel]
    this.Wpatch = this._randn(this.patchDim * this.dModel, 0, std / Math.sqrt(this.patchDim));
    this.bpatch = new Float32Array(this.dModel);

    // Position embedding: [numPatches, dModel]
    this.posEmbed = this._randn(this.numPatches * this.dModel, 0, std);

    // Encoder layers
    this.layers = [];
    for (let l = 0; l < this.numLayers; l++) {
      this.layers.push({
        preAttnGamma: new Float32Array(this.dModel).fill(1),
        WQ: this._randn(this.numHeads * this.dK * this.dModel, 0, std),
        WK: this._randn(this.numHeads * this.dK * this.dModel, 0, std),
        WV: this._randn(this.numHeads * this.dK * this.dModel, 0, std),
        WO: this._randn(this.dModel * this.numHeads * this.dK, 0, std),
        preFFNGamma: new Float32Array(this.dModel).fill(1),
        W1: this._randn(this.dModel * this.dModel * 4, 0, std),
        W2: this._randn(this.dModel * this.dModel * 4, 0, std),
        W3: this._randn(this.dModel * this.dModel * 4, 0, std),
      });
    }

    // Final norm
    this.normGamma = new Float32Array(this.dModel).fill(1);
  }

  _randn(n, mean = 0, std = 1) {
    const arr = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const u1 = Math.max(1e-12, Math.random());
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      arr[i] = mean + z * std;
    }
    return arr;
  }

  /**
   * Patchify: 图像 → patch 序列
   * @param {Float32Array} image - [H, W, C] 行优先
   * @returns {Float32Array} [numPatches, patchDim]
   */
  patchify(image) {
    const patches = new Float32Array(this.numPatches * this.patchDim);
    const C = this.channels;
    const H = this.imageSize;
    const W = this.imageSize;
    const P = this.patchSize;

    for (let pi = 0; pi < this.numPatchesPerSide; pi++) {
      for (let pj = 0; pj < this.numPatchesPerSide; pj++) {
        const patchIdx = pi * this.numPatchesPerSide + pj;
        const patchOff = patchIdx * this.patchDim;
        let k = 0;
        for (let i = 0; i < P; i++) {
          for (let j = 0; j < P; j++) {
            const h = pi * P + i;
            const w = pj * P + j;
            for (let c = 0; c < C; c++) {
              patches[patchOff + k] = image[(h * W + w) * C + c];
              k++;
            }
          }
        }
      }
    }
    return patches;
  }

  /**
   * 前向传播
   * @param {Float32Array} image - [H, W, C]
   * @returns {Float32Array} 视觉 token [numPatches, dModel]
   */
  forward(image) {
    // 1. Patch embedding
    const patches = this.patchify(image);
    let x = matmul(patches, this.Wpatch, this.numPatches, this.patchDim, this.dModel);
    // + bias + position
    for (let i = 0; i < this.numPatches; i++) {
      const off = i * this.dModel;
      for (let d = 0; d < this.dModel; d++) {
        x[off + d] += this.bpatch[d] + this.posEmbed[off + d];
      }
    }

    // 2. Encoder layers (non-causal, full attention)
    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.layers[l];
      const xNorm = rmsNormBatch(x, this.numPatches, this.dModel, layer.preAttnGamma, 1e-6);
      const attnOut = multiHeadAttention(xNorm, this.numPatches, this.dModel,
        this.numHeads, this.dK, this.numHeads,
        layer.WQ, layer.WK, layer.WV, layer.WO, null); // null mask = full attention
      for (let i = 0; i < x.length; i++) x[i] += attnOut[i];

      const xFFNNorm = rmsNormBatch(x, this.numPatches, this.dModel, layer.preFFNGamma, 1e-6);
      const ffOut = swiglu(xFFNNorm, this.numPatches, this.dModel, this.dModel * 4,
        layer.W1, layer.W2, layer.W3);
      for (let i = 0; i < x.length; i++) x[i] += ffOut[i];
    }

    // 3. Final norm
    return rmsNormBatch(x, this.numPatches, this.dModel, this.normGamma, 1e-6);
  }

  countParams() {
    let total = this.Wpatch.length + this.bpatch.length + this.posEmbed.length + this.normGamma.length;
    for (const layer of this.layers) {
      for (const v of Object.values(layer)) total += v.length;
    }
    return total;
  }
}

module.exports = { VisionEncoder };