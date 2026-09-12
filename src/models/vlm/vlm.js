/**
 * vlm.js — Vision-Language Model 多模态融合
 *
 * 架构 (LLaVA 风格):
 *
 *   图像 → VisionEncoder → 视觉 token [numPatches, dVis]
 *        ↓ linear projection (dVis → dModel)
 *   视觉 token [numPatches, dModel] + 文本 token [seqLen, dModel]
 *        ↓ concat
 *   联合序列 [numPatches + seqLen, dModel]
 *        ↓ LLM Backbone (Transformer)
 *   文本 logits [seqLen, vocabSize] (只对文本位置预测)
 *
 * 流程:
 *   1. encodeImage(image) → 视觉 token
 *   2. embedText(tokenIds) → 文本 token embedding
 *   3. concat → 联合序列
 *   4. Transformer 前向 → 隐藏状态
 *   5. 取文本位置 → logits
 *
 * 微型配置 (测试用):
 *   vision: imageSize=16, patchSize=8, dModel=16, numLayers=1
 *   text: vocabSize=32, numLayers=2
 */

const { VisionEncoder } = require('./vision_encoder.js');
const { rmsNormBatch } = require('../../nn/rmsnorm.js');
const { swiglu } = require('../../nn/swiglu.js');
const { applyRoPE } = require('../../nn/rope.js');
const { multiHeadAttention } = require('../../nn/attention.js');

class VLM {
  /**
   * @param {object} config
   * @param {object} config.vision - VisionEncoder 配置
   * @param {number} config.vocabSize - 词表大小
   * @param {number} config.dModel - LLM 隐藏维度
   * @param {number} config.numLayers - LLM 层数
   * @param {number} config.numHeads - 注意力头数
   * @param {number} config.numKVHeads - KV 头数 (GQA)
   * @param {number} config.dFF - FF 维度
   * @param {number} config.maxSeqLen - 最大序列长度
   */
  constructor(config) {
    this.config = config;
    this.vision = new VisionEncoder(config.vision);
    this.vocabSize = config.vocabSize || 32;
    this.dModel = config.dModel || 16;
    this.numLayers = config.numLayers || 2;
    this.numHeads = config.numHeads || 2;
    this.numKVHeads = config.numKVHeads || this.numHeads;
    this.dK = this.dModel / this.numHeads;
    this.dFF = config.dFF || this.dModel * 4;
    this.maxSeqLen = config.maxSeqLen || 512;
    this.ropeBase = config.ropeBase || 10000;

    if (this.dModel % this.numHeads !== 0) {
      throw new Error(`dModel ${this.dModel} 必须被 numHeads ${this.numHeads} 整除`);
    }
    if (this.numHeads % this.numKVHeads !== 0) {
      throw new Error(`numHeads ${this.numHeads} 必须被 numKVHeads ${this.numKVHeads} 整除`);
    }

    this._initWeights();
  }

  _initWeights() {
    const std = 0.02;

    // 视觉→文本投影: [dVis, dModel]  (vision.dModel → dModel)
    this.Wvis = this._randn(this.vision.dModel * this.dModel, 0, std / Math.sqrt(this.vision.dModel));

    // 文本 token embedding [vocabSize, dModel]
    this.Wembed = this._randn(this.vocabSize * this.dModel, 0, 0.5);

    // LLM layers
    this.layers = [];
    for (let l = 0; l < this.numLayers; l++) {
      this.layers.push({
        preAttnGamma: new Float32Array(this.dModel).fill(1),
        WQ: this._randn(this.numHeads * this.dK * this.dModel, 0, std / Math.sqrt(this.dK)),
        WK: this._randn(this.numKVHeads * this.dK * this.dModel, 0, std / Math.sqrt(this.dK)),
        WV: this._randn(this.numKVHeads * this.dK * this.dModel, 0, std),
        WO: this._randn(this.dModel * this.numHeads * this.dK, 0, std),
        preFFNGamma: new Float32Array(this.dModel).fill(1),
        W1: this._randn(this.dFF * this.dModel, 0, std / Math.sqrt(this.dModel)),
        W2: this._randn(this.dFF * this.dModel, 0, std / Math.sqrt(this.dModel)),
        W3: this._randn(this.dModel * this.dFF, 0, std / Math.sqrt(this.dFF)),
      });
    }

    this.normGamma = new Float32Array(this.dModel).fill(1);

    // 因果掩码 (预分配)
    this._causalMask = new Float32Array(this.maxSeqLen * this.maxSeqLen);
    for (let i = 0; i < this.maxSeqLen; i++) {
      for (let j = 0; j < this.maxSeqLen; j++) {
        this._causalMask[i * this.maxSeqLen + j] = j > i ? 1 : 0;
      }
    }
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
   * 图像 → 视觉 token [numPatches, dModel]
   * @param {Float32Array} image
   */
  encodeImage(image) {
    const visTokens = this.vision.forward(image); // [numPatches, dVis]
    const numPatches = this.vision.numPatches;
    // linear projection to dModel
    const projected = new Float32Array(numPatches * this.dModel);
    for (let i = 0; i < numPatches; i++) {
      const srcOff = i * this.vision.dModel;
      const dstOff = i * this.dModel;
      for (let d = 0; d < this.dModel; d++) {
        let s = 0;
        for (let k = 0; k < this.vision.dModel; k++) {
          s += visTokens[srcOff + k] * this.Wvis[k * this.dModel + d];
        }
        projected[dstOff + d] = s;
      }
    }
    return projected;
  }

  /**
   * 文本 token ids → embedding [seqLen, dModel]
   * @param {Array<number>} tokenIds
   */
  embedText(tokenIds) {
    const seqLen = tokenIds.length;
    const emb = new Float32Array(seqLen * this.dModel);
    for (let i = 0; i < seqLen; i++) {
      const tid = tokenIds[i];
      if (tid < 0 || tid >= this.vocabSize) {
        throw new Error(`token_id ${tid} 超出词表范围 [0, ${this.vocabSize - 1}]`);
      }
      const srcOff = tid * this.dModel;
      const dstOff = i * this.dModel;
      for (let d = 0; d < this.dModel; d++) {
        emb[dstOff + d] = this.Wembed[srcOff + d];
      }
    }
    return emb;
  }

  /**
   * 联合前向
   * @param {Float32Array} image - 图像 [H, W, C]
   * @param {Array<number>} tokenIds - 文本 token ids
   * @returns {Float32Array} logits [seqLen, vocabSize] (只对文本位置)
   */
  forward(image, tokenIds) {
    const visEmb = this.encodeImage(image); // [numPatches, dModel]
    const textEmb = this.embedText(tokenIds); // [seqLen, dModel]

    const numPatches = this.vision.numPatches;
    const seqLen = tokenIds.length;
    const totalLen = numPatches + seqLen;

    if (totalLen > this.maxSeqLen) {
      throw new Error(`联合序列长度 ${totalLen} > maxSeqLen ${this.maxSeqLen}`);
    }

    // 拼接 [numPatches, dModel] + [seqLen, dModel]
    let x = new Float32Array(totalLen * this.dModel);
    x.set(visEmb, 0);
    x.set(textEmb, numPatches * this.dModel);

    // 位置: 视觉 patch 和文本 token 统一从 0 编号
    const positions = new Array(totalLen);
    for (let i = 0; i < totalLen; i++) positions[i] = i;

    // 因果掩码: 只对文本位置需要因果 (视觉 token 全可见)
    // 简化: 对整个序列用因果掩码 (视觉 patch 也因果, 对 patch 序列无影响因为 patch 顺序不重要)
    const mask = this._getCausalMask(totalLen);

    // LLM 层
    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.layers[l];
      const xNorm = rmsNormBatch(x, totalLen, this.dModel, layer.preAttnGamma, 1e-6);
      const xWithRoPE = applyRoPE(xNorm, totalLen, this.dModel, positions, this.ropeBase);
      const attnOut = multiHeadAttention(xWithRoPE, totalLen, this.dModel,
        this.numHeads, this.dK, this.numKVHeads,
        layer.WQ, layer.WK, layer.WV, layer.WO, mask);
      for (let i = 0; i < x.length; i++) x[i] += attnOut[i];

      const xFFNNorm = rmsNormBatch(x, totalLen, this.dModel, layer.preFFNGamma, 1e-6);
      const ffOut = swiglu(xFFNNorm, totalLen, this.dModel, this.dFF,
        layer.W1, layer.W2, layer.W3);
      for (let i = 0; i < x.length; i++) x[i] += ffOut[i];
    }

    // Final norm
    const xFinal = rmsNormBatch(x, totalLen, this.dModel, this.normGamma, 1e-6);

    // 取文本位置 (最后 seqLen 个), 输出 logits
    const logits = new Float32Array(seqLen * this.vocabSize);
    for (let i = 0; i < seqLen; i++) {
      const xOff = (numPatches + i) * this.dModel;
      const loOff = i * this.vocabSize;
      for (let j = 0; j < this.vocabSize; j++) {
        let s = 0;
        const wOff = j * this.dModel;
        for (let d = 0; d < this.dModel; d++) {
          s += xFinal[xOff + d] * this.Wembed[wOff + d];
        }
        logits[loOff + j] = s;
      }
    }

    return logits;
  }

  /** 因果掩码 (从预分配截取) */
  _getCausalMask(seqLen) {
    const mask = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i++) {
      const srcOff = i * this.maxSeqLen;
      const dstOff = i * seqLen;
      for (let j = 0; j < seqLen; j++) {
        mask[dstOff + j] = this._causalMask[srcOff + j];
      }
    }
    return mask;
  }

  /**
   * 简单描述生成 (greedy)
   * @param {Float32Array} image
   * @param {Array<number>} promptIds
   * @param {number} maxNewTokens
   * @param {number} temperature
   * @returns {Array<number>}
   */
  generate(image, promptIds, maxNewTokens = 20, temperature = 0) {
    let tokens = [...promptIds];
    for (let step = 0; step < maxNewTokens; step++) {
      if (this.vision.numPatches + tokens.length > this.maxSeqLen) break;
      const logits = this.forward(image, tokens);
      const lastLogits = logits.subarray((tokens.length - 1) * this.vocabSize, tokens.length * this.vocabSize);
      tokens.push(this._sample(lastLogits, temperature));
    }
    return tokens;
  }

  _sample(logits, temperature) {
    if (temperature === 0) {
      let maxVal = -Infinity, maxIdx = 0;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
      }
      return maxIdx;
    }
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > maxVal) maxVal = logits[i];
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      exps[i] = Math.exp((logits[i] - maxVal) / temperature);
      sum += exps[i];
    }
    const probs = new Float32Array(exps.length);
    for (let i = 0; i < exps.length; i++) probs[i] = exps[i] / sum;
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r <= cum) return i;
    }
    return probs.length - 1;
  }

  countParams() {
    let total = this.Wvis.length + this.Wembed.length + this.normGamma.length;
    for (const layer of this.layers) {
      for (const v of Object.values(layer)) total += v.length;
    }
    total += this.vision.countParams();
    return total;
  }
}

module.exports = { VLM };