/**
 * jamba.js — Jamba 风格混合架构 (Mamba + Transformer 交替)
 *
 * 架构:
 *   Layer 0:  Mamba Block        (全局依赖, 线性时间)
 *   Layer 1:  Transformer Block  (局部语义, 精确注意力)
 *   Layer 2:  Mamba Block
 *   Layer 3:  Transformer Block
 *   ...
 *
 * 优势:
 *   - 长序列上 Mamba 处理全局依赖 (O(n))
 *   - 短序列上 Transformer 处理精确局部 (O(n²) 但 n 小)
 *   - 交替结构互补, 效果优于纯 Mamba 或纯 Transformer
 *
 * 参数:
 *   vocabSize, dModel, numLayers, numHeads, numKVHeads, dFF (Transformer 部分)
 *   dState, dConv (Mamba 部分)
 */

const { TransformerModel } = require('../text/transformer3b.js');
const { MambaBlock } = require('../mamba/mamba_block.js');
const { rmsNormBatch } = require('../../nn/rmsnorm.js');
const { applyRoPE } = require('../../nn/rope.js');
const { multiHeadAttention } = require('../../nn/attention.js');

class JambaModel {
  constructor(config) {
    this.vocabSize = config.vocabSize || 32768;
    this.dModel = config.dModel || 512;
    this.numLayers = config.numLayers || 8;
    this.numHeads = config.numHeads || 8;
    this.numKVHeads = config.numKVHeads || 4;
    this.dK = this.dModel / this.numHeads;
    this.dFF = config.dFF || this.dModel * 2;
    this.maxSeqLen = config.maxSeqLen || 2048;
    this.dState = config.dState || 16;
    this.dConv = config.dConv || 4;
    this.ropeBase = config.ropeBase || 10000;
    this.rmsNormEps = config.rmsNormEps || 1e-6;

    // 初始化权重
    this.Wembed = this._randn(this.vocabSize * this.dModel, 0, 0.5);
    this.normGamma = new Float32Array(this.dModel).fill(1);

    // 交替层: 偶数层 = Mamba, 奇数层 = Transformer
    this.blocks = [];
    for (let l = 0; l < this.numLayers; l++) {
      if (l % 2 === 0) {
        this.blocks.push({ type: 'mamba', block: new MambaBlock(this.dModel, this.dState, this.dConv) });
      } else {
        this.blocks.push({ type: 'transformer', block: this._makeTransformerLayer() });
      }
    }

    // 预分配因果掩码
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
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      arr[i] = mean + z * std;
    }
    return arr;
  }

  _makeTransformerLayer() {
    const { dModel, numHeads, numKVHeads, dK, dFF } = this;
    const qkScale = 0.02 / Math.sqrt(dK);
    return {
      WQ: this._randn(numHeads * dK * dModel, 0, qkScale),
      WK: this._randn(numKVHeads * dK * dModel, 0, qkScale),
      WV: this._randn(numKVHeads * dK * dModel, 0, 0.02),
      WO: this._randn(dModel * numHeads * dK, 0, 0.02),
      preAttnGamma: new Float32Array(dModel).fill(1),
      preFFNGamma: new Float32Array(dModel).fill(1),
      W1: this._randn(dFF * dModel, 0, 0.02 / Math.sqrt(dModel)),
      W2: this._randn(dFF * dModel, 0, 0.02 / Math.sqrt(dModel)),
      W3: this._randn(dModel * dFF, 0, 0.02 / Math.sqrt(dFF)),
    };
  }

  /**
   * 前向传播
   * @param {Array<number>} inputIds
   * @param {number} seqLen
   * @returns {Float32Array} logits [seqLen, vocabSize]
   */
  forward(inputIds, seqLen) {
    const positions = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) positions[i] = i;

    // Embedding
    let x = new Float32Array(seqLen * this.dModel);
    for (let i = 0; i < seqLen; i++) {
      const tid = inputIds[i];
      const srcOff = tid * this.dModel;
      const dstOff = i * this.dModel;
      for (let d = 0; d < this.dModel; d++) x[dstOff + d] = this.Wembed[srcOff + d];
    }

    // 每层前向
    for (let l = 0; l < this.numLayers; l++) {
      const { type, block } = this.blocks[l];
      if (type === 'mamba') {
        x = block.forward(x, seqLen);
      } else {
        x = this._forwardTransformerLayer(x, block, seqLen, positions);
      }
    }

    // Final RMSNorm
    x = rmsNormBatch(x, seqLen, this.dModel, this.normGamma, this.rmsNormEps);

    // Output projection (tie embedding)
    const logits = new Float32Array(seqLen * this.vocabSize);
    for (let i = 0; i < seqLen; i++) {
      const xOff = i * this.dModel;
      for (let j = 0; j < this.vocabSize; j++) {
        let s = 0;
        const wOff = j * this.dModel;
        for (let d = 0; d < this.dModel; d++) s += x[xOff + d] * this.Wembed[wOff + d];
        logits[i * this.vocabSize + j] = s;
      }
    }

    return logits;
  }

  _forwardTransformerLayer(x, layer, seqLen, positions) {
    const xNorm = rmsNormBatch(x, seqLen, this.dModel, layer.preAttnGamma, this.rmsNormEps);
    const xRoPE = applyRoPE(xNorm, seqLen, this.dModel, positions, this.ropeBase);

    const mask = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i++) {
      const srcOff = i * this.maxSeqLen;
      const dstOff = i * seqLen;
      for (let j = 0; j < seqLen; j++) mask[dstOff + j] = this._causalMask[srcOff + j];
    }

    const attnOut = multiHeadAttention(xRoPE, seqLen, this.dModel,
      this.numHeads, this.dK, this.numKVHeads,
      layer.WQ, layer.WK, layer.WV, layer.WO, mask);

    for (let i = 0; i < seqLen * this.dModel; i++) x[i] += attnOut[i];

    const xFFN = rmsNormBatch(x, seqLen, this.dModel, layer.preFFNGamma, this.rmsNormEps);
    const { swiglu } = require('../../nn/swiglu.js');
    const ffOut = swiglu(xFFN, seqLen, this.dModel, this.dFF, layer.W1, layer.W2, layer.W3);
    for (let i = 0; i < seqLen * this.dModel; i++) x[i] += ffOut[i];

    return x;
  }

  /**
   * 生成 token
   * @param {Array<number>} promptIds
   * @param {number} maxNewTokens
   * @param {number} temperature
   * @returns {Array<number>}
   */
  generate(promptIds, maxNewTokens = 100, temperature = 0) {
    let tokens = [...promptIds];
    for (let step = 0; step < maxNewTokens; step++) {
      const seqLen = tokens.length;
      if (seqLen > this.maxSeqLen) break;
      const logits = this.forward(tokens, seqLen);
      const lastLogits = logits.subarray((seqLen - 1) * this.vocabSize, seqLen * this.vocabSize);
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
    for (let i = 0; i < logits.length; i++) { if (logits[i] > maxVal) maxVal = logits[i]; }
    const probs = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp((logits[i] - maxVal) / temperature);
      sum += probs[i];
    }
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i] / sum;
      if (r <= cum) return i;
    }
    return probs.length - 1;
  }

  /** 统计 Mamba 层和 Transformer 层数量 */
  countLayerTypes() {
    let mamba = 0, transformer = 0;
    for (const { type } of this.blocks) {
      if (type === 'mamba') mamba++;
      else transformer++;
    }
    return { mamba, transformer };
  }

  countParams() {
    let total = this.Wembed.length + this.normGamma.length;
    for (const { block } of this.blocks) {
      total += block.countParams ? block.countParams() : 0;
    }
    for (const { type, block } of this.blocks) {
      if (type === 'transformer') {
        total += block.WQ.length + block.WK.length + block.WV.length + block.WO.length;
        total += block.preAttnGamma.length + block.preFFNGamma.length;
        total += block.W1.length + block.W2.length + block.W3.length;
      }
    }
    return total;
  }
}

module.exports = { JambaModel };