/**
 * transformer3b.js — 自研 Decoder-only Transformer 模型
 *
 * 架构 (LLaMA/Mistral 风格):
 *   1. Token Embedding (tie embedding: Woutput = Wembed)
 *   2. numLayers 层 TransformerBlock
 *   3. Final RMSNorm + Linear Projection → logits
 *
 * TransformerBlock:
 *   x = x + MultiHeadAttention(RMSNorm(x))    # Pre-Norm residual
 *   x = x + SwiGLU(RMSNorm(x))                # Pre-Norm residual
 *
 * 3B 配置:
 *   vocabSize: 32768
 *   dModel: 2560
 *   numLayers: 32
 *   numHeads: 32
 *   numKVHeads: 8  (GQA)
 *   dFF: 6912
 *   context: 2048
 *
 * 测试用微型配置 (验证代码路径):
 *   vocabSize: 64, dModel: 16, numLayers: 2, numHeads: 4, numKVHeads: 2, dFF: 32
 */

const { rmsNorm, rmsNormBatch } = require('../../nn/rmsnorm.js');
const { swiglu } = require('../../nn/swiglu.js');
const { applyRoPE } = require('../../nn/rope.js');
const { multiHeadAttention } = require('../../nn/attention.js');
const { AdamW } = require('../../nn/optimizer.js');
const { LRScheduler } = require('../../training/scheduler.js');
const { Vocab } = require('../../tokenizer/vocab.js');

// 默认 3B 配置
const CONFIG_3B = {
  vocabSize: 32768,
  dModel: 2560,
  numLayers: 32,
  numHeads: 32,
  numKVHeads: 8,
  dFF: 6912,
  maxSeqLen: 2048,
  ropeBase: 10000,
  rmsNormEps: 1e-6,
};

class TransformerModel {
  /**
   * @param {object} config
   */
  constructor(config = CONFIG_3B) {
    this.vocabSize = config.vocabSize;
    this.dModel = config.dModel;
    this.numLayers = config.numLayers;
    this.numHeads = config.numHeads;
    this.numKVHeads = config.numKVHeads;
    this.dK = this.dModel / this.numHeads;
    this.dFF = config.dFF || this.dModel * 2;
    this.maxSeqLen = config.maxSeqLen || 2048;
    this.ropeBase = config.ropeBase || 10000;
    this.rmsNormEps = config.rmsNormEps || 1e-6;

    if (this.dModel % this.numHeads !== 0) {
      throw new Error(`dModel ${this.dModel} 必须被 numHeads ${this.numHeads} 整除`);
    }
    if (this.numHeads % this.numKVHeads !== 0) {
      throw new Error(`numHeads ${this.numHeads} 必须被 numKVHeads ${this.numKVHeads} 整除`);
    }

    // 初始化权重
    this._initWeights();

    // 因果掩码: 一次性分配 maxSeqLen × maxSeqLen, 使用时截取前 seqLen 行
    this._causalMask = new Float32Array(this.maxSeqLen * this.maxSeqLen);
    for (let i = 0; i < this.maxSeqLen; i++) {
      for (let j = 0; j < this.maxSeqLen; j++) {
        this._causalMask[i * this.maxSeqLen + j] = j > i ? 1 : 0;
      }
    }
  }

  _initWeights() {
    const scale = (n) => 0.02 / Math.sqrt(n);

    // 1. Embedding (vocabSize × dModel)
    const embScale = 0.5; // uniform(-0.5, 0.5)
    this.Wembed = this._randn(this.vocabSize * this.dModel, 0, embScale);

    // 2. 每层权重
    this.layers = [];
    for (let l = 0; l < this.numLayers; l++) {
      this.layers.push(this._initLayerWeights(l));
    }

    // 3. Final RMSNorm gamma
    this.normGamma = new Float32Array(this.dModel).fill(1);

    // Woutput = Wembed (tie embedding, 不额外存, 输出时用 Wembed 的转置)
  }

  _initLayerWeights(layerIdx) {
    const { dModel, numHeads, numKVHeads, dK, dFF } = this;

    // Query: WQ [numHeads*dK × dModel]
    // Key:   WK [numKVHeads*dK × dModel]
    // Value: WV [numKVHeads*dK × dModel]
    // Output: WO [dModel × numHeads*dK]
    const qkScale = 0.02 / Math.sqrt(dK);
    const wScale = 0.02;
    const ffScale = 0.02 / Math.sqrt(dModel);

    return {
      WQ: this._randn(numHeads * dK * dModel, 0, qkScale),
      WK: this._randn(numKVHeads * dK * dModel, 0, qkScale),
      WV: this._randn(numKVHeads * dK * dModel, 0, wScale),
      WO: this._randn(dModel * numHeads * dK, 0, wScale),
      // Pre-attention RMSNorm
      preAttnGamma: new Float32Array(dModel).fill(1),
      // Pre-FFN RMSNorm
      preFFNGamma: new Float32Array(dModel).fill(1),
      // SwiGLU: W1, W2 [dFF × dModel], W3 [dModel × dFF]
      W1: this._randn(dFF * dModel, 0, ffScale),
      W2: this._randn(dFF * dModel, 0, ffScale),
      W3: this._randn(dModel * dFF, 0, 0.02 / Math.sqrt(dFF)),
    };
  }

  /** 正态分布随机 Float32Array (Box-Muller) */
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

  /**
   * 前向传播
   * @param {Array<number>} inputIds - token ids [seqLen]
   * @param {number} seqLen - 实际序列长度
   * @returns {Float32Array} logits [seqLen, vocabSize]
   */
  forward(inputIds, seqLen) {
    if (seqLen > this.maxSeqLen) {
      throw new Error(`seqLen ${seqLen} > maxSeqLen ${this.maxSeqLen}`);
    }

    // 位置序列 [0, 1, 2, ..., seqLen-1]
    const positions = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) positions[i] = i;

    // Step 1: Embedding → [seqLen, dModel]
    let x = new Float32Array(seqLen * this.dModel);
    for (let i = 0; i < seqLen; i++) {
      const tid = inputIds[i];
      if (tid < 0 || tid >= this.vocabSize) {
        throw new Error(`token_id ${tid} 超出词表范围 [0, ${this.vocabSize - 1}]`);
      }
      const srcOff = tid * this.dModel;
      const dstOff = i * this.dModel;
      for (let d = 0; d < this.dModel; d++) {
        x[dstOff + d] = this.Wembed[srcOff + d];
      }
    }

    // Step 2: 每层前向
    for (let l = 0; l < this.numLayers; l++) {
      x = this._forwardLayer(x, l, seqLen, positions);
    }

    // Step 3: Final RMSNorm
    x = rmsNormBatch(x, seqLen, this.dModel, this.normGamma, this.rmsNormEps);

    // Step 4: Output projection → logits [seqLen, vocabSize]
    // tie embedding: logits = x @ Wembed.T
    // 注意: Wembed 是 [vocabSize, dModel], x 是 [seqLen, dModel]
    // logits[i][j] = sum_d x[i][d] * Wembed[j][d]
    const logits = new Float32Array(seqLen * this.vocabSize);
    for (let i = 0; i < seqLen; i++) {
      const xOff = i * this.dModel;
      for (let j = 0; j < this.vocabSize; j++) {
        let s = 0;
        const wOff = j * this.dModel;
        for (let d = 0; d < this.dModel; d++) {
          s += x[xOff + d] * this.Wembed[wOff + d];
        }
        logits[i * this.vocabSize + j] = s;
      }
    }

    return logits;
  }

  /**
   * 单层前向
   * @param {Float32Array} x - [seqLen, dModel]
   * @param {number} layerIdx
   * @param {number} seqLen
   * @param {Array} positions
   * @returns {Float32Array}
   */
  _forwardLayer(x, layerIdx, seqLen, positions) {
    const layer = this.layers[layerIdx];

    // Pre-attention RMSNorm
    const xNorm = rmsNormBatch(x, seqLen, this.dModel, layer.preAttnGamma, this.rmsNormEps);

    // RoPE applied inside attention (before QK projection, conceptually)
    // Here we apply RoPE to the input to attention (the "q_input")
    const xWithRoPE = applyRoPE(xNorm, seqLen, this.dModel, positions, this.ropeBase);

    // Multi-Head Attention (with causal mask)
    const attnOut = multiHeadAttention(
      xWithRoPE, seqLen, this.dModel,
      this.numHeads, this.dK, this.numKVHeads,
      layer.WQ, layer.WK, layer.WV, layer.WO,
      this._getCausalMask(seqLen)
    );

    // Residual: x = x + attnOut
    for (let i = 0; i < seqLen * this.dModel; i++) {
      x[i] += attnOut[i];
    }

    // Pre-FFN RMSNorm
    const xFFNNorm = rmsNormBatch(x, seqLen, this.dModel, layer.preFFNGamma, this.rmsNormEps);

    // SwiGLU
    const ffOut = swiglu(xFFNNorm, seqLen, this.dModel, this.dFF,
      layer.W1, layer.W2, layer.W3);

    // Residual: x = x + ffOut
    for (let i = 0; i < seqLen * this.dModel; i++) {
      x[i] += ffOut[i];
    }

    return x;
  }

  /** 获取因果掩码 (从预分配的 maxSeqLen 掩码中截取) */
  _getCausalMask(seqLen) {
    // 注意: attention.js 使用 mask[i*nSeq+j] 索引, 但我们预分配的是 [maxSeqLen, maxSeqLen]
    // 需要临时构建 [seqLen, seqLen] 的视图
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
   * 生成 token (简单 greedy sampling)
   * @param {Array<number>} promptIds
   * @param {number} maxNewTokens
   * @param {number} temperature - 0 = greedy, >0 = softmax sampling
   * @returns {Array<number>} 生成的完整 token 序列
   */
  generate(promptIds, maxNewTokens = 100, temperature = 0) {
    let tokens = [...promptIds];

    for (let step = 0; step < maxNewTokens; step++) {
      const seqLen = tokens.length;
      if (seqLen > this.maxSeqLen) break;

      const logits = this.forward(tokens, seqLen);
      const lastLogits = logits.subarray((seqLen - 1) * this.vocabSize, seqLen * this.vocabSize);

      const nextToken = this._sample(lastLogits, temperature);
      tokens.push(nextToken);
    }

    return tokens;
  }

  /** softmax + sampling */
  _sample(logits, temperature) {
    if (temperature === 0) {
      // greedy: argmax
      let maxVal = -Infinity, maxIdx = 0;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
      }
      return maxIdx;
    }

    // softmax with temperature
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxVal) maxVal = logits[i];
    }
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      exps[i] = Math.exp((logits[i] - maxVal) / temperature);
      sum += exps[i];
    }
    const probs = new Float32Array(exps.length);
    for (let i = 0; i < exps.length; i++) probs[i] = exps[i] / sum;

    // 累积采样
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r <= cum) return i;
    }
    return probs.length - 1;
  }

  /** 获取所有参数字段 (用于训练/保存) */
  getParams() {
    const params = { 'Wembed': this.Wembed };
    for (let l = 0; l < this.numLayers; l++) {
      const p = this.layers[l];
      params[`layer${l}.WQ`] = p.WQ;
      params[`layer${l}.WK`] = p.WK;
      params[`layer${l}.WV`] = p.WV;
      params[`layer${l}.WO`] = p.WO;
      params[`layer${l}.preAttnGamma`] = p.preAttnGamma;
      params[`layer${l}.preFFNGamma`] = p.preFFNGamma;
      params[`layer${l}.W1`] = p.W1;
      params[`layer${l}.W2`] = p.W2;
      params[`layer${l}.W3`] = p.W3;
    }
    params['normGamma'] = this.normGamma;
    return params;
  }

  /** 统计总参数量 */
  countParams() {
    let total = 0;
    for (const p of Object.values(this.getParams())) {
      total += p.length;
    }
    return total;
  }
}

module.exports = { TransformerModel, CONFIG_3B };