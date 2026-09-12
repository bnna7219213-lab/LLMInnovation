/**
 * transformer100b.js — 100B MoE (Mixture of Experts) 配置
 *
 * 100B 模型采用稀疏 MoE 架构:
 *   - 标准 Transformer 层 + MoE FFN (替换密集 SwiGLU)
 *   - 每个 token 路由到 top-k 个专家 (通常 k=2)
 *   - 专家并行 (EP) 跨设备
 *
 * 100B 配置 (参考 Mixtral 8x22B / DeepSeek-V2 风格):
 *   vocabSize: 128000
 *   dModel: 8192
 *   numLayers: 64
 *   numHeads: 64
 *   numKVHeads: 8  (GQA, 强 KV 压缩)
 *   dFF: 8192 (专家内)
 *   numExperts: 8
 *   topK: 2
 *   maxSeqLen: 32768
 *
 * MoE 路由:
 *   gate = softmax(x @ Wgate)  [numExperts]
 *   选取 top-k 专家, 加权求和各专家输出
 *
 * 优势:
 *   - 总参数 ~114B, 但每 token 只激活 ~14B (计算效率高)
 *   - 专家并行跨多 GPU
 */

class MoERouter {
  /**
   * @param {object} config
   * @param {number} config.dModel
   * @param {number} config.numExperts
   * @param {number} config.topK
   */
  constructor(config) {
    this.dModel = config.dModel;
    this.numExperts = config.numExperts;
    this.topK = config.topK || 2;

    // 路由权重 [dModel, numExperts]
    const std = 0.02 / Math.sqrt(this.dModel);
    this.Wgate = new Float32Array(this.dModel * this.numExperts);
    for (let i = 0; i < this.Wgate.length; i++) {
      this.Wgate[i] = (Math.random() - 0.5) * 2 * std;
    }
  }

  /**
   * 路由: 计算每个 token 的专家得分并选 top-k
   * @param {Float32Array} x - [dModel]
   * @returns {{expertIdx: number[], weights: number[]}}
   */
  route(x) {
    // gate logits
    const logits = new Float32Array(this.numExperts);
    let maxVal = -Infinity;
    for (let e = 0; e < this.numExperts; e++) {
      let s = 0;
      for (let d = 0; d < this.dModel; d++) {
        s += x[d] * this.Wgate[d * this.numExperts + e];
      }
      logits[e] = s;
      if (s > maxVal) maxVal = s;
    }

    // softmax
    const probs = new Float32Array(this.numExperts);
    let sum = 0;
    for (let e = 0; e < this.numExperts; e++) {
      probs[e] = Math.exp(logits[e] - maxVal);
      sum += probs[e];
    }
    for (let e = 0; e < this.numExperts; e++) probs[e] /= sum;

    // 选 top-k
    const sorted = Array.from(probs).map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
    const expertIdx = [];
    const weights = [];
    for (let k = 0; k < this.topK; k++) {
      expertIdx.push(sorted[k].i);
      weights.push(sorted[k].p);
    }
    // 归一化 top-k 权重
    const wSum = weights.reduce((a, b) => a + b, 0);
    for (let k = 0; k < this.topK; k++) weights[k] /= wSum;

    return { expertIdx, weights };
  }
}

/**
 * 100B 模型配置 (MoE)
 */
const CONFIG_100B = {
  vocabSize: 128000,
  dModel: 8192,
  numLayers: 64,
  numHeads: 64,
  numKVHeads: 8,
  dFF: 8192,
  maxSeqLen: 32768,
  ropeBase: 10000,
  rmsNormEps: 1e-6,
  // MoE 配置
  moe: {
    numExperts: 8,
    topK: 2,
    expertParallel: 8, // 专家并行度
  },
};

/**
 * 估算参数量
 * @param {object} config
 * @returns {number}
 */
function estimateParams(config = CONFIG_100B) {
  const { vocabSize, dModel, numLayers, numHeads, numKVHeads, dFF, moe } = config;

  // Embedding + final norm
  let total = vocabSize * dModel; // Wembed (tie)
  total += dModel; // final norm

  // 每层
  const dK = dModel / numHeads;
  const attnParams = numHeads * dK * dModel * 2 // WQ, WO
    + numKVHeads * dK * dModel * 2; // WK, WV
  const moeParams = moe.numExperts * (2 * dFF * dModel + dModel * dFF); // W1, W2, W3 per expert
  const gateParams = dModel * moe.numExperts; // router

  total += numLayers * (attnParams + moeParams + gateParams + 2 * dModel); // 2 norm gamma

  return total;
}

module.exports = { MoERouter, CONFIG_100B, estimateParams };