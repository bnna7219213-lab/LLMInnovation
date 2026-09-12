/**
 * mamba_block.js — Mamba Block (选择性状态空间模型)
 *
 * 架构:
 *   1. 1D 卷积 (kernel=dConv) → 局部混合
 *   2. 投影 x → Δ, A, B, C (选择性参数)
 *   3. Selective Scan
 *   4. 输出投影
 *
 * 相比 Transformer Attention:
 *   - 时间复杂度: O(n) vs O(n²)
 *   - 推理显存: O(1) vs O(n) (无 KV cache)
 *   - 长序列: 极优 vs 需要 RoPE 插值
 *
 * 参数:
 *   dModel: 隐藏维度
 *   dState: SSM 状态维度 (通常 16)
 *   dConv: 卷积核大小 (通常 4)
 */

const { selectiveScanSequential } = require('./selective_scan.js');

/**
 * 1D 卷积 (简化版, 无 padding)
 * @param {Float32Array} x - [seqLen, channels]
 * @param {Float32Array} W - [dConv, channels]  (conv weights)
 * @param {Float32Array} b - [channels]  bias
 * @param {number} seqLen
 * @param {number} channels
 * @param {number} dConv
 * @returns {Float32Array} [seqLen, channels]
 */
function conv1d(x, W, b, seqLen, channels, dConv) {
  const out = new Float32Array(seqLen * channels);
  for (let t = 0; t < seqLen; t++) {
    for (let c = 0; c < channels; c++) {
      let s = 0;
      for (let k = 0; k < dConv; k++) {
        const srcT = t - (dConv - 1) + k; // causal: 只看当前及之前的
        if (srcT >= 0 && srcT < seqLen) {
          s += x[srcT * channels + c] * W[k * channels + c];
        }
      }
      out[t * channels + c] = s + (b ? b[c] : 0);
    }
  }
  return out;
}

/**
 * silu / swish 激活函数
 */
function silu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i] / (1 + Math.exp(-x[i]));
  }
  return out;
}

/**
 * softplus 激活 (用于 Δ 参数, 保证正数)
 */
function softplus(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    // softplus(x) = ln(1 + e^x), 数值稳定版本
    const xi = x[i];
    out[i] = xi > 20 ? xi : Math.log(1 + Math.exp(xi));
  }
  return out;
}

class MambaBlock {
  constructor(dModel, dState = 16, dConv = 4) {
    this.dModel = dModel;
    this.dState = dState;
    this.dConv = dConv;

    // 1D 卷积权重: [dConv, dModel]
    this.Wconv = this._randn(dConv * dModel, 0, 0.05);
    this.bconv = new Float32Array(dModel);

    // 投影矩阵: x → [Δ, A, B, C, x_proj] 各 dState 维 (除了 x_proj 是 dModel)
    // 简化: 用 4 个独立投影
    this.W_delta = this._randn(dModel * dState, 0, 0.02); // Δ: [dModel, dState]
    this.W_A = this._randn(dModel * dState, 0, 0.02);    // A: [dModel, dState]
    this.W_B = this._randn(dModel * dState, 0, 0.02);    // B: [dModel, dState]
    this.W_C = this._randn(dModel * dState, 0, 0.02);    // C: [dModel, dState]
    this.W_x = this._randn(dModel * dModel, 0, 0.02);    // x_proj: [dModel, dModel]

    // 输出投影: [dModel, dModel]
    this.W_out = this._randn(dModel * dModel, 0, 0.02 / Math.sqrt(dModel));
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

  /**
   * 前向传播
   * @param {Float32Array} x - 输入 [seqLen, dModel]
   * @param {number} seqLen
   * @returns {Float32Array} 输出 [seqLen, dModel]
   */
  forward(x, seqLen) {
    // Step 1: 1D 卷积 + silu
    const convOut = conv1d(x, this.Wconv, this.bconv, seqLen, this.dModel, this.dConv);
    const xSilu = silu(convOut);

    // Step 2: 投影 x → Δ, A, B, C
    const delta = this._matmulVec(xSilu, this.W_delta, seqLen, this.dModel, this.dState);
    const A = this._matmulVec(xSilu, this.W_A, seqLen, this.dModel, this.dState);
    const B = this._matmulVec(xSilu, this.W_B, seqLen, this.dModel, this.dState);
    const C = this._matmulVec(xSilu, this.W_C, seqLen, this.dModel, this.dState);

    // Step 3: 激活 Δ 和 A
    // Δ: softplus → 正数 (步长)
    const deltaAct = softplus(delta);
    // A: 负数 (衰减, -|A|)
    const AAct = new Float32Array(A.length);
    for (let i = 0; i < A.length; i++) AAct[i] = -Math.abs(A[i]);

    // Step 4: x_proj (输入投影)
    const xProj = this._matmulVec(xSilu, this.W_x, seqLen, this.dModel, this.dModel);

    // Step 5: Selective Scan
    // 注意: B, C, x 需要乘以 Δ 的广播
    // B_scaled[t][d] = Δ[t][d] * B[t][d] * xProj[t][d]
    // C_scaled[t][d] = Δ[t][d] * C[t][d]
    // A_scaled[t][d] = exp(Δ[t][d] * A[t][d]) (保证 |A| < 1)
    const B_scaled = new Float32Array(seqLen * this.dState);
    const C_scaled = new Float32Array(seqLen * this.dState);
    const A_scaled = new Float32Array(seqLen * this.dState);

    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < this.dState; d++) {
        const tOff = t * this.dState;
        B_scaled[tOff + d] = deltaAct[tOff + d] * B[tOff + d] * xProj[t * this.dModel + d];
        C_scaled[tOff + d] = deltaAct[tOff + d] * C[tOff + d];
        A_scaled[tOff + d] = Math.exp(deltaAct[tOff + d] * AAct[tOff + d]);
      }
    }

    // Selective Scan: h_t = A_scaled * h_{t-1} + B_scaled * x_t, y_t = C_scaled * h_t
    // B_scaled 已包含 x_proj, 所以 x_t = 1
    const xOnes = new Float32Array(seqLen * this.dState).fill(1);
    const scanOut = selectiveScanSequential(xOnes, A_scaled, B_scaled, C_scaled, seqLen, this.dState);

    // Step 6: 输出投影
    // scanOut: [seqLen, dState] → [seqLen, dModel]
    const output = this._matmulVec(scanOut, this.W_out, seqLen, this.dState, this.dModel);

    return output;
  }

  /** 矩阵向量乘: [n, inDim] @ [inDim, outDim] = [n, outDim] */
  _matmulVec(x, W, n, inDim, outDim) {
    const out = new Float32Array(n * outDim);
    for (let i = 0; i < n; i++) {
      const xOff = i * inDim;
      for (let j = 0; j < outDim; j++) {
        let s = 0;
        const wOff = j * inDim;
        for (let k = 0; k < inDim; k++) {
          s += x[xOff + k] * W[wOff + k];
        }
        out[i * outDim + j] = s;
      }
    }
    return out;
  }

  getParams() {
    return {
      Wconv: this.Wconv, bconv: this.bconv,
      W_delta: this.W_delta, W_A: this.W_A,
      W_B: this.W_B, W_C: this.W_C,
      W_x: this.W_x, W_out: this.W_out,
    };
  }

  countParams() {
    let total = 0;
    for (const p of Object.values(this.getParams())) total += p.length;
    return total;
  }
}

module.exports = { MambaBlock, selectiveScanSequential };