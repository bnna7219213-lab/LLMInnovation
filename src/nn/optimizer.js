/**
 * optimizer.js — AdamW 优化器
 *
 * 公式:
 *   m_t = β1 · m_{t-1} + (1 - β1) · g_t          # 一阶矩估计
 *   v_t = β2 · v_{t-1} + (1 - β2) · g_t²        # 二阶矩估计
 *   m̂_t = m_t / (1 - β1^t)                       # 偏差修正
 *   v̂_t = v_t / (1 - β2^t)
 *   θ = θ - lr · (m̂_t / (√v̂_t + ε) + λ · θ)      # AdamW 权重衰减
 *
 * LLaMA 配置: β1=0.9, β2=0.95, ε=1e-8
 *
 * @param {object} opts
 * @param {number} opts.learningRate - 学习率, 默认 3e-4
 * @param {number} opts.beta1 - 一阶矩, 默认 0.9
 * @param {number} opts.beta2 - 二阶矩, 默认 0.95
 * @param {number} opts.eps - 数值稳定性, 默认 1e-8
 * @param {number} opts.weightDecay - 权重衰减, 默认 0.1
 */
class AdamW {
  constructor(opts = {}) {
    this.lr = opts.learningRate || 3e-4;
    this.beta1 = opts.beta1 || 0.9;
    this.beta2 = opts.beta2 || 0.95;
    this.eps = opts.eps || 1e-8;
    this.weightDecay = opts.weightDecay || 0.1;
    this.step = 0;
  }

  /**
   * 创建 AdamW 状态 (m, v 一阶/二阶矩)
   * @param {Float32Array} param - 参数字段
   * @returns {{m: Float32Array, v: Float32Array}}
   */
  createState(param) {
    return { m: new Float32Array(param.length), v: new Float32Array(param.length) };
  }

  /**
   * 更新单个参数字段 (会递增 step)
   * @param {Float32Array} param - [in-place 修改]
   * @param {Float32Array} grad - 梯度
   * @param {{m: Float32Array, v: Float32Array}} state - Adam 状态
   * @param {boolean} applyWeightDecay - 是否应用权重衰减
   */
  stepUpdate(param, grad, state, applyWeightDecay = true) {
    this.step++;
    this._applyUpdate(param, grad, state, applyWeightDecay);
  }

  /**
   * 批量更新多个参数字段 (所有字段共享同一个 step, 只递增 1 次)
   * @param {Array} paramsAndGrads - [{param, grad, applyDecay?}]
   * @param {Array} states - 优化器状态数组
   */
  update(paramsAndGrads, states) {
    this.step++;
    for (let i = 0; i < paramsAndGrads.length; i++) {
      const { param, grad, applyDecay = true } = paramsAndGrads[i];
      this._applyUpdate(param, grad, states[i], applyDecay);
    }
  }

  /** 内部: 实际执行参数更新 (不递增 step) */
  _applyUpdate(param, grad, state, applyWeightDecay) {
    const n = param.length;
    const { beta1, beta2, eps } = this;
    const beta1Pow = Math.pow(beta1, this.step);
    const beta2Pow = Math.pow(beta2, this.step);
    const mHat = 1 - beta1Pow;
    const vHat = 1 - beta2Pow;

    for (let i = 0; i < n; i++) {
      const g = grad[i];
      state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
      state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
      const m = state.m[i] / mHat;
      const v = state.v[i] / vHat;
      let u = m / (Math.sqrt(v) + eps);
      if (applyWeightDecay) u += this.weightDecay * param[i];
      param[i] -= this.lr * u;
    }
  }

  getStep() { return this.step; }
  setLearningRate(lr) { this.lr = lr; }
  getLearningRate() { return this.lr; }
}

module.exports = { AdamW };