/**
 * scheduler.js — 学习率调度器
 *
 * 支持:
 *   1. Cosine Decay + Warmup (LLaMA 标准)
 *   2. Linear Warmup + Constant
 *   3. Step Decay
 *
 * Cosine 公式:
 *   warmup 阶段: lr = lr_max * (step / warmupSteps)
 *   decay 阶段:  lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * (step - warmup) / (totalSteps - warmup)))
 */

/**
 * @param {object} opts
 * @param {number} opts.maxLR - 最大学习率
 * @param {number} opts.minLR - 最小学习率, 默认 0
 * @param {number} opts.warmupSteps - warmup 步数
 * @param {number} opts.totalSteps - 总训练步数 (decay 结束)
 * @param {string} opts.schedule - 'cosine' | 'linear' | 'step'
 * @param {number} opts.stepDecayFactor - step decay 衰减因子 (默认 0.1)
 * @param {number} opts.stepDecayEvery - 每多少步衰减一次 (默认 1000)
 */
class LRScheduler {
  constructor(opts) {
    this.maxLR = opts.maxLR || 3e-4;
    this.minLR = opts.minLR || 0;
    this.warmupSteps = opts.warmupSteps || 2000;
    this.totalSteps = opts.totalSteps || 100000;
    this.schedule = opts.schedule || 'cosine';
    this.stepDecayFactor = opts.stepDecayFactor || 0.1;
    this.stepDecayEvery = opts.stepDecayEvery || 1000;
  }

  /**
   * 获取当前步的学习率
   * @param {number} step - 当前训练步数 (从 0 开始)
   * @returns {number}
   */
  getLR(step) {
    if (step < 0) throw new Error('step 不能为负数');

    switch (this.schedule) {
      case 'cosine': return this._cosineLR(step);
      case 'linear': return this._linearLR(step);
      case 'step':   return this._stepLR(step);
      default:
        throw new Error(`未知调度器: ${this.schedule}`);
    }
  }

  /** Cosine + Warmup */
  _cosineLR(step) {
    if (step < this.warmupSteps) {
      return this.maxLR * (step / this.warmupSteps);
    }
    const progress = (step - this.warmupSteps) / (this.totalSteps - this.warmupSteps);
    const clamped = Math.max(0, Math.min(1, progress));
    return this.minLR + 0.5 * (this.maxLR - this.minLR) * (1 + Math.cos(Math.PI * clamped));
  }

  /** Linear Warmup + Constant */
  _linearLR(step) {
    if (step < this.warmupSteps) {
      return this.maxLR * (step / this.warmupSteps);
    }
    return this.maxLR;
  }

  /** Step Decay */
  _stepLR(step) {
    const numDecays = Math.floor(step / this.stepDecayEvery);
    return this.maxLR * Math.pow(this.stepDecayFactor, numDecays);
  }
}

module.exports = { LRScheduler };