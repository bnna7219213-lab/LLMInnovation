/**
 * rope.js — Rotary Position Embedding (RoPE) + NTK 插值
 *
 * 公式 (对每两个连续维度):
 *   angle = pos * i / (base ** (2*i / dModel))   # i = 0..dModel/2-1
 *   [x_2i', x_{2i+1}'] = [cos(θ), -sin(θ); sin(θ), cos(θ)] · [x_2i, x_{2i+1}]
 *
 * NTK 插值: base 调整为 base * scale ** ((scale - 1) / alpha)
 *   用于外推到训练外的长序列
 *   alpha=1, scale=2 → 支持 2x context
 *   alpha=1, scale=8 → 支持 8x context (Llama 3)
 *
 * @param {Float32Array} x - 输入 [..., dModel]  (最后一个维度是 dModel)
 * @param {number} nVec - 向量数量
 * @param {number} dModel - 隐藏维度 (必须为偶数)
 * @param {number[]} positions - 每个向量的位置 [nVec]
 * @param {number} base - 基础频率, 默认 10000 (LLaMA)
 * @param {number} scale - NTK 插值 scale, 默认 1 (无插值)
 * @param {number} alpha - NTK alpha, 默认 1
 * @returns {Float32Array} 输出 [..., dModel]
 */
function applyRoPE(x, nVec, dModel, positions, base = 10000, scale = 1, alpha = 1) {
  if (dModel % 2 !== 0) throw new Error('dModel 必须为偶数');

  // NTK 插值调整 base
  const ntkBase = base * Math.pow(scale, (scale - 1) / alpha);

  const nHalf = dModel / 2;
  const output = new Float32Array(x.length);

  // 预计算每个位置的角度 (nHalf 个角度 per position)
  // 对于每个向量 i, 需要 [nHalf] 个角度
  for (let vi = 0; vi < nVec; vi++) {
    const pos = positions[vi];
    const xOff = vi * dModel;
    const oOff = vi * dModel;

    for (let i = 0; i < nHalf; i++) {
      // 计算角度
      const freq = i / nHalf;
      const invFreq = 1 / Math.pow(ntkBase, freq);
      const angle = pos * invFreq;

      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      const x1 = x[xOff + 2 * i];
      const x2 = x[xOff + 2 * i + 1];

      // 2D 旋转
      output[oOff + 2 * i] = x1 * cosA - x2 * sinA;
      output[oOff + 2 * i + 1] = x1 * sinA + x2 * cosA;
    }
  }

  return output;
}

module.exports = { applyRoPE };