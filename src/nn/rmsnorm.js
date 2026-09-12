/**
 * rmsnorm.js — RMS Norm (Root Mean Square Layer Normalization)
 *
 * 公式: output = x / RMS(x, axis=-1) * gamma
 *   RMS(x) = sqrt(mean(x², axis=-1))
 *
 * 相比 LayerNorm 快 ~20%, 无需减去 mean
 * 是 LLaMA / GPT-4 / Mistral 的标配归一化层
 *
 * @param {Float32Array} x - 输入 [..., dModel]
 * @param {Float32Array} gamma - 可学习缩放参数 [dModel]
 * @param {number} eps - 数值稳定性常数, 默认 1e-6
 * @returns {Float32Array} 输出 [..., dModel]
 */
function rmsNorm(x, gamma, eps = 1e-6) {
  const dModel = gamma.length;
  const n = x.length;
  const out = new Float32Array(n);

  // 按最后一个维度计算 RMS
  const numVectors = n / dModel;
  for (let i = 0; i < numVectors; i++) {
    const offset = i * dModel;
    // 计算 mean(x²)
    let sumSq = 0;
    for (let j = 0; j < dModel; j++) {
      const val = x[offset + j];
      sumSq += val * val;
    }
    const rms = Math.sqrt(sumSq / dModel + eps);

    // output = x / rms * gamma
    for (let j = 0; j < dModel; j++) {
      out[offset + j] = (x[offset + j] / rms) * gamma[j];
    }
  }

  return out;
}

/**
 * 批量 RMSNorm: 对矩阵 (rows × dModel) 操作
 * @param {Float32Array} X - [rows, dModel]
 * @param {number} rows
 * @param {number} dModel
 * @param {Float32Array} gamma
 * @returns {Float32Array}
 */
function rmsNormBatch(X, rows, dModel, gamma, eps = 1e-6) {
  const out = new Float32Array(rows * dModel);
  for (let i = 0; i < rows; i++) {
    const offset = i * dModel;
    let sumSq = 0;
    for (let j = 0; j < dModel; j++) {
      const val = X[offset + j];
      sumSq += val * val;
    }
    const rms = Math.sqrt(sumSq / dModel + eps);
    for (let j = 0; j < dModel; j++) {
      out[offset + j] = (X[offset + j] / rms) * gamma[j];
    }
  }
  return out;
}

module.exports = { rmsNorm, rmsNormBatch };