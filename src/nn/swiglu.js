/**
 * swiglu.js — SwiGLU Feed-Forward Network
 *
 * 公式: output = W3 · (swish(W1 · x) ⊙ (W2 · x))
 *   swish(x) = x · sigmoid(x)
 *
 * SwiGLU 相比原始 GELU FFN 快 ~15%, 是 LLaMA/Mistral 的标准 FFN
 *
 * 参数:
 *   W1: [dFF, dModel]  (注意: 列优先存储, 便于 matmul)
 *   W2: [dFF, dModel]
 *   W3: [dModel, dFF]
 *
 * @param {Float32Array} x - 输入 [..., dModel]
 * @param {number} nVec - 向量数量 (行数)
 * @param {number} dModel - 隐藏维度
 * @param {number} dFF - FF 维度
 * @param {Float32Array} W1 - [dFF × dModel]
 * @param {Float32Array} W2 - [dFF × dModel]
 * @param {Float32Array} W3 - [dModel × dFF]
 * @returns {Float32Array} output [..., dModel]
 */
function swiglu(x, nVec, dModel, dFF, W1, W2, W3) {
  // step 1: a = x @ W1ᵀ → [nVec, dFF]
  // step 2: b = x @ W2ᵀ → [nVec, dFF]
  // step 3: c = a * sigmoid(a) * b → [nVec, dFF]
  // step 4: output = c @ W3 → [nVec, dModel]

  const ab = new Float32Array(nVec * dFF);  // a 和 b 并存
  const c = new Float32Array(nVec * dFF);

  // W1ᵀ: W1[i][j] → W1[j][i], 但存储是 row-major [dFF, dModel]
  // x @ W1ᵀ = x @ (W1ᵀ)
  // 因为 W1 是 [dFF, dModel], W1ᵀ 是 [dModel, dFF]
  // 所以 x[i][k] = sum_j x[i][j] * W1ᵀ[j][k] = sum_j x[i][j] * W1[k][j]
  // 即: ab[i*dFF + k] = sum_j x[i*dModel + j] * W1[k*dModel + j]

  for (let i = 0; i < nVec; i++) {
    const xOff = i * dModel;
    // a = x @ W1ᵀ
    for (let k = 0; k < dFF; k++) {
      let s = 0;
      const w1Off = k * dModel;
      for (let j = 0; j < dModel; j++) {
        s += x[xOff + j] * W1[w1Off + j];
      }
      ab[i * dFF + k] = s;
    }
    // b = x @ W2ᵀ
    for (let k = 0; k < dFF; k++) {
      let s = 0;
      const w2Off = k * dModel;
      for (let j = 0; j < dModel; j++) {
        s += x[xOff + j] * W2[w2Off + j];
      }
      // c = swish(a) * b
      const a = ab[i * dFF + k];
      const b = s;
      const swishA = a / (1 + Math.exp(-a));
      c[i * dFF + k] = swishA * b;
    }
  }

  // step 4: output = c @ W3 → [nVec, dModel]
  const output = new Float32Array(nVec * dModel);
  for (let i = 0; i < nVec; i++) {
    const cOff = i * dFF;
    for (let k = 0; k < dModel; k++) {
      let s = 0;
      const w3Off = k * dFF;
      for (let j = 0; j < dFF; j++) {
        s += c[cOff + j] * W3[w3Off + j];
      }
      output[i * dModel + k] = s;
    }
  }

  return output;
}

module.exports = { swiglu };