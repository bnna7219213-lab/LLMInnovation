/**
 * attention.js — Multi-Head Attention (MHA) + Grouped Query Attention (GQA)
 *
 * 公式:
 *   Q = X @ WQ  →  [n, numHeads, dK]
 *   K = X @ WK  →  [n, numHeads, dK]
 *   V = X @ WV  →  [n, numHeads, dK]
 *   attention(Q,K,V) = softmax(QKᵀ / √dK + mask) @ V
 *   output = concat(heads) @ WO
 *
 * GQA 模式:
 *   Q 有 numHeads 个 head, K/V 有 numGQAGroups 个 group
 *   每个 KV group 被 numHeads / numGQAGroups 个 Q head 共享
 *   减少 KV cache 显存 (Llama 3 标配)
 *
 * 本实现: 手写 softmax + causal mask, 不依赖外部库
 *
 * @param {Float32Array} x - 输入 [nSeq, dModel]
 * @param {number} nSeq - 序列长度
 * @param {number} dModel - 隐藏维度
 * @param {number} numHeads - 注意力头数
 * @param {number} dK - 每头维度 (dModel / numHeads)
 * @param {number} numKVHeads - KV 头数 (GQA: numKVHeads = numGQAGroups; MHA: = numHeads)
 * @param {Float32Array} WQ - [numHeads × dK, dModel]
 * @param {Float32Array} WK - [numKVHeads × dK, dModel]
 * @param {Float32Array} WV - [numKVHeads × dK, dModel]
 * @param {Float32Array} WO - [dModel, numHeads × dK]
 * @param {Float32Array|null} mask - 因果掩码 [nSeq, nSeq], -∞ 表示遮盖
 * @returns {Float32Array} 输出 [nSeq, dModel]
 */
function multiHeadAttention(x, nSeq, dModel, numHeads, dK, numKVHeads,
                             WQ, WK, WV, WO, mask = null) {
  // 验证
  if (numHeads % numKVHeads !== 0) {
    throw new Error(`numHeads ${numHeads} 必须能被 numKVHeads ${numKVHeads} 整除`);
  }

  const headsPerKV = numHeads / numKVHeads; // 每个 KV 被多少个 Q 共享
  const totalQ = numHeads * dK;
  const totalKV = numKVHeads * dK;
  const sqrtDk = Math.sqrt(dK);

  // --- Step 1: 计算 Q, K, V ---
  // Q: [nSeq, totalQ], K: [nSeq, totalKV], V: [nSeq, totalKV]
  const Q = matmul(x, WQ, nSeq, dModel, totalQ);
  const K = matmul(x, WK, nSeq, dModel, totalKV);
  const V = matmul(x, WV, nSeq, dModel, totalKV);

  // --- Step 2: Attention(Q, K, V) per head ---
  // output: [nSeq, totalQ]
  const attnOut = new Float32Array(nSeq * totalQ);

  for (let h = 0; h < numHeads; h++) {
    const kvGroup = Math.floor(h / headsPerKV); // GQA: 共享哪个 KV group

    const qHeadOff = h * dK;        // Q 的 head 偏移
    const kHeadOff = kvGroup * dK;  // K 的 head 偏移
    const vHeadOff = kvGroup * dK;  // V 的 head 偏移

    // 对每个位置 i, 计算 attention(Q_i, K_all, V_all)
    for (let i = 0; i < nSeq; i++) {
      const qOff = i * totalQ + qHeadOff;

      // 计算 scores = Q_i · K_jᵀ, 对所有 j
      let maxScore = -Infinity;
      let sumExp = 0;

      // 第一次遍历: 找 max + 算 softmax 的 exp
      const scores = new Float32Array(nSeq);
      for (let j = 0; j < nSeq; j++) {
        let s = 0;
        const kOff = j * totalKV + kHeadOff;
        for (let d = 0; d < dK; d++) {
          s += Q[qOff + d] * K[kOff + d];
        }
        s /= sqrtDk;
        if (mask && mask[i * nSeq + j] !== 0) s = -Infinity;
        scores[j] = s;
        if (s > maxScore) maxScore = s;
      }

      // 第二次遍历: 计算 softmax
      for (let j = 0; j < nSeq; j++) {
        scores[j] = Math.exp(scores[j] - maxScore);
        sumExp += scores[j];
      }

      if (sumExp === 0) sumExp = 1; // 全 -∞ 时避免除零

      // 第三次遍历: 加权求和 V
      let acc = new Float32Array(dK);
      for (let j = 0; j < nSeq; j++) {
        const weight = scores[j] / sumExp;
        if (weight === 0) continue;
        const vOff = j * totalKV + vHeadOff;
        for (let d = 0; d < dK; d++) {
          acc[d] += weight * V[vOff + d];
        }
      }

      // 写入输出
      const outOff = i * totalQ + qHeadOff;
      for (let d = 0; d < dK; d++) {
        attnOut[outOff + d] = acc[d];
      }
    }
  }

  // --- Step 3: output = attnOut @ WO ---
  return matmul(attnOut, WO, nSeq, totalQ, dModel);
}

/**
 * 矩阵乘法: A (m×k) @ B (k×n) = C (m×n)
 * @param {Float32Array} A - [m, k]
 * @param {Float32Array} B - [k, n]
 * @param {number} m - 行数
 * @param {number} k - 列数 (共享维度)
 * @param {number} n - B 的列数
 * @returns {Float32Array} C [m, n]
 */
function matmul(A, B, m, k, n) {
  const C = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    const aOff = i * k;
    for (let j = 0; j < n; j++) {
      let s = 0;
      const bOff = j * k;
      for (let l = 0; l < k; l++) {
        s += A[aOff + l] * B[bOff + l];
      }
      C[i * n + j] = s;
    }
  }
  return C;
}

module.exports = { multiHeadAttention, matmul };