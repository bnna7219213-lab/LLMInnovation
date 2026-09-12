/**
 * diffusion_schedule.js — Diffusion 时间步调度器
 *
 * 公式 (DDPM):
 *   Forward:  q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
 *   Closed form: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
 *   Reverse:  x_{t-1} = 1/√α_t · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
 *
 * 调度策略:
 *   1. Linear: β_t = β_min + t/T * (β_max - β_min)
 *   2. Cosine: ᾱ_t = cos(π/2 · t/T)²
 *   3. LCM:    稀疏步采样, 用于快速推理
 */

/**
 * 生成 β 调度
 * @param {string} schedule - 'linear' | 'cosine'
 * @param {number} T - 总步数
 * @param {object} params - 调度参数
 * @returns {{betas: Float32Array, alpha: Float32Array, alphaBar: Float32Array, sigma: Float32Array}}
 */
function generateSchedule(schedule = 'cosine', T = 1000, params = {}) {
  const betas = new Float32Array(T);
  const alpha = new Float32Array(T);
  const alphaBar = new Float32Array(T);
  const sigma = new Float32Array(T);

  if (schedule === 'cosine') {
    // Cosine schedule: ᾱ_t = cos(π/2 · t/T)²
    // 这是 DDIM / Stable Diffusion 使用的调度
    const s = params.s || 0.008; // cosine offset
    for (let t = 0; t < T; t++) {
      const cosAngle = Math.cos(Math.PI / 2 * (t + s) / (T + s));
      alphaBar[t] = cosAngle * cosAngle;
      // β_t 从 ᾱ 反推: ᾱ_t = α_1 * α_2 * ... * α_t, α_t = 1 - β_t
      if (t === 0) {
        alpha[t] = alphaBar[0];
        betas[0] = 1 - alphaBar[0];
      } else {
        alpha[t] = alphaBar[t] / alphaBar[t - 1];
        betas[t] = 1 - alpha[t];
      }
    }
  } else {
    // Linear schedule: β_t = β_min + t/T * (β_max - β_min)
    const betaMin = params.betaMin || 0.0001;
    const betaMax = params.betaMax || 0.02;
    for (let t = 0; t < T; t++) {
      betas[t] = betaMin + (t / (T - 1)) * (betaMax - betaMin);
    }
    // 计算 ᾱ_t = α_1 * α_2 * ... * α_t
    let prod = 1;
    for (let t = 0; t < T; t++) {
      alpha[t] = 1 - betas[t];
      prod *= alpha[t];
      alphaBar[t] = prod;
    }
  }

  // σ_t = β_t (DDPM 的简化版)
  for (let t = 0; t < T; t++) {
    sigma[t] = betas[t];
  }

  return { betas, alpha, alphaBar, sigma };
}

/**
 * 前向过程: 在时间步 t 加噪
 * @param {Float32Array} x0 - 原始干净图像 [batch * channels * height * width]
 * @param {Float32Array} tArray - 时间步 [batch]
 * @param {Float32Array} noise - 高斯噪声 [batch * channels * height * width]
 * @param {Float32Array} alphaBar - 累积 α [T]
 * @returns {Float32Array[]} [batch] 每个元素是 [channels * height * width]
 */
function addNoise(x0, tArray, noise, alphaBar) {
  const batch = tArray.length;
  const nPixels = x0.length / batch;
  const results = [];

  for (let b = 0; b < batch; b++) {
    const xt = new Float32Array(nPixels);
    const ab = Math.sqrt(alphaBar[tArray[b]]);
    const oneMinusAb = Math.sqrt(1 - alphaBar[tArray[b]]);
    for (let i = 0; i < nPixels; i++) {
      xt[i] = ab * x0[b * nPixels + i] + oneMinusAb * noise[b * nPixels + i];
    }
    results.push(xt);
  }

  return results;
}

/**
 * DDPM 反向采样: 从 x_t 到 x_{t-1}
 * @param {Float32Array} x_t - 当前时间步图像 [nPixels]
 * @param {number} t - 当前时间步
 * @param {Float32Array} epsilonPred - UNet 预测的噪声 [nPixels]
 * @param {Float32Array} alpha - α [T]
 * @param {Float32Array} alphaBar - ᾱ [T]
 * @param {Float32Array} sigma - σ [T]
 * @param {Float32Array} z - 新噪声 [nPixels] (t=0 时为全 0)
 * @param {number} nPixels
 * @returns {Float32Array} x_{t-1} [nPixels]
 */
function reverseStep(x_t, t, epsilonPred, alpha, alphaBar, sigma, z, nPixels) {
  const sqrtAlpha = Math.sqrt(alpha[t]);
  const oneMinusAb = 1 - alphaBar[t];

  // 特殊处理 t=0: α_0 接近 1, ᾱ_0 接近 1, 用简化公式
  // x_{-1} = x0_pred (裁剪到 [-1,1])
  if (t === 0) {
    const x0Pred = new Float32Array(nPixels);
    for (let i = 0; i < nPixels; i++) {
      x0Pred[i] = (x_t[i] - epsilonPred[i]) / sqrtAlpha;
      x0Pred[i] = Math.max(-1, Math.min(1, x0Pred[i]));
    }
    return x0Pred;
  }

  const sqrtOneMinusAb = Math.sqrt(oneMinusAb);

  // x_{t-1} = 1/√α_t * (x_t - (1-ᾱ_t)/√(1-ᾱ_t) * ε_θ) + σ_t * z
  const x0Pred = new Float32Array(nPixels);
  for (let i = 0; i < nPixels; i++) {
    x0Pred[i] = (x_t[i] - oneMinusAb / sqrtOneMinusAb * epsilonPred[i]) / sqrtAlpha;
    x0Pred[i] = Math.max(-1, Math.min(1, x0Pred[i]));
  }

  const x_tm1 = new Float32Array(nPixels);
  for (let i = 0; i < nPixels; i++) {
    x_tm1[i] = x0Pred[i] + sigma[t] * z[i];
  }

  return x_tm1;
}

/**
 * DDIM 反向采样 (确定性, 可跳过步)
 * @param {Float32Array} x_t - 当前时间步图像
 * @param {number} t - 当前时间步
 * @param {number} tPrev - 下一个时间步
 * @param {Float32Array} epsilonPred - UNet 预测的噪声
 * @param {Float32Array} alphaBar - ᾱ [T]
 * @param {number} eta - 噪声系数 (0 = 确定性, 1 = DDPM)
 * @param {number} nPixels
 * @returns {Float32Array} x_{t-1}
 */
function ddimStep(x_t, t, tPrev, epsilonPred, alphaBar, eta = 0, nPixels) {
  const abT = alphaBar[t];
  const abTMinus1 = alphaBar[tPrev];

  // x0_pred = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
  const sqrtAbT = Math.sqrt(abT);
  const sqrtOneMinusAbT = Math.sqrt(1 - abT);

  const x0Pred = new Float32Array(nPixels);
  for (let i = 0; i < nPixels; i++) {
    x0Pred[i] = (x_t[i] - sqrtOneMinusAbT * epsilonPred[i]) / sqrtAbT;
    x0Pred[i] = Math.max(-1, Math.min(1, x0Pred[i]));
  }

  // σ_t = η * √(1-ᾱ_{t-1})/√(1-ᾱ_t) * √(1-ᾱ_t/ᾱ_{t-1})
  // 简化: σ_t = η * √(1-ᾱ_{t-1}) + √(ᾱ_{t-1}/ᾱ_t * (1-ᾱ_t))
  // 更简化: DDIM 直接计算
  const sigma = eta * Math.sqrt((1 - abTMinus1) / (1 - abT) * (1 - abT / abTMinus1));

  // x_{t-1} = √ᾱ_{t-1} * x0_pred + √(1-ᾱ_{t-1} - σ²) * ε_θ + σ * z
  const sqrtAbTm1 = Math.sqrt(abTMinus1);
  const coefficient = Math.sqrt(Math.max(0, 1 - abTMinus1 - sigma * sigma));

  return { x0Pred, sigma, sqrtAbTm1, coefficient };
}

module.exports = {
  generateSchedule,
  addNoise,
  reverseStep,
  ddimStep,
};