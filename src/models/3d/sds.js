/**
 * sds.js — Score Distillation Sampling (SDS)
 *
 * SDS 公式 (来自 Score-Based Diffusion for 3D 论文):
 *
 *   1. 从 3D 模型渲染 2D 图像:
 *      x_0 = render(θ, view)        // θ = 3D 模型参数 (如 triplane)
 *
 *   2. 前向加噪 (使用文本 prompt 对应的时间步 t):
 *      z_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
 *
 *   3. UNet 预测噪声:
 *      ε_pred = ε_θ(z_t, t, text_emb)
 *
 *   4. 权重函数 (平衡不同时间步的贡献):
 *      w(t) = (1-ᾱ_t) · √ᾱ_t / (1-α_t)
 *
 *   5. SDS 得分:
 *      g_t = ε_pred - ε
 *
 *   6. 梯度:
 *      ∂L/∂θ = w(t) · 2 · g_t · √ᾱ_t · ∂x_0/∂θ
 *      (其中 ∂x_0/∂θ 通过体积渲染的微分链式法则获得)
 *
 *   简化版本 (无自动微分):
 *     - 用有限差分近似 ∂x_0/∂θ
 *     - 或用随机投影 (SVRG) 加速
 *
 * 参数:
 *   t_min: 最小时间步 (通常 50)
 *   t_max: 最大时间步 (通常 2000, 实际 T=1000)
 *   view: 相机参数 (视角随机采样)
 *
 * 参数量估算 (单次 SDS 计算):
 *   - 渲染: 图像大小 × 采样数 × 3
 *   - UNet 前向: N_pixels × text_len 的 cross-attention
 *   - 梯度: 与模型参数量相同
 */

class SDS {
  constructor(config) {
    this.tMin = config.tMin || 50;
    this.tMax = config.tMax || 1000;
    this.imgSize = config.imgSize || 32;
    this.numSamples = config.numSamples || 16;
    this.fov = config.fov || Math.PI / 4;
    this.distance = config.distance || 3;
  }

  /**
   * 权重函数 w(t) = (1-ᾱ_t) · √ᾱ_t / (1-α_t)
   * @param {number} t - 时间步
   * @param {Float32Array} alpha - α [T+1]
   * @param {Float32Array} alphaBar - ᾱ [T+1]
   * @returns {number} 权重
   */
  _weight(t, alpha, alphaBar) {
    const alpha_t = alpha[t];
    const alphaBar_t = alphaBar[t];
    if (alpha_t >= 1) return 0; // 避免除零
    return (1 - alphaBar_t) * Math.sqrt(alphaBar_t) / (1 - alpha_t);
  }

  /**
   * 单步 SDS 前向: 计算损失和梯度信号
   * @param {Triplane} triplane - 3D 模型
   * @param {number} t - 时间步
   * @param {Float32Array} textEmb - 文本嵌入
   * @param {Float32Array} alpha - α [T+1]
   * @param {Float32Array} alphaBar - ᾱ [T+1]
   * @param {number[]} viewAngles - 视角 [azimuth, elevation] 弧度
   * @param {Float32Array} randNoise - 随机噪声 [imgSize*imgSize*3]
   * @returns {{loss: number, gradSignal: Float32Array, epsilon: Float32Array, x0: Float32Array}}
   */
  step(triplane, t, textEmb, alpha, alphaBar, viewAngles, randNoise) {
    // 1. 从 triplane 渲染 2D 图像 x_0
    const x0 = this._renderView(triplane, viewAngles);

    const nPixels = this.imgSize * this.imgSize * 3;

    // 2. 前向加噪: z_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    const sqrtAlphaBar = Math.sqrt(alphaBar[t]);
    const sqrtOneMinusAlphaBar = Math.sqrt(1 - alphaBar[t]);

    const zt = new Float32Array(nPixels);
    const epsilon = randNoise;
    for (let i = 0; i < nPixels; i++) {
      zt[i] = sqrtAlphaBar * x0[i] + sqrtOneMinusAlphaBar * epsilon[i];
    }

    return { zt, epsilon, x0, sqrtAlphaBar, sqrtOneMinusAlphaBar };
  }

  /**
   * 从相机视角渲染 triplane
   * @param {Triplane} triplane
   * @param {number[]} viewAngles - [azimuth, elevation] 弧度
   * @returns {Float32Array} [imgSize, imgSize, 3]
   */
  _renderView(triplane, viewAngles) {
    const [azimuth, elevation] = viewAngles;

    // 简单相机: 旋转后渲染 (简化: 使用 triplane 的相机渲染)
    // 实际 SDS 需要对 triplane 应用视角变换
    // 这里做简化: 直接用 triplane.renderCamera 并偏移视角
    const imgH = this.imgSize;
    const imgW = this.imgSize;

    // 创建临时偏移版本的 triplane 来模拟视角
    // (实际实现需要修改渲染管线, 这里用 triplane 自带的相机渲染)
    const colorImg = triplane.renderCamera(imgH, imgW, this.fov, this.distance, this.numSamples);

    // 对颜色进行视角偏移 (简化版: 应用旋转矩阵到射线)
    const rotated = new Float32Array(colorImg.length);
    for (let i = 0; i < colorImg.length; i++) {
      rotated[i] = colorImg[i]; // 简化: 直接复制
    }

    return rotated;
  }

  /**
   * 完整 SDS 训练步:
   * 1. 采样随机视角和时间步
   * 2. 渲染图像
   * 3. 前向加噪
   * 4. UNet 预测
   * 5. 计算损失
   *
   * @param {Triplane} triplane
   * @param {UNet} unet
   * @param {Float32Array} textEmb
   * @param {Float32Array} alpha
   * @param {Float32Array} alphaBar
   * @returns {{loss: number, view: number[], t: number}}
   */
  trainStep(triplane, unet, textEmb, alpha, alphaBar) {
    // 随机视角
    const azimuth = Math.random() * Math.PI * 2;
    const elevation = (Math.random() - 0.5) * Math.PI * 0.5;
    const viewAngles = [azimuth, elevation];

    // 随机时间步 (偏向较晚期, 即图像清晰度较高)
    const t = this._sampleT();

    // 随机噪声
    const nPixels = this.imgSize * this.imgSize * 3;
    const epsilon = new Float32Array(nPixels);
    for (let i = 0; i < nPixels; i++) {
      epsilon[i] = this._gauss();
    }

    // 前向
    const { zt } = this.step(triplane, t, textEmb, alpha, alphaBar, viewAngles, epsilon);

    // UNet 预测噪声
    const ztFlat = [zt]; // batch=1, UNet 期望 [nPixels]
    const epsilonPred = unet.forward([zt], [t], [textEmb], 1)[0];

    // 计算 SDS 损失: L = 0.5 * w(t) * ||ε_pred - ε||²
    const w = this._weight(t, alpha, alphaBar);
    let loss = 0;
    for (let i = 0; i < nPixels; i++) {
      const diff = epsilonPred[i] - epsilon[i];
      loss += diff * diff;
    }
    loss = 0.5 * w * loss;

    return { loss, view: viewAngles, t, epsilonPred, epsilon };
  }

  /**
   * 有限差分梯度估计 (对单个平面像素)
   * @param {Triplane} triplane
   * @param {number} planeIdx - 平面索引 0/1/2
   * @param {number} pixelIdx - 平面内像素索引
   * @param {number} channel - 通道索引
   * @param {number} eps - 扰动大小
   * @returns {number} 梯度
   */
  finiteDiffGradient(triplane, planeIdx, pixelIdx, channel, eps, unet, textEmb, alpha, alphaBar) {
    const planes = [triplane.planeXY, triplane.planeXZ, triplane.planeYZ];
    const target = planes[planeIdx];
    const oldVal = target[pixelIdx * triplane.latentChannels + channel];

    // 正向
    target[pixelIdx * triplane.latentChannels + channel] = oldVal + eps;
    const { loss: lossPlus } = this.trainStep(triplane, unet, textEmb, alpha, alphaBar);

    // 反向
    target[pixelIdx * triplane.latentChannels + channel] = oldVal - eps;
    const { loss: lossMinus } = this.trainStep(triplane, unet, textEmb, alpha, alphaBar);

    // 恢复
    target[pixelIdx * triplane.latentChannels + channel] = oldVal;

    return (lossPlus - lossMinus) / (2 * eps);
  }

  /**
   * 随机时间步采样 (偏向后期, 图像更清晰)
   * @returns {number} 时间步
   */
  _sampleT() {
    // 使用 sqrt 分布, 偏向 tMin 附近
    const u = Math.random();
    const t = Math.round(Math.pow(
      this.tMin + (this.tMax - this.tMin) * u * u, 1
    ));
    return Math.max(this.tMin, Math.min(this.tMax, t));
  }

  /**
   * 标准正态采样
   */
  _gauss() {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

module.exports = { SDS };