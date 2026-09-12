/**
 * triplane.js — Triplane 三维表示
 *
 * Triplane 用三个正交平面 (XY, XZ, YZ) 表示三维场景
 * 每个平面尺寸: [planeSize × planeSize × latentChannels]
 * 总参数量: 3 * planeSize * planeSize * latentChannels
 *
 * 优点:
 *   - 比 SDF 网格 (128³=2M) 参数更少 (3 * 64 * 64 * 8 = 76800)
 *   - 比 NeRF MLP (百万级) 可解释性更强
 *   - 支持任意分辨率的 3D 点查询
 *
 * 查询流程:
 *   1. 3D 点 (x, y, z) ∈ [-1, 1]³
 *   2. 在三个平面上分别采样: XY[2y,2x], XZ[2z,2x], YZ[2y,2z]
 *   3. 双线性插值获取特征
 *   4. 三个特征拼接 → MLP → (SDF, color)
 *
 * 典型配置:
 *   planeSize: 64
 *   latentChannels: 8
 *   MLP 隐藏层: 256
 *
 * 参数量: 3 * 64 * 64 * 8 = 76800 (≈3× 一个 256×256 的 8 通道图像)
 */

class Triplane {
  constructor(config) {
    this.planeSize = config.planeSize || 64;
    this.latentChannels = config.latentChannels || 8;

    // 三个平面
    this.planeXY = new Float32Array(this.planeSize * this.planeSize * this.latentChannels);
    this.planeXZ = new Float32Array(this.planeSize * this.planeSize * this.latentChannels);
    this.planeYZ = new Float32Array(this.planeSize * this.planeSize * this.latentChannels);

    // 初始化: 均值为零, 极小随机
    const seedStd = 0.01;
    for (let i = 0; i < this.planeXY.length; i++) {
      this.planeXY[i] = (Math.random() - 0.5) * 2 * seedStd;
      this.planeXZ[i] = (Math.random() - 0.5) * 2 * seedStd;
      this.planeYZ[i] = (Math.random() - 0.5) * 2 * seedStd;
    }

    // SDF 查询的隐藏层尺寸 (可选, 用于前向查询)
    this.mlpHidden = config.mlpHidden || 64;
    this.numMLPLayers = config.numMLPLayers || 2;

    // MLP: [latentChannels * 3] → [hidden] → ... → [hidden] → [1 + 3] (SDF + RGB)
    const inDim = this.latentChannels * 3;
    this.mlpWeights = [];
    let prevDim = inDim;
    for (let l = 0; l < this.numMLPLayers; l++) {
      this.mlpWeights.push({
        W: new Float32Array(prevDim * this.mlpHidden),
        b: new Float32Array(this.mlpHidden),
      });
      const wStd = Math.sqrt(2.0 / prevDim);
      for (let i = 0; i < prevDim * this.mlpHidden; i++) {
        this.mlpWeights[l].W[i] = (Math.random() - 0.5) * 2 * wStd;
      }
      for (let i = 0; i < this.mlpHidden; i++) this.mlpWeights[l].b[i] = 0;
      prevDim = this.mlpHidden;
    }
    // 输出层: hidden → [SDF + RGB]
    this.mlpOut = {
      W: new Float32Array(prevDim * 4),
      b: new Float32Array(4),
    };
    const wOutStd = Math.sqrt(2.0 / prevDim);
    for (let i = 0; i < prevDim * 4; i++) {
      this.mlpOut.W[i] = (Math.random() - 0.5) * 2 * wOutStd;
    }
    for (let i = 0; i < 4; i++) this.mlpOut.b[i] = 0;
  }

  /**
   * 获取平面索引
   * @param {number} py - 归一化坐标 [0, 1]
   * @param {number} px - 归一化坐标 [0, 1]
   * @returns {number} 平面索引
   */
  _planeIndex(py, px) {
    const y = Math.max(0, Math.min(this.planeSize - 1, Math.round(py * (this.planeSize - 1))));
    const x = Math.max(0, Math.min(this.planeSize - 1, Math.round(px * (this.planeSize - 1))));
    return y * this.planeSize * this.latentChannels + x * this.latentChannels;
  }

  /**
   * 双线性插值查询单个平面的特征
   * @param {Float32Array} plane - 平面数据
   * @param {number} py - 归一化 y [0, 1]
   * @param {number} px - 归一化 x [0, 1]
   * @returns {Float32Array} 特征 [latentChannels]
   */
  _bilinearSample(plane, py, px) {
    const fpy = py * (this.planeSize - 1);
    const fpx = px * (this.planeSize - 1);
    const y0 = Math.max(0, Math.min(this.planeSize - 2, Math.floor(fpy)));
    const x0 = Math.max(0, Math.min(this.planeSize - 2, Math.floor(fpx)));
    const y1 = y0 + 1;
    const x1 = x0 + 1;
    const ty = fpy - y0;
    const tx = fpx - x0;

    const out = new Float32Array(this.latentChannels);
    const i00 = y0 * this.planeSize * this.latentChannels + x0 * this.latentChannels;
    const i01 = y0 * this.planeSize * this.latentChannels + x1 * this.latentChannels;
    const i10 = y1 * this.planeSize * this.latentChannels + x0 * this.latentChannels;
    const i11 = y1 * this.planeSize * this.latentChannels + x1 * this.latentChannels;

    for (let c = 0; c < this.latentChannels; c++) {
      const v00 = plane[i00 + c];
      const v01 = plane[i01 + c];
      const v10 = plane[i10 + c];
      const v11 = plane[i11 + c];
      out[c] = (1 - ty) * (1 - tx) * v00 +
               (1 - ty) * tx * v01 +
               ty * (1 - tx) * v10 +
               ty * tx * v11;
    }
    return out;
  }

  /**
   * 查询单个 3D 点的特征
   * @param {number} x - [-1, 1]
   * @param {number} y - [-1, 1]
   * @param {number} z - [-1, 1]
   * @returns {Float32Array} 特征 [latentChannels * 3]
   */
  query(x, y, z) {
    // 转换到 [0, 1]
    const px = (x + 1) / 2;
    const py = (y + 1) / 2;
    const pz = (z + 1) / 2;

    // XY 平面: (x, y)
    const featXY = this._bilinearSample(this.planeXY, py, px);
    // XZ 平面: (x, z)
    const featXZ = this._bilinearSample(this.planeXZ, pz, px);
    // YZ 平面: (y, z)
    const featYZ = this._bilinearSample(this.planeYZ, pz, py);

    // 拼接: [latentCh * 3]
    const feat = new Float32Array(this.latentChannels * 3);
    feat.set(featXY, 0);
    feat.set(featXZ, this.latentChannels);
    feat.set(featYZ, this.latentChannels * 2);
    return feat;
  }

  /**
   * 前向传播: 特征 → SDF + RGB
   * @param {Float32Array} feat - [latentCh * 3]
   * @returns {Float32Array} [1 + 3] = [sdf, r, g, b]
   */
  forward(feat) {
    let h = feat;
    for (const layer of this.mlpWeights) {
      const newH = new Float32Array(this.mlpHidden);
      for (let j = 0; j < this.mlpHidden; j++) {
        let s = layer.b[j];
        for (let k = 0; k < h.length; k++) {
          s += h[k] * layer.W[k * this.mlpHidden + j];
        }
        // GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        newH[j] = 0.5 * s * (1 + Math.tanh(0.79788456 * (s + 0.044715 * s * s * s)));
      }
      h = newH;
    }
    // 输出层: [SDF, R, G, B]
    const out = new Float32Array(4);
    for (let j = 0; j < 4; j++) {
      let s = this.mlpOut.b[j];
      for (let k = 0; k < h.length; k++) {
        s += h[k] * this.mlpOut.W[k * 4 + j];
      }
      out[j] = j === 0 ? s : Math.max(0, Math.min(1, s)); // SDF 不裁剪, RGB 裁剪到 [0,1]
    }
    return out;
  }

  /**
   * 查询 3D 点的 SDF + RGB
   * @param {number} x - [-1, 1]
   * @param {number} y - [-1, 1]
   * @param {number} z - [-1, 1]
   * @returns {Float32Array} [sdf, r, g, b]
   */
  querySDF(x, y, z) {
    const feat = this.query(x, y, z);
    return this.forward(feat);
  }

  /**
   * 体积渲染: 沿光线采样
   * @param {Float32Array} rayOrigin - [3] 光线原点
   * @param {Float32Array} rayDir - [3] 光线方向 (未归一化)
   * @param {number} numSamples - 采样数
   * @returns {{color: Float32Array, depth: number}} 渲染结果
   */
  volumeRender(rayOrigin, rayDir, numSamples = 32) {
    // 归一化方向
    const dirLen = Math.sqrt(rayDir[0] ** 2 + rayDir[1] ** 2 + rayDir[2] ** 2);
    const dir = [rayDir[0] / dirLen, rayDir[1] / dirLen, rayDir[2] / dirLen];

    // 沿光线等间距采样
    const tStart = -3;
    const tEnd = 3;
    const dt = (tEnd - tStart) / numSamples;

    let color = new Float32Array([0, 0, 0]);
    let depth = 0;
    let accumulatedT = 0; // 累积透明度 (1 - α)
    let totalWeight = 0;
    const sigmoidK = 10;

    for (let i = 0; i < numSamples; i++) {
      const t = tStart + i * dt + dt / 2;
      const x = rayOrigin[0] + t * dir[0];
      const y = rayOrigin[1] + t * dir[1];
      const z = rayOrigin[2] + t * dir[2];

      // 只查询 [-1, 1]³ 范围
      if (Math.abs(x) > 1 || Math.abs(y) > 1 || Math.abs(z) > 1) continue;

      const result = this.querySDF(x, y, z);
      const sdf = result[0];

      // SDF → 密度: σ = σ_sigmoid(K * (-sdf))
      const sigma = 1 / (1 + Math.exp(-sigmoidK * (-sdf)));
      const alpha = 1 - Math.exp(-sigma * dt);

      const rayColor = [result[1], result[2], result[3]];
      const weight = alpha * accumulatedT;

      color[0] += weight * rayColor[0];
      color[1] += weight * rayColor[1];
      color[2] += weight * rayColor[2];

      totalWeight += weight;
      accumulatedT *= (1 - alpha);
      depth += weight * t;
    }

    // 归一化深度
    if (totalWeight > 0) depth = depth / totalWeight;
    else depth = 0;

    return { color, depth };
  }

  /**
   * 简单相机渲染
   * @param {number} imgH - 图像高度
   * @param {number} imgW - 图像宽度
   * @param {number} fov - 视场角 (弧度)
   * @param {number} distance - 相机距离
   * @param {number} numSamples - 每像素采样数
   * @returns {Float32Array} [imgH, imgW, 3] RGB
   */
  renderCamera(imgH, imgW, fov, distance = 3, numSamples = 16) {
    const halfH = 0.5 * imgH * Math.tan(fov / 2);
    const halfW = 0.5 * imgW * Math.tan(fov / 2);

    const colorImg = new Float32Array(imgH * imgW * 3);

    for (let y = 0; y < imgH; y++) {
      for (let x = 0; x < imgW; x++) {
        // 屏幕坐标
        const sx = (x - (imgW - 1) / 2) / imgW * imgW * 0.5 / halfW;
        const sy = (y - (imgH - 1) / 2) / imgH * imgH * 0.5 / halfH;

        // 光线方向 (相机看向 -Z, X 右, Y 上)
        const rayDir = [sx, -sy, -1];
        const rayOrigin = [0, 0, distance];

        const result = this.volumeRender(rayOrigin, rayDir, numSamples);
        const idx = (y * imgW + x) * 3;
        colorImg[idx] = result.color[0];
        colorImg[idx + 1] = result.color[1];
        colorImg[idx + 2] = result.color[2];
      }
    }
    return colorImg;
  }

  /**
   * 获取 SDF 等值面 (marching cubes 简化版: 2D 切片)
   * @param {number} ySlice - y 坐标
   * @param {number} resolution - 分辨率
   * @returns {Float32Array} [resolution, resolution] 二值化等值面
   */
  getIsoSurface2D(ySlice = 0, resolution = 16) {
    const surface = new Float32Array(resolution * resolution);
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = -1 + 2 * i / (resolution - 1);
        const z = -1 + 2 * j / (resolution - 1);
        const result = this.querySDF(x, ySlice, z);
        surface[i * resolution + j] = result[0] < 0 ? 1 : 0;
      }
    }
    return surface;
  }

  countParams() {
    let total = this.planeXY.length + this.planeXZ.length + this.planeYZ.length;
    for (const layer of this.mlpWeights) {
      total += layer.W.length + layer.b.length;
    }
    total += this.mlpOut.W.length + this.mlpOut.b.length;
    return total;
  }
}

module.exports = { Triplane };