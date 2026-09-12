/**
 * vae.js — Variational Autoencoder (VAE)
 *
 * 用于将图像压缩到低维潜空间, 然后 UNet 在潜空间中操作
 *
 * Encoder: 图像 → (μ, σ) → 潜变量 z = μ + σ * ε
 * Decoder: 潜变量 z → 图像
 *
 * 参数 (以 Tiny VAE 为例):
 *   inputChannels: 3 (RGB)
 *   latentChannels: 4
 *   inputSize: 64 (高度和宽度)
 *   latentSize: 8 (压缩 8x)
 *   blockChannels: [64, 128, 256, 512]
 *
 * 典型 SD VAE: 256×256 → 32×32×4 = 4096 维 → 从 196608 压到 4096 (48x)
 */

class VAE {
  constructor(config) {
    this.inputChannels = config.inputChannels || 3;
    this.latentChannels = config.latentChannels || 4;
    this.inputSize = config.inputSize || 64;
    this.latentSize = config.latentSize || 8;
    this.blockChannels = config.blockChannels || [32, 64, 128];

    // 初始化权重
    this._initWeights();
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

  _initWeights() {
    // Encoder 层
    this.encConv = [];
    let prevCh = this.inputChannels;
    let size = this.inputSize;
    for (let i = 0; i < this.blockChannels.length; i++) {
      const ch = this.blockChannels[i];
      this.encConv.push({
        W: this._randn(3 * 3 * prevCh * ch, 0, 0.02 / Math.sqrt(3 * 3 * prevCh)),
        size: size,
        channels: ch,
      });
      prevCh = ch;
      size = Math.floor(size / 2); // stride=2 下采样
    }
    this.encSize = size;
    this.encLatentCh = prevCh;

    // Decoder 层 (对称)
    this.decConv = [];
    let curSize = size;
    let curCh = prevCh;
    for (let i = this.blockChannels.length - 1; i >= 0; i--) {
      const ch = this.blockChannels[i];
      this.decConv.push({
        W: this._randn(3 * 3 * curCh * ch, 0, 0.02 / Math.sqrt(3 * 3 * curCh)),
        size: curSize * 2, // 上采样
        channels: ch,
      });
      curSize = curSize * 2;
      curCh = ch;
    }
    this.decConv.push({
      W: this._randn(3 * 3 * curCh * this.inputChannels, 0, 0.02 / Math.sqrt(3 * 3 * curCh)),
      size: this.inputSize,
      channels: this.inputChannels,
    });

    // 潜空间映射: 从 [encSize*encSize*encLatentCh] → [latentSize*latentSize*latentChannels]
    const inDim = this.encSize * this.encSize * this.encLatentCh;
    const outDim = this.latentSize * this.latentSize * this.latentChannels;
    this.W_mu = this._randn(inDim * outDim, 0, 0.02 / Math.sqrt(inDim));
    this.W_sigma = this._randn(inDim * outDim, 0, 0.02 / Math.sqrt(inDim));
  }

  /**
   * 简化版卷积 (2D, stride=2 下采样, padding=1)
   * @param {Float32Array} x - [H, W, C]
   * @param {Float32Array} W - [3, 3, C_in, C_out]
   * @param {number} H - 输入高度
   * @param {number} W_w - 输入宽度
   * @param {number} C_in
   * @param {number} C_out
   * @returns {Float32Array} [H/2, W/2, C_out]
   */
  _conv2d(x, W, H, W_w, C_in, C_out, stride = 2) {
    const OH = Math.floor((H - 2) / stride) + 1;
    const OW = Math.floor((W_w - 2) / stride) + 1;
    const out = new Float32Array(OH * OW * C_out);

    for (let oh = 0; oh < OH; oh++) {
      for (let ow = 0; ow < OW; ow++) {
        for (let c = 0; c < C_out; c++) {
          let s = 0;
          for (let kh = 0; kh < 3; kh++) {
            for (let kw = 0; kw < 3; kw++) {
              const srcH = oh * stride + kh - 1;
              const srcW = ow * stride + kw - 1;
              if (srcH >= 0 && srcH < H && srcW >= 0 && srcW < W_w) {
                for (let cin = 0; cin < C_in; cin++) {
                  s += x[(srcH * W_w + srcW) * C_in + cin] * W[(kh * 3 + kw) * C_in * C_out + cin * C_out + c];
                }
              }
            }
          }
          out[(oh * OW + ow) * C_out + c] = s;
        }
      }
    }
    return { data: out, H: OH, W: OW, C: C_out };
  }

  /**
   * 简化版转置卷积 (上采样)
   * @param {Float32Array} x - [H, W, C_in]
   * @param {Float32Array} W - [3, 3, C_in, C_out]
   * @param {number} OH - 输出高度
   * @param {number} OW - 输出宽度
   * @returns {Float32Array} [OH, OW, C_out]
   */
  _convTranspose(x, W, H, W_w, C_in, C_out, OH, OW, stride = 2) {
    const out = new Float32Array(OH * OW * C_out);

    for (let oh = 0; oh < OH; oh++) {
      for (let ow = 0; ow < OW; ow++) {
        for (let c = 0; c < C_out; c++) {
          let s = 0;
          for (let kh = 0; kh < 3; kh++) {
            for (let kw = 0; kw < 3; kw++) {
              const srcH = Math.floor((oh - kh + 1) / stride);
              const srcW = Math.floor((ow - kw + 1) / stride);
              if (srcH >= 0 && srcH < H && srcW >= 0 && srcW < W_w) {
                for (let cin = 0; cin < C_in; cin++) {
                  s += x[(srcH * W_w + srcW) * C_in + cin] * W[(kh * 3 + kw) * C_in * C_out + cin * C_out + c];
                }
              }
            }
          }
          out[(oh * OW + ow) * C_out + c] = s;
        }
      }
    }
    return { data: out, H: OH, W: OW, C: C_out };
  }

  /**
   * Encoder: 图像 → (μ, σ)
   * @param {Float32Array} x - [batch, inputSize, inputSize, inputChannels]
   * @param {number} batch
   * @returns {{mu: Float32Array, sigma: Float32Array}} [batch, latentSize, latentSize, latentChannels]
   */
  encode(x, batch) {
    let curData = x;
    let curH = this.inputSize;
    let curW = this.inputSize;
    let curC = this.inputChannels;

    // 下采样
    for (let i = 0; i < this.blockChannels.length; i++) {
      const layer = this.encConv[i];
      const conv = this._conv2d(curData, layer.W, curH, curW, curC, layer.channels);
      curData = conv.data;
      curH = conv.H;
      curW = conv.W;
      curC = layer.channels;
    }

    // 计算 μ 和 σ
    const mu = new Float32Array(batch * this.latentSize * this.latentSize * this.latentChannels);
    const sigma = new Float32Array(batch * this.latentSize * this.latentSize * this.latentChannels);
    const inDim = curH * curW * curC;
    const outDim = this.latentSize * this.latentSize * this.latentChannels;

    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < outDim; i++) {
        let s_mu = 0, s_sig = 0;
        const bOff = b * inDim;
        for (let j = 0; j < inDim; j++) {
          const wIdx = j * outDim + i;
          s_mu += curData[bOff + j] * (this.W_mu[wIdx] || 0);
          s_sig += curData[bOff + j] * (this.W_sigma[wIdx] || 0);
        }
        mu[b * outDim + i] = s_mu;
        sigma[b * outDim + i] = s_sig;
      }
    }

    return { mu, sigma };
  }

  /**
   * Decoder: 潜变量 → 图像
   * @param {Float32Array} z - [batch, latentSize, latentSize, latentChannels]
   * @param {number} batch
   * @returns {Float32Array} [batch, inputSize, inputSize, inputChannels]
   */
  decode(z, batch) {
    // 将潜变量映射回全连接空间
    const inDim = this.latentSize * this.latentSize * this.latentChannels;
    const outDim = this.encSize * this.encSize * this.encLatentCh;
    let curData = new Float32Array(batch * outDim);

    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < outDim; i++) {
        let s = 0;
        for (let j = 0; j < inDim; j++) {
          const wIdx = j * outDim + i;
          s += z[b * inDim + j] * (this.W_mu[wIdx] || 0);
        }
        curData[b * outDim + i] = s;
      }
    }

    let curH = this.encSize;
    let curW = this.encSize;
    let curC = this.encLatentCh;

    // 上采样
    for (const layer of this.decConv) {
      const conv = this._convTranspose(curData, layer.W, curH, curW, curC, layer.channels,
        layer.size, layer.size, 2);
      curData = conv.data;
      curH = layer.size;
      curW = layer.size;
      curC = layer.channels;
    }

    // 输出裁剪到 [-1, 1]
    for (let i = 0; i < curData.length; i++) {
      curData[i] = Math.max(-1, Math.min(1, curData[i]));
    }

    return curData;
  }

  /**
   * 重参数化技巧: z = μ + σ * ε
   * @param {Float32Array} mu
   * @param {Float32Array} sigma
   * @param {Float32Array} eps - 高斯噪声
   * @returns {Float32Array} z
   */
  reparameterize(mu, sigma, eps) {
    const z = new Float32Array(mu.length);
    for (let i = 0; i < mu.length; i++) {
      z[i] = mu[i] + sigma[i] * eps[i];
    }
    return z;
  }

  getLatentShape() {
    return [this.latentSize, this.latentSize, this.latentChannels];
  }
}

module.exports = { VAE };