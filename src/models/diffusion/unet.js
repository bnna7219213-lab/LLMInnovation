/**
 * unet.js — UNet 去噪模型
 *
 * 标准 UNet: skip 在 conv 前保存 (输入尺寸), up block 合并对应 skip
 *
 * Down block:  skip(input) → conv(stride=2) → 半尺寸
 * Up block:    up(stride=2) → 合并 skip → conv
 *
 * 空间演进 (latentSize=8, blockChannels=[16, 32]):
 *   Down0: 8x8x4 → skip(8x8x4) → conv → 4x4x16
 *   Down1: 4x4x16 → skip(4x4x16) → conv → 2x2x32
 *   Middle: 2x2x32
 *   Up0: 2x2x32 → up → 4x4x32 + skip(4x4x16) → conv → 4x4x16
 *   Up1: 4x4x16 → up → 8x8x16 + skip(8x8x4) → conv → 8x8x4
 */

class UNet {
  constructor(config) {
    this.latentChannels = config.latentChannels || 4;
    this.latentSize = config.latentSize || 8;
    this.textEmbedDim = config.textEmbedDim || 64;
    this.textTokenLen = config.textTokenLen || 16;
    this.blockChannels = config.blockChannels || [32, 64, 128];
    this.numHeads = config.numHeads || 4;

    this.timeEmbedDim = this.blockChannels[0] * 4;
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
    // Down blocks: 每层保存 skip (输入尺寸), 然后 conv(stride=2)
    this.downBlocks = [];
    let prevCh = this.latentChannels;
    let size = this.latentSize;
    for (let i = 0; i < this.blockChannels.length; i++) {
      const ch = this.blockChannels[i];
      const dK = Math.max(1, Math.floor(ch / this.numHeads));
      // 1x1 conv 匹配通道数 (如果 skip 通道 != ch)
      const skipCh = prevCh;
      const W_skip = skipCh !== ch ? this._randn(1 * 1 * skipCh * ch, 0, 0.02) : null;
      this.downBlocks.push({
        W_conv: this._randn(3 * 3 * prevCh * ch, 0, 0.02 / Math.sqrt(3 * 3 * prevCh)),
        W_time_proj: this._randn(this.timeEmbedDim * ch, 0, 0.02),
        W_Q: this._randn(ch * dK, 0, 0.02 / Math.sqrt(ch)),
        W_K: this._randn(this.textEmbedDim * dK, 0, 0.02 / Math.sqrt(this.textEmbedDim)),
        W_V: this._randn(this.textEmbedDim * ch, 0, 0.02 / Math.sqrt(this.textEmbedDim)),
        W_attn_out: this._randn(dK * ch, 0, 0.02 / Math.sqrt(dK)),
        W_skip,
        size,
        ch,
        dK,
      });
      prevCh = ch;
      size = Math.floor(size / 2);
    }

    // Middle
    const midDk = Math.max(1, Math.floor(prevCh / this.numHeads));
    this.middleBlock = {
      W_conv: this._randn(3 * 3 * prevCh * prevCh, 0, 0.02 / Math.sqrt(3 * 3 * prevCh)),
      W_time_proj: this._randn(this.timeEmbedDim * prevCh, 0, 0.02),
      W_Q: this._randn(prevCh * midDk, 0, 0.02 / Math.sqrt(prevCh)),
      W_K: this._randn(this.textEmbedDim * midDk, 0, 0.02 / Math.sqrt(this.textEmbedDim)),
      W_V: this._randn(this.textEmbedDim * prevCh, 0, 0.02 / Math.sqrt(this.textEmbedDim)),
      W_attn_out: this._randn(midDk * prevCh, 0, 0.02 / Math.sqrt(midDk)),
      size,
      ch: prevCh,
      dK: midDk,
    };

    // Up blocks: up(stride=2) → 合并 skip → conv
    this.upBlocks = [];
    let curSize = size;
    for (let i = this.blockChannels.length - 1; i >= 0; i--) {
      const ch = this.blockChannels[i];
      const dK = Math.max(1, Math.floor(ch / this.numHeads));
      // skip 通道 = 对应 down block 的输入通道
      const skipIdx = this.blockChannels.length - 1 - (this.blockChannels.length - 1 - i);
      const skipBlock = this.downBlocks[skipIdx];
      const skipCh = skipBlock ? skipBlock.size === this.latentSize ? this.latentChannels : (i === 0 ? this.latentChannels : this.blockChannels[i + 1]) : ch;
      this.upBlocks.push({
        W_up: this._randn(2 * 2 * prevCh * ch, 0, 0.02),
        W_skip: skipCh !== ch ? this._randn(1 * 1 * skipCh * ch, 0, 0.02) : null,
        W_time_proj: this._randn(this.timeEmbedDim * ch, 0, 0.02),
        W_Q: this._randn(ch * dK, 0, 0.02 / Math.sqrt(ch)),
        W_K: this._randn(this.textEmbedDim * dK, 0, 0.02 / Math.sqrt(this.textEmbedDim)),
        W_V: this._randn(this.textEmbedDim * ch, 0, 0.02 / Math.sqrt(this.textEmbedDim)),
        W_attn_out: this._randn(dK * ch, 0, 0.02 / Math.sqrt(dK)),
        // merged = up + skip, 通道数 = ch (相加后仍是 ch 通道)
        W_conv: this._randn(3 * 3 * ch * ch, 0, 0.02 / Math.sqrt(3 * 3 * ch)),
        size: curSize * 2,
        ch,
        dK,
      });
      prevCh = ch;
      curSize = curSize * 2;
    }

    this.W_out = this._randn(3 * 3 * prevCh * this.latentChannels, 0, 0.01);
  }

  _timeEmbed(t, dim) {
    const emb = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      const freq = Math.exp(-Math.log(10000) * 2 * (i % 2) / dim);
      const angle = t * freq;
      emb[i] = (i % 2 === 0) ? Math.cos(angle) : Math.sin(angle);
    }
    return emb;
  }

  _silu(x) { return 1 / (1 + Math.exp(-x)); }

  _conv2d(x, W, H, W_w, C_in, C_out) {
    const out = new Float32Array(H * W_w * C_out);
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W_w; w++) {
        for (let c = 0; c < C_out; c++) {
          let s = 0;
          for (let kh = 0; kh < 3; kh++) {
            for (let kw = 0; kw < 3; kw++) {
              const sh = h + kh - 1;
              const sw = w + kw - 1;
              if (sh >= 0 && sh < H && sw >= 0 && sw < W_w) {
                for (let ci = 0; ci < C_in; ci++) {
                  s += x[(sh * W_w + sw) * C_in + ci] * W[(kh * 3 + kw) * C_in * C_out + ci * C_out + c];
                }
              }
            }
          }
          out[(h * W_w + w) * C_out + c] = s * this._silu(s);
        }
      }
    }
    return { data: out, H, W: W_w, C: C_out };
  }

  _conv2dDown(x, W, H, W_w, C_in, C_out) {
    const OH = Math.floor(H / 2);
    const OW = Math.floor(W_w / 2);
    const out = new Float32Array(OH * OW * C_out);
    for (let oh = 0; oh < OH; oh++) {
      for (let ow = 0; ow < OW; ow++) {
        for (let c = 0; c < C_out; c++) {
          let s = 0;
          for (let kh = 0; kh < 3; kh++) {
            for (let kw = 0; kw < 3; kw++) {
              const sh = oh * 2 + kh - 1;
              const sw = ow * 2 + kw - 1;
              if (sh >= 0 && sh < H && sw >= 0 && sw < W_w) {
                for (let ci = 0; ci < C_in; ci++) {
                  s += x[(sh * W_w + sw) * C_in + ci] * W[(kh * 3 + kw) * C_in * C_out + ci * C_out + c];
                }
              }
            }
          }
          out[(oh * OW + ow) * C_out + c] = s * this._silu(s);
        }
      }
    }
    return { data: out, H: OH, W: OW, C: C_out };
  }

  _convTranspose(x, W, H, W_w, C_in, C_out, OH, OW) {
    const out = new Float32Array(OH * OW * C_out);
    for (let oh = 0; oh < OH; oh++) {
      for (let ow = 0; ow < OW; ow++) {
        for (let c = 0; c < C_out; c++) {
          let s = 0;
          for (let kh = 0; kh < 2; kh++) {
            for (let kw = 0; kw < 2; kw++) {
              const sh = Math.floor((oh - kh + 1) / 2);
              const sw = Math.floor((ow - kw + 1) / 2);
              if (sh >= 0 && sh < H && sw >= 0 && sw < W_w) {
                for (let ci = 0; ci < C_in; ci++) {
                  s += x[(sh * W_w + sw) * C_in + ci] * W[(kh * 2 + kw) * C_in * C_out + ci * C_out + c];
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

  _applyTimeEmbed(x, timeVec, W_time_proj, H, W_w, C) {
    const timeProj = new Float32Array(C);
    for (let i = 0; i < C; i++) {
      let s = 0;
      for (let j = 0; j < timeVec.length; j++) s += timeVec[j] * W_time_proj[j * C + i];
      timeProj[i] = s;
    }
    for (let p = 0; p < H * W_w; p++) {
      for (let c = 0; c < C; c++) x[p * C + c] += timeProj[c];
    }
  }

  _crossAttn(x, textEmb, W_Q, W_K, W_V, W_attn_out, nPix, C, dK) {
    const nText = this.textTokenLen;
    const Q = new Float32Array(nPix * dK);
    for (let i = 0; i < nPix; i++) {
      for (let j = 0; j < dK; j++) {
        let s = 0;
        const wOff = j * C;
        for (let k = 0; k < C; k++) s += x[i * C + k] * W_Q[wOff + k];
        Q[i * dK + j] = s;
      }
    }
    const K = new Float32Array(nText * dK);
    for (let i = 0; i < nText; i++) {
      for (let j = 0; j < dK; j++) {
        let s = 0;
        const wOff = j * this.textEmbedDim;
        for (let k = 0; k < this.textEmbedDim; k++) s += textEmb[i * this.textEmbedDim + k] * W_K[wOff + k];
        K[i * dK + j] = s;
      }
    }
    const V = new Float32Array(nText * C);
    for (let i = 0; i < nText; i++) {
      for (let j = 0; j < C; j++) {
        let s = 0;
        const wOff = j * this.textEmbedDim;
        for (let k = 0; k < this.textEmbedDim; k++) s += textEmb[i * this.textEmbedDim + k] * W_V[wOff + k];
        V[i * C + j] = s;
      }
    }
    const scores = new Float32Array(nPix * nText);
    const invSqrtDk = 1 / Math.sqrt(dK);
    for (let i = 0; i < nPix; i++) {
      for (let j = 0; j < nText; j++) {
        let s = 0;
        for (let k = 0; k < dK; k++) s += Q[i * dK + k] * K[j * dK + k];
        scores[i * nText + j] = s * invSqrtDk;
      }
    }
    const attnOut = new Float32Array(nPix * C);
    for (let i = 0; i < nPix; i++) {
      let maxVal = -Infinity;
      for (let j = 0; j < nText; j++) if (scores[i * nText + j] > maxVal) maxVal = scores[i * nText + j];
      // 裁剪防止溢出: scores 限制在 [-50, 50]
      const clipped = new Float32Array(nText);
      let sum = 0;
      for (let j = 0; j < nText; j++) {
        clipped[j] = Math.max(-50, Math.min(50, scores[i * nText + j] - maxVal));
        sum += Math.exp(clipped[j]);
      }
      if (sum === 0 || !isFinite(sum)) sum = 1;
      for (let j = 0; j < nText; j++) clipped[j] = Math.exp(clipped[j]) / sum;

      for (let c = 0; c < C; c++) {
        let s = 0;
        for (let j = 0; j < nText; j++) s += clipped[j] * V[j * C + c];
        attnOut[i * C + c] = s;
      }
    }
    const projected = new Float32Array(nPix * C);
    for (let i = 0; i < nPix; i++) {
      for (let j = 0; j < C; j++) {
        let s = 0;
        const wOff = j * dK;
        for (let k = 0; k < dK; k++) s += attnOut[i * dK + k] * W_attn_out[wOff + k];
        projected[i * C + j] = s;
      }
    }
    return projected;
  }

  forward(x, t, textEmb, batch) {
    const out = [];
    for (let b = 0; b < batch; b++) {
      const bTextEmb = textEmb ? textEmb[b] : new Float32Array(this.textTokenLen * this.textEmbedDim);
      out.push(this._forwardOne(x[b], t[b], bTextEmb));
    }
    return out;
  }

  _forwardOne(x, t, textEmb) {
    const timeVec = this._timeEmbed(t, this.timeEmbedDim);

    // Down path: 保存 skip (输入尺寸), 然后 conv(stride=2)
    let curX = x;
    let curH = this.latentSize;
    let curW = this.latentSize;
    let curC = this.latentChannels;

    const skips = [];
    for (const block of this.downBlocks) {
      // 保存 skip: 如果 skip 通道 != 目标通道, 先 1x1 conv 匹配
      let skipData = curX;
      if (block.W_skip) {
        // 1x1 conv: [curH, curW, curC] → [curH, curW, block.ch]
        const out = new Float32Array(curH * curW * block.ch);
        for (let h = 0; h < curH; h++) {
          for (let w = 0; w < curW; w++) {
            for (let c = 0; c < block.ch; c++) {
              let s = 0;
              for (let ci = 0; ci < curC; ci++) s += curX[(h * curW + w) * curC + ci] * block.W_skip[ci * block.ch + c];
              out[(h * curW + w) * block.ch + c] = s * this._silu(s);
            }
          }
        }
        skipData = out;
      }
      skips.push({ data: skipData, H: curH, W: curW, C: skipData.length / (curH * curW) });

      // conv(stride=2) 下采样
      const conv = this._conv2dDown(curX, block.W_conv, curH, curW, curC, block.ch);
      let y = conv.data;
      this._applyTimeEmbed(y, timeVec, block.W_time_proj, conv.H, conv.W, block.ch);
      if (textEmb) {
        const attn = this._crossAttn(y, textEmb, block.W_Q, block.W_K, block.W_V,
          block.W_attn_out, conv.H * conv.W, block.ch, block.dK);
        for (let i = 0; i < y.length; i++) y[i] += attn[i];
      }
      curX = y;
      curH = conv.H;
      curW = conv.W;
      curC = block.ch;
    }

    // Middle
    let mid = curX;
    this._applyTimeEmbed(mid, timeVec, this.middleBlock.W_time_proj, curH, curW, curC);
    if (textEmb) {
      const attn = this._crossAttn(mid, textEmb, this.middleBlock.W_Q, this.middleBlock.W_K,
        this.middleBlock.W_V, this.middleBlock.W_attn_out, curH * curW, curC, this.middleBlock.dK);
      for (let i = 0; i < mid.length; i++) mid[i] += attn[i];
    }

    // Up path: up(stride=2) → 合并 skip → conv
    // skips: [down0_skip(输入尺寸), down1_skip(输入尺寸), ...]
    // up[i] 合并 skip[skips.length - 1 - i]
    for (let ui = 0; ui < this.upBlocks.length; ui++) {
      const block = this.upBlocks[ui];
      const skipIdx = skips.length - 1 - ui;
      const skip = skips[skipIdx];

      // up(stride=2)
      const up = this._convTranspose(mid, block.W_up, curH, curW, curC, block.ch,
        block.size, block.size, 2);

      // 合并 skip: 如果 skip 通道 != block.ch, 1x1 conv 匹配
      let skipMerged = skip.data;
      if (block.W_skip && skip.C !== block.ch) {
        const out = new Float32Array(up.H * up.W * block.ch);
        for (let h = 0; h < up.H; h++) {
          for (let w = 0; w < up.W; w++) {
            for (let c = 0; c < block.ch; c++) {
              let s = 0;
              for (let ci = 0; ci < skip.C; ci++) s += skip.data[(h * skip.W + w) * skip.C + ci] * block.W_skip[ci * block.ch + c];
              out[(h * up.W + w) * block.ch + c] = s;
            }
          }
        }
        skipMerged = out;
      }

      // 相加 (残差连接)
      const merged = new Float32Array(up.H * up.W * block.ch);
      for (let i = 0; i < merged.length; i++) merged[i] = up.data[i] + skipMerged[i];

      this._applyTimeEmbed(merged, timeVec, block.W_time_proj, up.H, up.W, block.ch);
      if (textEmb) {
        const attn = this._crossAttn(merged, textEmb, block.W_Q, block.W_K, block.W_V,
          block.W_attn_out, up.H * up.W, block.ch, block.dK);
        for (let i = 0; i < merged.length; i++) merged[i] += attn[i];
      }

      // 输出 conv
      const outConv = this._conv2d(merged, block.W_conv, up.H, up.W, block.ch, block.ch);
      mid = outConv.data;
      curH = outConv.H;
      curW = outConv.W;
      curC = block.ch;
    }

    // 输出卷积
    const output = this._conv2d(mid, this.W_out, curH, curW, curC, this.latentChannels);
    return output.data;
  }

  countParams() {
    let total = this.W_out.length;
    for (const block of this.downBlocks) {
      total += block.W_conv.length + block.W_time_proj.length + block.W_Q.length + block.W_K.length + block.W_V.length + block.W_attn_out.length;
      if (block.W_skip) total += block.W_skip.length;
    }
    total += this.middleBlock.W_conv.length + this.middleBlock.W_time_proj.length + this.middleBlock.W_Q.length + this.middleBlock.W_K.length + this.middleBlock.W_V.length + this.middleBlock.W_attn_out.length;
    for (const block of this.upBlocks) {
      total += block.W_up.length + block.W_time_proj.length + block.W_Q.length + block.W_K.length + block.W_V.length + block.W_attn_out.length;
      if (block.W_skip) total += block.W_skip.length;
    }
    return total;
  }
}

module.exports = { UNet };