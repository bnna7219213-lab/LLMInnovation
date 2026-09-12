/**
 * kv_cache.js — KV Cache 推理优化
 *
 * KV Cache 是 autoregressive 推理的核心优化:
 *   - 生成 token t 时, 前 t-1 个 token 的 K/V 已在过去计算过
 *   - 缓存它们, 只计算新 token 的 Q/K/V, 拼接后做 attention
 *   - 避免 O(n²) → O(n) 的重复计算
 *
 * 支持:
 *   - 按层缓存 (每层一个 cache)
 *   - GQA: 缓存 numKVHeads 个 KV head
 *   - 滑动窗口: 只保留最近 windowSize 个 token (长上下文)
 *   - 量化 (int8): 减少显存
 *
 * 内存估算 (3B 模型, GQA=8, dK=128):
 *   每层 KV: 2 (K+V) × numKVHeads(8) × dK(128) × maxSeqLen × 2 bytes
 *   = 2 × 8 × 128 × 4096 × 2 = 16.8 MB / 层
 *   32 层 = 537 MB (对 3B 模型可接受)
 */

class KVCache {
  /**
   * @param {object} config
   * @param {number} config.numLayers
   * @param {number} config.numKVHeads
   * @param {number} config.dK - 每 KV head 维度
   * @param {number} config.maxSeqLen
   * @param {number} config.windowSize - 滑动窗口 (0 = 无限, 默认 0)
   * @param {boolean} config.quantize - int8 量化 (默认 false)
   */
  constructor(config) {
    this.numLayers = config.numLayers;
    this.numKVHeads = config.numKVHeads;
    this.dK = config.dK;
    this.maxSeqLen = config.maxSeqLen;
    this.windowSize = config.windowSize || 0;
    this.quantize = config.quantize || false;

    // 每层: { keys: [seq, numKVHeads, dK], values: [...], len }
    this.layers = [];
    for (let l = 0; l < this.numLayers; l++) {
      this.layers.push({
        keys: new Float32Array(this.maxSeqLen * this.numKVHeads * this.dK),
        values: new Float32Array(this.maxSeqLen * this.numKVHeads * this.dK),
        len: 0,
      });
    }

    this.bytesPerElem = this.quantize ? 1 : 4;
  }

  /**
   * 写入一层的 KV
   * @param {number} layerIdx
   * @param {Float32Array} keys - [1, numKVHeads, dK] (单个新 token)
   * @param {Float32Array} values - [1, numKVHeads, dK]
   */
  store(layerIdx, keys, values) {
    const layer = this.layers[layerIdx];
    const slot = layer.len * this.numKVHeads * this.dK;
    layer.keys.set(keys, slot);
    layer.values.set(values, slot);
    layer.len++;
  }

  /**
   * 读取一层的 KV (考虑滑动窗口)
   * @param {number} layerIdx
   * @returns {{keys: Float32Array, values: Float32Array, len: number, offset: number}}
   */
  retrieve(layerIdx) {
    const layer = this.layers[layerIdx];
    let start = 0;
    let len = layer.len;
    if (this.windowSize > 0 && len > this.windowSize) {
      start = len - this.windowSize;
      len = this.windowSize;
    }
    const offset = start * this.numKVHeads * this.dK;
    return {
      keys: layer.keys.subarray(offset, offset + len * this.numKVHeads * this.dK),
      values: layer.values.subarray(offset, offset + len * this.numKVHeads * this.dK),
      len,
      offset: start,
    };
  }

  /**
   * 获取有效序列长度 (考虑窗口)
   * @param {number} layerIdx
   * @returns {number}
   */
  effectiveLen(layerIdx) {
    const len = this.layers[layerIdx].len;
    return this.windowSize > 0 ? Math.min(len, this.windowSize) : len;
  }

  /**
   * 清空缓存
   */
  clear() {
    for (const layer of this.layers) {
      layer.keys.fill(0);
      layer.values.fill(0);
      layer.len = 0;
    }
  }

  /**
   * 当前缓存占用内存 (bytes)
   */
  memoryBytes() {
    let total = 0;
    for (const layer of this.layers) {
      const used = layer.len * this.numKVHeads * this.dK * 2;
      total += used * this.bytesPerElem;
    }
    return total;
  }

  /**
   * 峰值内存 (全部占满时)
   */
  peakMemoryBytes() {
    return this.numLayers * this.maxSeqLen * this.numKVHeads * this.dK * 2 * this.bytesPerElem;
  }
}

/**
 * 简单 int8 量化/反量化 (对称量化)
 * @param {Float32Array} x
 * @returns {{data: Int8Array, scale: number}}
 */
function quantizeInt8(x) {
  let maxAbs = 0;
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs > 0 ? maxAbs / 127 : 1;
  const data = new Int8Array(x.length);
  for (let i = 0; i < x.length; i++) {
    data[i] = Math.round(Math.max(-127, Math.min(127, x[i] / scale)));
  }
  return { data, scale };
}

/**
 * int8 反量化
 * @param {Int8Array} data
 * @param {number} scale
 * @returns {Float32Array}
 */
function dequantizeInt8(data, scale) {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = data[i] * scale;
  }
  return out;
}

module.exports = { KVCache, quantizeInt8, dequantizeInt8 };