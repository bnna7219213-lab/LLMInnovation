/**
 * vocab.js — BPE Vocabulary
 *
 * 维护词表: token_id → Int32Array(byte sequence) 的映射
 * 初始词表 = 256 个 UTF-8 字节 token (0x00..0xFF)
 * 支持: 保存/加载 JSON、byte→token 查找
 */

// 每个 token 的字节序列, 用 Int32Array 存
// 前 256 个 token 就是单字节 0x00..0xFF
const BASE_BYTES = 256;

/**
 * @param {number} initialSize - 初始词表大小, 默认 256 (单字节)
 */
class Vocab {
  constructor(initialSize = BASE_BYTES) {
    if (initialSize !== BASE_BYTES) {
      throw new Error(`vocab.js 仅支持初始词表大小 ${BASE_BYTES}`);
    }
    // token_id → Int32Array of bytes
    this._tokens = new Map();
    // 反向: string 化 byte seq → token_id (用于合并查找)
    this._keyToId = new Map();
    this._size = 0;

    // 初始化 256 个字节 token
    for (let b = 0; b < BASE_BYTES; b++) {
      const bytes = new Int32Array([b]);
      this._tokens.set(b, bytes);
      this._keyToId.set(this._keyOf(bytes), b);
      this._size++;
    }
  }

  get size() { return this._size; }

  /** 字节序列 → 字符串 key (用于 Map 查找) */
  _keyOf(bytes) {
    let s = '';
    for (let i = 0; i < bytes.length; i++) s += bytes[i].toString(16).padStart(2, '0');
    return s;
  }

  /** 添加一个新 token (合并结果), 返回新 token_id */
  add(bytes) {
    const key = this._keyOf(bytes);
    if (this._keyToId.has(key)) {
      return this._keyToId.get(key);
    }
    const id = this._size;
    this._tokens.set(id, bytes);
    this._keyToId.set(key, id);
    this._size++;
    return id;
  }

  /** token_id → byte array */
  getBytes(tokenId) {
    const bytes = this._tokens.get(tokenId);
    if (!bytes) throw new Error(`token_id ${tokenId} 不在词表中 (size=${this._size})`);
    return bytes;
  }

  /** 查找字节序列对应的 token_id (单字节快速查找) */
  byteToId(byteVal) {
    if (byteVal >= 0 && byteVal < BASE_BYTES) return byteVal;
    return -1;
  }

  /** 保存为 JSON (可序列化的扁平结构) */
  toJSON() {
    const out = { size: this._size, tokens: [] };
    for (let i = 0; i < this._size; i++) {
      const b = this._tokens.get(i);
      out.tokens.push(Array.from(b));
    }
    return out;
  }

  /** 从 JSON 恢复 */
  static fromJSON(data) {
    const v = new Vocab(BASE_BYTES);
    // 清空重设
    v._tokens.clear();
    v._keyToId.clear();
    v._size = 0;
    if (data.size && Array.isArray(data.tokens)) {
      for (const bytesArr of data.tokens) {
        v._tokens.set(v._size, new Int32Array(bytesArr));
        v._keyToId.set(v._keyOf(new Int32Array(bytesArr)), v._size);
        v._size++;
      }
    } else {
      // 回退: 只有初始 256
      for (let b = 0; b < BASE_BYTES; b++) {
        const bytes = new Int32Array([b]);
        v._tokens.set(b, bytes);
        v._keyToId.set(v._keyOf(bytes), b);
        v._size++;
      }
    }
    return v;
  }
}

module.exports = { Vocab, BASE_BYTES };