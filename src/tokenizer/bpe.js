/**
 * bpe.js — Byte-Pair Encoding Tokenizer
 *
 * 核心:
 *   - train(corpus, targetVocabSize): 训练 merges 规则
 *   - encode(text): 文本 → token ids
 *   - decode(ids): token ids → 文本
 *
 * 算法:
 *   1. 将所有文本转为 UTF-8 字节序列
 *   2. 每次找到全局最高频的相邻字节对
 *   3. 将该对合并为一个新 token
 *   4. 重复直到词表大小 = targetVocabSize
 *
 * 编码:
 *   将文本转为字节, 贪心应用所有 merges (按顺序), 剩余 token 即结果
 */

const { Vocab } = require('./vocab.js');

/**
 * BPE Tokenizer
 *
 * @param {object} opts
 * @param {number} opts.targetVocabSize - 目标词表大小 (推荐 32768)
 * @param {Vocab} opts.vocab - 已训练的 Vocab (可选)
 * @param {Array} opts.merges - 已训练的 merges 列表 (可选)
 */
class BPETokenizer {
  constructor(opts = {}) {
    this.vocab = opts.vocab || new Vocab();
    this.merges = opts.merges || []; // 每个元素 [leftId, rightId]
    this.targetVocabSize = opts.targetVocabSize || 32768;
  }

  /**
   * 训练 BPE
   *
   * @param {Array<string>} corpus - 文本数组 (每段 100-1000 chars)
   * @param {number} targetSize - 目标词表大小
   * @returns {BPETokenizer} 新的 tokenizer (或修改当前实例)
   */
  train(corpus, targetSize = this.targetVocabSize) {
    if (!corpus || corpus.length === 0) throw new Error('corpus 不能为空');

    const vocab = new Vocab();
    const merges = [];

    // 将每段文本转为字节 token 序列 (初始就是字节)
    const docTokens = corpus.map(text => {
      const bytes = strToBytes(text);
      const arr = [];
      for (let i = 0; i < bytes.length; i++) {
        arr.push(bytes[i]); // 初始每个 token 就是单个字节
      }
      return arr;
    });

    const numMerges = targetSize - vocab.size; // 需要多少次合并
    if (numMerges <= 0) return this;

    // 预分配: 每次合并后 docTokens 中每个 token 可能变长
    for (let step = 0; step < numMerges; step++) {
      // 统计所有相邻对 (a, b) 的频率
      const pairCounts = new Map(); // "a b" → count
      for (const doc of docTokens) {
        for (let i = 0; i < doc.length - 1; i++) {
          const a = doc[i], b = doc[i + 1];
          const key = a + ' ' + b;
          pairCounts.set(key, (pairCounts.get(key) || 0) + 1);
        }
      }

      // 找最高频对
      let bestPair = null, bestCount = 0;
      for (const [key, count] of pairCounts) {
        if (count > bestCount) {
          bestCount = count;
          bestPair = key;
        }
      }

      if (!bestPair || bestCount === 0) break;

      const [leftStr, rightStr] = bestPair.split(' ');
      const leftId = parseInt(leftStr, 10);
      const rightId = parseInt(rightStr, 10);

      // 获取字节序列
      const leftBytes = vocab.getBytes(leftId);
      const rightBytes = vocab.getBytes(rightId);
      const newBytes = new Int32Array(leftBytes.length + rightBytes.length);
      for (let i = 0; i < leftBytes.length; i++) newBytes[i] = leftBytes[i];
      for (let i = 0; i < rightBytes.length; i++) newBytes[leftBytes.length + i] = rightBytes[i];

      // 添加新 token
      const newId = vocab.add(newBytes);

      // 记录 merge
      merges.push([leftId, rightId, newId]);

      // 应用合并到所有文档: 将 [leftId, rightId] 替换为 newId
      // 用链表结构高效替换
      for (let d = 0; d < docTokens.length; d++) {
        const doc = docTokens[d];
        const newDoc = [];
        let i = 0;
        while (i < doc.length) {
          if (i < doc.length - 1 && doc[i] === leftId && doc[i + 1] === rightId) {
            newDoc.push(newId);
            i += 2;
          } else {
            newDoc.push(doc[i]);
            i++;
          }
        }
        docTokens[d] = newDoc;
      }
    }

    this.vocab = vocab;
    this.merges = merges;
    return this;
  }

  /**
   * 编码: 文本 → token ids
   *
   * @param {string} text
   * @returns {Array<number>}
   */
  encode(text) {
    if (typeof text !== 'string') throw new Error('encode 输入必须是 string');
    if (text.length === 0) return [];

    // 转 UTF-8 字节
    const bytes = strToBytes(text);
    let tokens = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) tokens[i] = bytes[i];

    if (this.merges.length === 0) return tokens;

    // 构建 merge 查找表: 对每个 merge, 找到其在序列中位置
    // 用 "对" 链表: 每个 token 节点存 (id, next 指针)
    // 贪心从左到右应用, 使用配对索引加速
    //
    // 高效实现: 将所有 merges 构建为 (leftId, rightId) → newId 的 Map
    // 然后迭代合并直到无法合并
    //
    // 但 O(numMerges × seqLen) 对于 32K merges 和长文本可能慢
    // 优化: 用 rank 数组, 每个 merge 有一个优先级, 按优先级合并
    //
    // 这里用简化版本: 对每个 merge, 扫描序列找匹配对, 替换
    // 对长文本 (数千 token) 和 32K merges, 大约 32K × 数千 = 1 亿操作
    // Node.js 每秒 ~1 亿次, 可行
    //
    // 更好的实现: 用 Trie + 贪心匹配, 一次扫描搞定

    // --- Trie-based greedy merge (一次扫描, 最优) ---
    // 将所有 merge 规则构建为字典树:
    //   根 → 第一个字节 → ... → 第 n 个字节 → newTokenId
    // 扫描字节流, 在 Trie 上匹配最长前缀, 匹配成功则输出 token

    if (this.merges.length > 0) {
      // 构建字典: 字节序列 → newTokenId
      // 按字节序列长度排序, 先匹配长的 (贪心)
      const rules = this.merges.map(m => ({
        leftId: m[0],
        rightId: m[1],
        newId: m[2],
        leftBytes: this.vocab.getBytes(m[0]),
        rightBytes: this.vocab.getBytes(m[1]),
        fullBytes: null, // 下面计算
      }));

      // 计算每个 merge 的完整字节序列
      for (const r of rules) {
        const len = r.leftBytes.length + r.rightBytes.length;
        const arr = new Int32Array(len);
        for (let i = 0; i < r.leftBytes.length; i++) arr[i] = r.leftBytes[i];
        for (let i = 0; i < r.rightBytes.length; i++) arr[r.leftBytes.length + i] = r.rightBytes[i];
        r.fullBytes = arr;
      }

      // 按字节序列长度降序 (贪心: 先匹配最长的)
      rules.sort((a, b) => b.fullBytes.length - a.fullBytes.length);

      // 贪心匹配: 从 bytes 中找最长匹配
      tokens = this._greedyMatch(bytes, rules);
    }

    return tokens;
  }

  /**
   * 贪心匹配: 从字节流中按最长优先匹配所有规则
   * @param {Uint8Array} bytes
   * @param {Array} rules 已按长度降序的规则
   * @returns {Array<number>}
   */
  _greedyMatch(bytes, rules) {
    const result = [];
    let i = 0;
    const n = bytes.length;

    while (i < n) {
      let matched = false;

      // 按长度降序尝试每个规则
      for (const rule of rules) {
        const rb = rule.fullBytes;
        if (i + rb.length > n) continue;
        let ok = true;
        for (let j = 0; j < rb.length; j++) {
          if (bytes[i + j] !== rb[j]) { ok = false; break; }
        }
        if (ok) {
          result.push(rule.newId);
          i += rb.length;
          matched = true;
          break;
        }
      }

      if (!matched) {
        // 没有规则匹配, 输出单个字节 token
        result.push(bytes[i]);
        i++;
      }
    }

    return result;
  }

  /**
   * 解码: token ids → 文本
   *
   * @param {Array<number>} tokenIds
   * @returns {string}
   */
  decode(tokenIds) {
    let totalLen = 0;
    for (const tid of tokenIds) {
      totalLen += this.vocab.getBytes(tid).length;
    }
    const allBytes = new Int32Array(totalLen);
    let offset = 0;
    for (const tid of tokenIds) {
      const b = this.vocab.getBytes(tid);
      for (let i = 0; i < b.length; i++) {
        allBytes[offset + i] = b[i];
      }
      offset += b.length;
    }
    return bytesToStr(allBytes);
  }

  /** 保存训练结果为 JSON */
  toJSON() {
    return {
      vocab: this.vocab.toJSON(),
      merges: this.merges,
      targetVocabSize: this.targetVocabSize,
    };
  }

  /** 从 JSON 恢复 */
  static fromJSON(data) {
    const vocab = Vocab.fromJSON(data.vocab);
    return new BPETokenizer({
      vocab,
      merges: data.merges,
      targetVocabSize: data.targetVocabSize,
    });
  }
}

// ============ 工具函数: UTF-8 编解码 ============

/** 字符串 → UTF-8 字节 (Int32Array) */
function strToBytes(str) {
  // 先转 UTF-8 字节, 然后放到 Int32Array
  const encoder = new TextEncoder();
  const u8 = encoder.encode(str);
  const out = new Int32Array(u8.length);
  for (let i = 0; i < u8.length; i++) out[i] = u8[i];
  return out;
}

/** 字节 (Int32Array) → 字符串 (UTF-8 解码) */
function bytesToStr(bytes) {
  const u8 = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) u8[i] = bytes[i];
  return new TextDecoder('utf-8', { fatal: false }).decode(u8);
}

module.exports = { BPETokenizer, strToBytes, bytesToStr };