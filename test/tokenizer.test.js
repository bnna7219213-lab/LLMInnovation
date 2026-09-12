/**
 * tokenizer.test.js — Phase 0 Tokenizer 测试
 *
 * 测试内容:
 *   1. Vocab: 初始 256 字节, add 新 token
 *   2. BPE 训练: 小语料合并
 *   3. encode/decode 可逆
 *   4. 中英文、特殊字符处理
 *   5. 空字符串、OOV
 *   6. 序列化/反序列化
 */

const { Vocab } = require('../src/tokenizer/vocab.js');
const { BPETokenizer } = require('../src/tokenizer/bpe.js');
const assert = require('assert');
const { describe, it } = require('node:test');

// ============ Vocab ============

describe('Vocab', () => {
  it('初始词表大小 = 256 (单字节)', () => {
    const v = new Vocab();
    assert.strictEqual(v.size, 256);
  });

  it('byteToId 正确映射 0-255', () => {
    const v = new Vocab();
    for (let b = 0; b < 256; b++) {
      assert.strictEqual(v.byteToId(b), b, `byte ${b} → ${v.byteToId(b)}`);
    }
  });

  it('getBytes 返回正确的单字节序列', () => {
    const v = new Vocab();
    for (let b = 0; b < 256; b++) {
      const bytes = v.getBytes(b);
      assert.strictEqual(bytes.length, 1);
      assert.strictEqual(bytes[0], b);
    }
  });

  it('add 新 token 返回递增 id', () => {
    const v = new Vocab();
    const id1 = v.add(new Int32Array([65, 66])); // 'AB'
    assert.strictEqual(id1, 256);
    const id2 = v.add(new Int32Array([65, 67])); // 'AC'
    assert.strictEqual(id2, 257);
    assert.strictEqual(v.size, 258);
  });

  it('重复 add 相同字节返回已有 id', () => {
    const v = new Vocab();
    const id1 = v.add(new Int32Array([65, 66]));
    const id2 = v.add(new Int32Array([65, 66]));
    assert.strictEqual(id1, id2);
  });

  it('toJSON / fromJSON 可逆', () => {
    const v = new Vocab();
    v.add(new Int32Array([65, 66]));
    v.add(new Int32Array([67, 68]));
    const json = v.toJSON();
    const v2 = Vocab.fromJSON(json);
    assert.strictEqual(v2.size, 258);
    assert.deepStrictEqual(Array.from(v2.getBytes(256)), [65, 66]);
    assert.deepStrictEqual(Array.from(v2.getBytes(257)), [67, 68]);
  });
});

// ============ BPE Tokenizer ============

describe('BPETokenizer', () => {
  // 测试用最小语料
  const sampleCorpus = [
    'hello world hello world',
    'hello hello world',
    'the world is hello',
    'hello world the end',
    'the end of hello',
  ];

  it('encode 英文文本 返回 token ids 数组', () => {
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    tokenizer.train(sampleCorpus, 512);
    const ids = tokenizer.encode('hello world');
    assert(Array.isArray(ids));
    assert(ids.length > 0, 'token 数应 > 0');
  });

  it('decode(encode(x)) === x (可逆性)', () => {
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    tokenizer.train(sampleCorpus, 512);

    const tests = [
      'hello world',
      'the quick brown fox',
      'a b c d e f g',
      'zzz yyy xxx',
      'one two three four five',
    ];

    for (const text of tests) {
      const ids = tokenizer.encode(text);
      const decoded = tokenizer.decode(ids);
      assert.strictEqual(decoded, text, `不可逆: "${text}" → "${decoded}"`);
    }
  });

  it('decode(encode(x)) === x (中文字符)', () => {
    const corpus = [
      '你好世界你好世界',
      '我是中国人',
      '人工智能是未来',
      '深度学习很重要',
      '你好你好世界',
    ];
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    tokenizer.train(corpus, 512);

    const text = '你好世界';
    const ids = tokenizer.encode(text);
    const decoded = tokenizer.decode(ids);
    assert.strictEqual(decoded, text, `中文不可逆: "${text}" → "${decoded}"`);
  });

  it('decode(encode(x)) === x (特殊字符和标点)', () => {
    const corpus = [
      'hello, world! how are you?',
      'a=b+c*d/e-f',
      '<html><body>hello</body></html>',
      '{"key": "value", "num": 123}',
      "don't stop, it's working",
    ];
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    tokenizer.train(corpus, 512);

    const tests = [
      'hello, world!',
      'a=b+c*d',
      '{"key": "value"}',
    ];

    for (const text of tests) {
      const ids = tokenizer.encode(text);
      const decoded = tokenizer.decode(ids);
      assert.strictEqual(decoded, text, `特殊字符不可逆: "${text}" → "${decoded}"`);
    }
  });

  it('encode 空字符串 返回空数组', () => {
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    tokenizer.train(sampleCorpus, 512);
    const ids = tokenizer.encode('');
    assert.deepStrictEqual(ids, []);
  });

  it('训练后词表大小 <= targetVocabSize (小语料不足时提前停止)', () => {
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    tokenizer.train(sampleCorpus, 512);
    assert(tokenizer.vocab.size >= 256, `词表应 >= 初始 256, 实际 ${tokenizer.vocab.size}`);
    assert(tokenizer.vocab.size <= 512, `词表应 <= 512, 实际 ${tokenizer.vocab.size}`);
    assert(tokenizer.merges.length === tokenizer.vocab.size - 256);
  });

  it('toJSON / fromJSON 可逆', () => {
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    tokenizer.train(sampleCorpus, 512);

    const json = tokenizer.toJSON();
    const restored = BPETokenizer.fromJSON(json);

    const text = 'hello world';
    const ids1 = tokenizer.encode(text);
    const ids2 = restored.encode(text);
    assert.deepStrictEqual(ids1, ids2);
  });

  it('encode 输入非 string 抛错', () => {
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    tokenizer.train(sampleCorpus, 512);
    assert.throws(() => tokenizer.encode(null), /必须是 string/);
    assert.throws(() => tokenizer.encode(123), /必须是 string/);
  });

  it('训练空 corpus 抛错', () => {
    const tokenizer = new BPETokenizer({ targetVocabSize: 512 });
    assert.throws(() => tokenizer.train([], 512), /不能为空/);
  });
});

// ============ 大语料压力测试 ============

describe('BPETokenizer (stress)', () => {
  it('训练 100 段语料 → targetVocabSize=1024 → 编码 100 条 全部可逆', () => {
    // 生成伪语料 (模拟真实文本的重复模式)
    const corpus = [];
    const baseSentences = [
      'the quick brown fox jumps over the lazy dog',
      'artificial intelligence is the future of technology',
      'hello world this is a test of the byte pair encoding tokenizer',
      'machine learning models are trained on large datasets',
      'deep learning is a subset of machine learning',
    ];
    for (let i = 0; i < 100; i++) {
      const parts = [];
      for (let j = 0; j < 5; j++) {
        parts.push(baseSentences[(i + j) % baseSentences.length]);
      }
      corpus.push(parts.join(' '));
    }

    const tokenizer = new BPETokenizer({ targetVocabSize: 1024 });
    tokenizer.train(corpus, 1024);

    assert.ok(tokenizer.vocab.size >= 256 && tokenizer.vocab.size <= 1024,
      `词表大小 ${tokenizer.vocab.size} 应在 [256, 1024]`);
    assert.strictEqual(tokenizer.merges.length, tokenizer.vocab.size - 256);

    // 测试可逆性
    for (const text of corpus) {
      const ids = tokenizer.encode(text);
      const decoded = tokenizer.decode(ids);
      assert.strictEqual(decoded, text);
    }
  });
});