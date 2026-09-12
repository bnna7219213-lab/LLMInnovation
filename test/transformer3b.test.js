/**
 * transformer3b.test.js — Phase 1 3B 模型测试
 *
 * 使用微型配置验证代码路径 (同一套代码, 不同参数规模):
 *   vocabSize: 64, dModel: 16, numLayers: 2, numHeads: 4, numKVHeads: 2, dFF: 32
 *
 * 测试内容:
 *   1. 构造器: 配置验证 + 权重初始化
 *   2. forward: 维度正确 + 因果 mask 正确
 *   3. generate: greedy sampling + temperature sampling
 *   4. 参数统计
 *   5. tie embedding 验证
 */

const { TransformerModel, CONFIG_3B } = require('../src/models/text/transformer3b.js');
const { BPETokenizer } = require('../src/tokenizer/bpe.js');
const assert = require('assert');
const { describe, it, beforeEach } = require('node:test');

// 测试用微型配置 (同一套代码, 极小参数)
const TINY_CONFIG = {
  vocabSize: 256,   // >= BPE 初始 256 字节 token
  dModel: 16,
  numLayers: 2,
  numHeads: 4,
  numKVHeads: 2,
  dFF: 32,
  maxSeqLen: 64,
};

// 生成一个微型 tokenizer (5 段短文本)
function makeTinyTokenizer() {
  const corpus = [
    'the cat sat on the mat',
    'a dog ran in the park',
    'the sun is bright today',
    'hello world this is a test',
    'the end of the sentence',
  ];
  const tok = new BPETokenizer({ targetVocabSize: 128 });
  tok.train(corpus, 64); // 不超过 vocabSize=64
  return tok;
}

describe('TransformerModel (tiny)', () => {
  let model, tokenizer;

  beforeEach(() => {
    model = new TransformerModel(TINY_CONFIG);
    tokenizer = makeTinyTokenizer();
  });

  it('构造器: 配置验证', () => {
    assert.strictEqual(model.vocabSize, 256);
    assert.strictEqual(model.dModel, 16);
    assert.strictEqual(model.numLayers, 2);
    assert.strictEqual(model.numHeads, 4);
    assert.strictEqual(model.numKVHeads, 2);
    assert.strictEqual(model.dK, 4);
    assert.strictEqual(model.dFF, 32);
    assert.strictEqual(model.maxSeqLen, 64);
  });

  it('构造器: dModel 不能被 numHeads 整除抛错', () => {
    assert.throws(
      () => new TransformerModel({ ...TINY_CONFIG, dModel: 15, numHeads: 4 }),
      /整除/
    );
  });

  it('构造器: numHeads 不能被 numKVHeads 整除抛错', () => {
    assert.throws(
      () => new TransformerModel({ ...TINY_CONFIG, numHeads: 5, numKVHeads: 2 }),
      /整除/
    );
  });

  it('forward: 输出维度正确 [seqLen, vocabSize]', () => {
    const inputIds = tokenizer.encode('the cat sat');
    const logits = model.forward(inputIds, inputIds.length);
    assert.strictEqual(logits.length, inputIds.length * model.vocabSize);
  });

  it('forward: 不同输入产生不同输出', () => {
    const ids1 = tokenizer.encode('the cat sat');
    const ids2 = tokenizer.encode('the dog ran');
    const logits1 = model.forward(ids1, ids1.length);
    const logits2 = model.forward(ids2, ids2.length);

    // 输出应该不同
    let diff = 0;
    for (let i = 0; i < logits1.length; i++) {
      diff += Math.abs(logits1[i] - logits2[i]);
    }
    assert.ok(diff > 0.01, `不同输入的输出几乎相同 (总差=${diff})`);
  });

  it('forward: 相同输入产生相同输出 (确定性, 无 dropout)', () => {
    const ids = tokenizer.encode('the cat sat');
    const logits1 = model.forward(ids, ids.length);
    const logits2 = model.forward(ids, ids.length);

    for (let i = 0; i < logits1.length; i++) {
      assert.strictEqual(logits1[i], logits2[i]);
    }
  });

  it('forward: token_id 超出词表范围抛错', () => {
    const ids = [0, 1, 256]; // 256 >= vocabSize=256
    assert.throws(() => model.forward(ids, 3), /超出词表/);
  });

  it('forward: seqLen 超过 maxSeqLen 抛错', () => {
    const ids = new Array(100).fill(0);
    assert.throws(() => model.forward(ids, 100), /maxSeqLen/);
  });

  it('forward: 因果 mask 正确 (位置 0 的 logit 不应受位置 1 影响)', () => {
    // 比较两个场景:
    //   A: input = [a, b]  →  pos0 只看 pos0
    //   B: input = [a, c]  →  pos0 只看 pos0
    // pos0 的 logits 应该完全相同 (因果 mask 屏蔽了 pos1)
    const idsA = tokenizer.encode('the cat sat');
    const idsB = [...idsA];
    idsB[1] = (idsB[1] + 1) % model.vocabSize; // 改变第二个 token

    const logitsA = model.forward(idsA, idsA.length);
    const logitsB = model.forward(idsB, idsB.length);

    // 位置 0 的 logits 应完全相同
    for (let j = 0; j < model.vocabSize; j++) {
      const a = logitsA[0 * model.vocabSize + j];
      const b = logitsB[0 * model.vocabSize + j];
      assert.strictEqual(a, b, `pos0 logit[${j}]: A=${a} B=${b} 应相同`);
    }
  });

  it('generate: greedy (temperature=0) 确定性输出', () => {
    const promptIds = tokenizer.encode('the cat');
    const out1 = model.generate(promptIds, 10, 0);
    const out2 = model.generate(promptIds, 10, 0);

    assert.deepStrictEqual(out1, out2);
    assert.strictEqual(out1.length, promptIds.length + 10);
  });

  it('generate: temperature>0 产生随机输出 (多次运行不完全相同)', () => {
    const promptIds = tokenizer.encode('the cat');
    const outputs = new Set();
    for (let i = 0; i < 5; i++) {
      const out = model.generate(promptIds, 5, 1.0);
      outputs.add(JSON.stringify(out));
    }
    // 5 次运行, 至少 2 种不同结果
    assert.ok(outputs.size >= 2, `温度采样应产生多样性, 实际只有 ${outputs.size} 种`);
  });

  it('generate: maxNewTokens=0 只返回 prompt', () => {
    const promptIds = tokenizer.encode('the cat');
    const out = model.generate(promptIds, 0, 0);
    assert.deepStrictEqual(out, promptIds);
  });

  it('参数统计: 总参数量正确', () => {
    const total = model.countParams();
    // Wembed: 256*16 = 4096
    // 每层: 256+128+128+256+16+16+512+512+512 = 2336
    // 2 层: 4672
    // normGamma: 16
    // 总计: 4096 + 4672 + 16 = 8784
    assert.strictEqual(total, 8784);
  });

  it('getParams: 返回所有参数字段', () => {
    const params = model.getParams();
    // 应有: Wembed + 每层 9 个 + normGamma
    const expectedKeys = 1 + 2 * 9 + 1;
    assert.strictEqual(Object.keys(params).length, expectedKeys);
    assert(params['Wembed'] instanceof Float32Array);
    assert(params['layer0.WQ'] instanceof Float32Array);
  });
});

describe('TransformerModel (3B config)', () => {
  it('CONFIG_3B 参数合理', () => {
    assert.strictEqual(CONFIG_3B.vocabSize, 32768);
    assert.strictEqual(CONFIG_3B.dModel, 2560);
    assert.strictEqual(CONFIG_3B.numLayers, 32);
    assert.strictEqual(CONFIG_3B.numHeads, 32);
    assert.strictEqual(CONFIG_3B.numKVHeads, 8);
    assert.strictEqual(CONFIG_3B.dFF, 6912);
  });
});