/**
 * mamba.test.js — Phase 3 Mamba + Jamba 测试
 *
 * 测试内容:
 *   1. Selective Scan: 顺序扫描正确性
 *   2. Mamba Block: 维度正确 + 确定性
 *   3. Jamba Model: 交替层结构 + forward + generate
 */

const { selectiveScanSequential } = require('../src/models/mamba/selective_scan.js');
const { MambaBlock } = require('../src/models/mamba/mamba_block.js');
const { JambaModel } = require('../src/models/hybrid/jamba.js');
const { BPETokenizer } = require('../src/tokenizer/bpe.js');
const assert = require('assert');
const { describe, it, beforeEach } = require('node:test');

// ============ Selective Scan ============

describe('SelectiveScan', () => {
  it('基本功能: 顺序扫描输出维度正确', () => {
    const seqLen = 4, dState = 2;
    const x = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const A = new Float32Array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]);
    const B = new Float32Array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
    const C = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]);

    const out = selectiveScanSequential(x, A, B, C, seqLen, dState);
    assert.strictEqual(out.length, seqLen * dState);
  });

  it('数学验证: h_t = A*h_{t-1} + B*x_t, y_t = C*h_t', () => {
    const seqLen = 3, dState = 1;
    const x = new Float32Array([1, 2, 3]);
    const A = new Float32Array([1, 1, 1]); // A=1: 纯累加
    const B = new Float32Array([1, 1, 1]); // B=1
    const C = new Float32Array([1, 1, 1]); // C=1

    // 手算:
    // h_0 = 1*0 + 1*1 = 1, y_0 = 1*1 = 1
    // h_1 = 1*1 + 1*2 = 3, y_1 = 1*3 = 3
    // h_2 = 1*3 + 1*3 = 6, y_2 = 1*6 = 6
    const out = selectiveScanSequential(x, A, B, C, seqLen, dState);
    assert.strictEqual(out[0], 1);
    assert.strictEqual(out[1], 3);
    assert.strictEqual(out[2], 6);
  });

  it('衰减行为: A<1 时状态衰减', () => {
    const seqLen = 5, dState = 1;
    const x = new Float32Array([1, 0, 0, 0, 0]); // 只有第一步有输入
    const A = new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5]); // 衰减
    const B = new Float32Array([1, 1, 1, 1, 1]);
    const C = new Float32Array([1, 1, 1, 1, 1]);

    // h_0 = 0 + 1*1 = 1, y_0 = 1
    // h_1 = 0.5*1 + 0 = 0.5, y_1 = 0.5
    // h_2 = 0.5*0.5 + 0 = 0.25, y_2 = 0.25
    // h_3 = 0.5*0.25 = 0.125, y_3 = 0.125
    // h_4 = 0.5*0.125 = 0.0625, y_4 = 0.0625
    const out = selectiveScanSequential(x, A, B, C, seqLen, dState);
    assert.ok(Math.abs(out[0] - 1) < 1e-6);
    assert.ok(Math.abs(out[1] - 0.5) < 1e-6);
    assert.ok(Math.abs(out[2] - 0.25) < 1e-6);
    assert.ok(Math.abs(out[3] - 0.125) < 1e-6);
    assert.ok(Math.abs(out[4] - 0.0625) < 1e-6);
  });

  it('不同 dState 维度正确', () => {
    const seqLen = 4, dState = 4;
    const x = new Float32Array(seqLen * dState);
    const A = new Float32Array(seqLen * dState);
    const B = new Float32Array(seqLen * dState);
    const C = new Float32Array(seqLen * dState);
    for (let i = 0; i < x.length; i++) { x[i] = 0.1; A[i] = 0.9; B[i] = 0.1; C[i] = 1; }

    const out = selectiveScanSequential(x, A, B, C, seqLen, dState);
    assert.strictEqual(out.length, seqLen * dState);
  });
});

// ============ Mamba Block ============

describe('MambaBlock', () => {
  const dModel = 32, dState = 4, dConv = 4;
  let block;

  beforeEach(() => { block = new MambaBlock(dModel, dState, dConv); });

  it('构造器: 参数维度正确', () => {
    assert.strictEqual(block.dModel, 32);
    assert.strictEqual(block.dState, 4);
    assert.strictEqual(block.dConv, 4);
  });

  it('forward: 输出维度 [seqLen, dModel]', () => {
    const seqLen = 8;
    const x = new Float32Array(seqLen * dModel);
    for (let i = 0; i < x.length; i++) x[i] = 0.01;

    const out = block.forward(x, seqLen);
    assert.strictEqual(out.length, seqLen * dModel);
  });

  it('forward: 零输入输出零', () => {
    const seqLen = 4;
    const x = new Float32Array(seqLen * dModel); // 全零
    const out = block.forward(x, seqLen);
    for (let i = 0; i < out.length; i++) {
      assert.ok(Math.abs(out[i]) < 1e-10, `out[${i}]=${out[i]}`);
    }
  });

  it('forward: 确定性 (相同输入相同输出)', () => {
    const seqLen = 4;
    const x = new Float32Array(seqLen * dModel);
    for (let i = 0; i < x.length; i++) x[i] = 0.05;

    const out1 = block.forward(x, seqLen);
    const out2 = block.forward(x, seqLen);
    for (let i = 0; i < out1.length; i++) assert.strictEqual(out1[i], out2[i]);
  });

  it('countParams: 参数量合理', () => {
    const total = block.countParams();
    assert.ok(total > 0, '参数量应 > 0');
    assert.ok(total < 100000, `参数量 ${total} 应 < 100000 (dModel=32)`);
  });
});

// ============ Jamba Model ============

function makeTinyTokenizer() {
  const corpus = [
    'the cat sat on the mat',
    'a dog ran in the park',
    'the sun is bright today',
    'hello world this is a test',
    'the end of the sentence',
  ];
  const tok = new BPETokenizer({ targetVocabSize: 128 });
  tok.train(corpus, 128);
  return tok;
}

describe('JambaModel', () => {
  const config = {
    vocabSize: 256,
    dModel: 16,
    numLayers: 4,   // 2 Mamba + 2 Transformer
    numHeads: 4,
    numKVHeads: 2,
    dFF: 32,
    maxSeqLen: 64,
    dState: 4,
    dConv: 4,
  };
  let model, tokenizer;

  beforeEach(() => {
    model = new JambaModel(config);
    tokenizer = makeTinyTokenizer();
  });

  it('构造器: 交替层结构正确 (Mamba, Transformer, Mamba, Transformer)', () => {
    const types = model.blocks.map(b => b.type);
    assert.deepStrictEqual(types, ['mamba', 'transformer', 'mamba', 'transformer']);
  });

  it('countLayerTypes: 2 Mamba + 2 Transformer', () => {
    const counts = model.countLayerTypes();
    assert.strictEqual(counts.mamba, 2);
    assert.strictEqual(counts.transformer, 2);
  });

  it('forward: 输出维度正确 [seqLen, vocabSize]', () => {
    const ids = tokenizer.encode('the cat sat');
    const logits = model.forward(ids, ids.length);
    assert.strictEqual(logits.length, ids.length * model.vocabSize);
  });

  it('forward: 确定性 (相同输入相同输出)', () => {
    const ids = tokenizer.encode('the cat sat');
    const logits1 = model.forward(ids, ids.length);
    const logits2 = model.forward(ids, ids.length);
    for (let i = 0; i < logits1.length; i++) assert.strictEqual(logits1[i], logits2[i]);
  });

  it('generate: greedy 确定性输出', () => {
    const promptIds = tokenizer.encode('the cat');
    const out1 = model.generate(promptIds, 5, 0);
    const out2 = model.generate(promptIds, 5, 0);
    assert.deepStrictEqual(out1, out2);
  });

  it('generate: temperature>0 有多样性', () => {
    const promptIds = tokenizer.encode('the cat');
    const outputs = new Set();
    for (let i = 0; i < 5; i++) {
      outputs.add(JSON.stringify(model.generate(promptIds, 3, 5.0))); // 高温度保证多样性
    }
    assert.ok(outputs.size >= 2, `温度采样应产生多样性, 实际 ${outputs.size} 种`);
  });

  it('generate: maxNewTokens=0 只返回 prompt', () => {
    const promptIds = tokenizer.encode('the cat');
    const out = model.generate(promptIds, 0, 0);
    assert.deepStrictEqual(out, promptIds);
  });

  it('countParams: 参数量合理', () => {
    const total = model.countParams();
    assert.ok(total > 1000, `参数量 ${total} 应 > 1000`);
  });
});