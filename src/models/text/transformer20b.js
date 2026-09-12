/**
 * transformer20b.js — 20B 模型配置
 *
 * 与 3B 完全相同的架构, 仅配置不同:
 *   vocabSize: 32768
 *   dModel: 6144
 *   numLayers: 48
 *   numHeads: 24
 *   numKVHeads: 8  (GQA, 每组 3 个 Q 共享 1 组 KV)
 *   dFF: 16384
 *   context: 8192
 *
 * 总参数: ~20B
 * fp16 显存: 40GB (推理)
 * fp8 显存: 20GB (推理)
 * 训练需要: 8×A100 (ZeRO-3) 或 4×H100
 *
 * 使用 TransformerModel (transformer3b.js) 配合 CONFIG_20B 即可实例化
 */

const { TransformerModel } = require('./transformer3b.js');

/** 20B 模型配置 */
const CONFIG_20B = {
  vocabSize: 32768,
  dModel: 6144,
  numLayers: 48,
  numHeads: 24,
  numKVHeads: 8,
  dFF: 16384,
  maxSeqLen: 8192,
  ropeBase: 10000,
  rmsNormEps: 1e-6,
};

/** 创建 20B 模型实例 */
function createTransformer20B() {
  return new TransformerModel(CONFIG_20B);
}

module.exports = { TransformerModel, CONFIG_20B, createTransformer20B };