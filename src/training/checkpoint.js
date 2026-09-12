/**
 * checkpoint.js — 训练 Checkpoint 保存/加载
 *
 * 存储: 模型参数字段 (Float32Array) + 优化器状态 + 训练元数据
 * 格式: JSON (小模型) 或 JSONL (大模型, 每参数字段一行)
 */

/**
 * 保存 checkpoint
 *
 * @param {object} params - 参数字典 { name: Float32Array }
 * @param {object} optimizerState - 优化器状态
 * @param {object} metadata - 元数据 (step, loss, lr, 等)
 * @param {string} path - 保存路径
 */
function saveCheckpoint(params, optimizerState, metadata, path) {
  const fs = require('fs');
  const data = {
    step: metadata.step,
    loss: metadata.loss,
    learningRate: metadata.learningRate,
    paramNames: Object.keys(params),
    params: {},
    optimizerState,
  };

  // 将 Float32Array 转为普通数组 (JSON 可序列化)
  for (const name of data.paramNames) {
    data.params[name] = Array.from(params[name]);
  }

  fs.writeFileSync(path, JSON.stringify(data, null, 2));
  return path;
}

/**
 * 加载 checkpoint
 *
 * @param {string} path
 * @returns {{params: object, optimizerState: object, metadata: object}}
 */
function loadCheckpoint(path) {
  const fs = require('fs');
  const data = JSON.parse(fs.readFileSync(path, 'utf-8'));

  const params = {};
  for (const name of data.paramNames) {
    params[name] = new Float32Array(data.params[name]);
  }

  return {
    params,
    optimizerState: data.optimizerState,
    metadata: {
      step: data.step,
      loss: data.loss,
      learningRate: data.learningRate,
    },
  };
}

module.exports = { saveCheckpoint, loadCheckpoint };