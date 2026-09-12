/**
 * benchmarks.js — 评估基准
 *
 * 自研 LLM 的核心评估指标:
 *   1. Perplexity (困惑度): exp(-avg log P(token|context))
 *   2. 准确率: 多选题 / 完形填空
 *   3. 生成质量: 重复率、多样性
 *
 * 无外部依赖, 纯 JS 实现
 */

/**
 * 计算困惑度
 * @param {Array<number>} tokenIds - token 序列
 * @param {Function} forwardFn - (contextIds) => logits [1, vocabSize] (下一个 token 的 logits)
 * @param {number} vocabSize
 * @returns {number} perplexity
 */
function perplexity(tokenIds, forwardFn, vocabSize) {
  if (tokenIds.length < 2) return NaN;

  let sumLogProb = 0;
  let count = 0;

  for (let i = 0; i < tokenIds.length - 1; i++) {
    const context = tokenIds.slice(0, i + 1);
    const logits = forwardFn(context);
    const nextToken = tokenIds[i + 1];

    // softmax over logits
    let maxVal = -Infinity;
    for (let j = 0; j < vocabSize; j++) {
      if (logits[j] > maxVal) maxVal = logits[j];
    }
    let sum = 0;
    for (let j = 0; j < vocabSize; j++) {
      sum += Math.exp(logits[j] - maxVal);
    }
    const logProb = (logits[nextToken] - maxVal) - Math.log(sum);
    sumLogProb += logProb;
    count++;
  }

  return Math.exp(-sumLogProb / count);
}

/**
 * 完形填空准确率 (top-1)
 * @param {Array<Array<number>>} contexts - 每个样本的上下文 token
 * @param {Array<number>} targets - 每个样本的目标 token
 * @param {Function} forwardFn
 * @returns {number} 0-1 准确率
 */
function clozeAccuracy(contexts, targets, forwardFn) {
  let correct = 0;
  for (let i = 0; i < contexts.length; i++) {
    const logits = forwardFn(contexts[i]);
    let maxVal = -Infinity, maxIdx = 0;
    for (let j = 0; j < logits.length; j++) {
      if (logits[j] > maxVal) { maxVal = logits[j]; maxIdx = j; }
    }
    if (maxIdx === targets[i]) correct++;
  }
  return contexts.length > 0 ? correct / contexts.length : 0;
}

/**
 * 生成重复率 (越低越好)
 * @param {Array<number>} generated - 生成的 token 序列
 * @param {number} n - n-gram 长度 (默认 2)
 * @returns {number} 重复的 n-gram 比例
 */
function repetitionRate(generated, n = 2) {
  if (generated.length < n) return 0;
  const seen = new Set();
  let repeats = 0;
  for (let i = 0; i <= generated.length - n; i++) {
    const gram = generated.slice(i, i + n).join(',');
    if (seen.has(gram)) repeats++;
    seen.add(gram);
  }
  const totalGrams = generated.length - n + 1;
  return totalGrams > 0 ? repeats / totalGrams : 0;
}

/**
 * 生成多样性 (不同样本间 unique n-gram 比例)
 * @param {Array<Array<number>>} samples - 多个生成样本
 * @param {number} n
 * @returns {number} 0-1, 越高越多样
 */
function diversity(samples, n = 3) {
  const allGrams = new Set();
  let total = 0;
  for (const sample of samples) {
    for (let i = 0; i <= sample.length - n; i++) {
      allGrams.add(sample.slice(i, i + n).join(','));
      total++;
    }
  }
  return total > 0 ? allGrams.size / total : 0;
}

/**
 * 简单基准运行器
 * @param {Array<{name: string, fn: Function}>} benchmarks
 * @returns {Array<{name: string, result: *, ms: number}>}
 */
function runBenchmarks(benchmarks) {
  const results = [];
  for (const bench of benchmarks) {
    const start = Date.now();
    const result = bench.fn();
    const ms = Date.now() - start;
    results.push({ name: bench.name, result, ms });
  }
  return results;
}

module.exports = {
  perplexity,
  clozeAccuracy,
  repetitionRate,
  diversity,
  runBenchmarks,
};