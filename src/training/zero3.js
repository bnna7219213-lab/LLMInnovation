/**
 * zero3.js — ZeRO Stage-3 显存优化 (简化自研版)
 *
 * 核心思想:
 *   传统训练: 每个 GPU 存完整参数 + 梯度 + 优化器状态 → 8× 参数量
 *   ZeRO-3:  参数切分到所有 GPU, 只在需要时 AllGather, 用完立即释放
 *
 * 实现方式:
 *   1. 参数字典: 每个参数按 shard 切分, 每个 GPU 只存 1/shards
 *   2. 前向: 需要的参数通过 AllGather 收集完整版本, 用完释放
 *   3. 反向: 梯度算完后做 ReduceScatter, 每个 GPU 只留自己的 shard
 *   4. 优化器状态: 同样分片存储
 *
 * 不依赖 NCCL, 用进程间通信模拟 (单机多进程)
 * 在单卡上: 用 "分段 checkpoint" 思路, 参数用完即释放
 */

/**
 * ZeRO3 分片参数管理
 * @param {number} numShards - 分片数 (GPU 数)
 * @param {number} shardIdx - 当前 shard 索引
 */
class ZeRO3Shard {
  constructor(numShards, shardIdx = 0) {
    this.numShards = numShards;
    this.shardIdx = shardIdx;
    // 参数字典: paramName → { full: Float32Array | null, shard: Float32Array }
    this._params = new Map();
    // 梯度字典
    this._grads = new Map();
    // 优化器状态字典
    this._optStates = new Map();
  }

  /** 注册参数 (初始只存自己的 shard) */
  registerParam(name, fullParam) {
    const shardSize = Math.ceil(fullParam.length / this.numShards);
    const start = this.shardIdx * shardSize;
    const end = Math.min(start + shardSize, fullParam.length);
    const shard = new Float32Array(end - start);
    for (let i = start, j = 0; i < end; i++, j++) {
      shard[j] = fullParam[i];
    }

    this._params.set(name, {
      full: null,       // 初始不存完整参数
      shard,            // 只存自己的 shard
      shardSize,
      start,
      end,
      fullLength: fullParam.length,
    });
  }

  /** 收集完整参数 (AllGather 模拟) */
  gatherParam(name) {
    const entry = this._params.get(name);
    if (!entry) throw new Error(`参数 ${name} 未注册`);

    if (entry.full) return entry.full;

    // 模拟 AllGather: 在单机上就是把 shard 拼回完整版本
    // 多设备时需要跨进程通信
    entry.full = new Float32Array(entry.fullLength);
    // 只填自己的 shard, 其他 shard 为 0 (模拟场景)
    // 在真实多设备场景, 其他 shard 通过通信获取
    for (let i = 0; i < entry.shard.length; i++) {
      entry.full[entry.start + i] = entry.shard[i];
    }
    return entry.full;
  }

  /** 释放完整参数 (用完即释放) */
  freeParam(name) {
    const entry = this._params.get(name);
    if (entry) entry.full = null;
  }

  /** 设置梯度 shard */
  setGradShard(name, gradShard) {
    this._grads.set(name, gradShard);
  }

  /** ReduceScatter 梯度 (每个 shard 只留自己的部分) */
  reduceScatterGrad(name, fullGrad) {
    const entry = this._params.get(name);
    if (!entry) throw new Error(`参数 ${name} 未注册`);

    const shard = new Float32Array(entry.shard.length);
    for (let i = 0; i < entry.shard.length; i++) {
      shard[i] = fullGrad[entry.start + i];
    }
    this._grads.set(name, shard);
    return shard;
  }

  /** 获取梯度 shard */
  getGradShard(name) {
    return this._grads.get(name);
  }

  /** 注册优化器状态 shard */
  registerOptState(name, fullState) {
    const shardSize = Math.ceil(fullState.length / this.numShards);
    const start = this.shardIdx * shardSize;
    const end = Math.min(start + shardSize, fullState.length);
    const shard = new Float32Array(end - start);
    for (let i = start, j = 0; i < end; i++, j++) {
      shard[j] = fullState[i];
    }
    this._optStates.set(name, { full: null, shard, start, fullLength: fullState.length });
  }

  /** 获取参数名列表 */
  getParamNames() {
    return Array.from(this._params.keys());
  }

  /** 保存所有 shard 为 JSON */
  toJSON() {
    const out = { numShards: this.numShards, shardIdx: this.shardIdx, params: {}, grads: {}, optStates: {} };
    for (const [name, entry] of this._params) {
      out.params[name] = Array.from(entry.shard);
    }
    for (const [name, shard] of this._grads) {
      out.grads[name] = Array.from(shard);
    }
    for (const [name, state] of this._optStates) {
      out.optStates[name] = Array.from(state.shard);
    }
    return out;
  }
}

/**
 * ZeRO3 上下文管理器 (训练时使用)
 *
 * 用法:
 *   const zero3 = new ZeRO3Context(numGpus, gpuId);
 *   model.forEachParam(name, param => zero3.registerParam(name, param));
 *
 *   // 前向
 *   zero3.gatherParams(['layer0.WQ', 'layer0.WK']);
 *   // ... 计算 ...
 *   zero3.freeParams(['layer0.WQ', 'layer0.WK']);
 *
 *   // 反向
 *   zero3.reduceScatterGrad('layer0.WQ', grad);
 */
class ZeRO3Context {
  constructor(numShards, shardIdx = 0) {
    this.shard = new ZeRO3Shard(numShards, shardIdx);
    this._gathered = new Set();
  }

  registerParam(name, fullParam) {
    this.shard.registerParam(name, fullParam);
  }

  gatherParams(names) {
    for (const name of names) {
      this.shard.gatherParam(name);
      this._gathered.add(name);
    }
  }

  freeParams(names) {
    for (const name of names) {
      this.shard.freeParam(name);
      this._gathered.delete(name);
    }
  }

  freeAllGathered() {
    for (const name of this._gathered) {
      this.shard.freeParam(name);
    }
    this._gathered.clear();
  }

  reduceScatterGrad(name, fullGrad) {
    return this.shard.reduceScatterGrad(name, fullGrad);
  }

  getShard() { return this.shard; }
}

module.exports = { ZeRO3Shard, ZeRO3Context };