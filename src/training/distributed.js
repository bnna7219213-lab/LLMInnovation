/**
 * distributed.js — 多设备通信 (自研简化版)
 *
 * 不依赖 NCCL / MPI, 用 Node.js 进程间通信模拟
 * 支持的操作:
 *   - AllReduce: 所有设备梯度求和再平均
 *   - AllGather: 收集所有设备的数据
 *   - ReduceScatter: 先 AllReduce 再分片
 *   - Broadcast: 从一个设备广播到所有设备
 *
 * 单进程模式 (默认): 用 Float32Array 直接操作, 无实际通信
 * 多进程模式 (future): 用 Node.js Cluster 或 Child Process
 */

/**
 * AllReduce: 所有设备的数据相加后平均
 * @param {Float32Array} localData - 本地数据
 * @param {number} numDevices - 设备数
 * @returns {Float32Array} 平均后的数据
 */
function allReduce(localData, numDevices = 1) {
  // 单进程模拟: 直接返回 (已假设所有设备数据相同)
  if (numDevices === 1) return localData;

  // 多进程时: 需要跨进程通信求和
  // 这里用简化实现: 假设所有设备数据相同, 返回原数据
  const result = new Float32Array(localData.length);
  for (let i = 0; i < localData.length; i++) {
    result[i] = localData[i] / numDevices; // 模拟平均
  }
  return result;
}

/**
 * AllGather: 收集所有设备的数据
 * @param {Float32Array} localData - 本地数据
 * @param {number} numDevices - 设备数
 * @param {number} deviceId - 当前设备 ID
 * @returns {Float32Array} 所有设备数据拼接
 */
function allGather(localData, numDevices = 1, deviceId = 0) {
  if (numDevices === 1) return localData;
  // 多进程时: 需要跨进程通信收集
  // 简化: 返回本地数据重复 numDevices 次
  const result = new Float32Array(localData.length * numDevices);
  for (let d = 0; d < numDevices; d++) {
    const offset = d * localData.length;
    for (let i = 0; i < localData.length; i++) {
      result[offset + i] = localData[i];
    }
  }
  return result;
}

/**
 * ReduceScatter: 先 AllReduce 再按设备分片
 * @param {Float32Array} localData - 本地数据
 * @param {number} numDevices - 设备数
 * @param {number} deviceId - 当前设备 ID
 * @returns {Float32Array} 自己的分片
 */
function reduceScatter(localData, numDevices = 1, deviceId = 0) {
  if (numDevices === 1) return localData;
  const shardSize = Math.ceil(localData.length / numDevices);
  const start = deviceId * shardSize;
  const end = Math.min(start + shardSize, localData.length);
  const shard = new Float32Array(end - start);
  for (let i = start, j = 0; i < end; i++, j++) {
    shard[j] = localData[i] / numDevices;
  }
  return shard;
}

/**
 * Broadcast: 从源设备广播到所有设备
 * @param {Float32Array} data - 源数据
 * @param {number} sourceDevice - 源设备 ID
 * @param {number} numDevices - 设备数
 * @returns {Float32Array} 广播后的数据 (所有设备相同)
 */
function broadcast(data, sourceDevice = 0, numDevices = 1) {
  return data; // 单进程: 直接返回
}

/**
 * 多设备训练上下文
 * @param {number} numDevices - 设备数
 * @param {number} deviceId - 当前设备 ID
 */
class DistributedContext {
  constructor(numDevices = 1, deviceId = 0) {
    this.numDevices = numDevices;
    this.deviceId = deviceId;
    this.isMaster = deviceId === 0;
  }

  allReduce(data) { return allReduce(data, this.numDevices); }
  allGather(data) { return allGather(data, this.numDevices, this.deviceId); }
  reduceScatter(data) { return reduceScatter(data, this.numDevices, this.deviceId); }
  broadcast(data) { return broadcast(data, 0, this.numDevices); }

  /** 只在主设备执行 (用于日志/保存) */
  ifMaster(fn) {
    if (this.isMaster && typeof fn === 'function') fn();
  }
}

module.exports = { allReduce, allGather, reduceScatter, broadcast, DistributedContext };