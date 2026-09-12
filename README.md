# LLMInnovation — 自研多模态 LLM 研究项目

从零实现一套多模态大模型技术栈，无外部预训练模型依赖（不依赖 PyTorch/TensorFlow/Transformers），纯 JavaScript + Float32Array 手写神经网络。

## 项目状态

| Phase | 内容 | 测试 | 文档 |
|---|---|---|---|
| Phase 0 | 基础设施（BPE 分词器 + NN 原语 + 训练基建） | 41/41 ✅ | [docs/phase0-infra.md](docs/phase0-infra.md) |
| Phase 1 | 3B 基线 Transformer | 56/56 ✅ | [docs/phase1-3b.md](docs/phase1-3b.md) |
| Phase 2 | 20B + ZeRO-3 + 分布式 | 72/72 ✅ | [docs/phase2-20b-zero3.md](docs/phase2-20b-zero3.md) |
| Phase 3 | Mamba + Jamba 混合架构 | 89/89 ✅ | [docs/phase3-mamba-jamba.md](docs/phase3-mamba-jamba.md) |
| Phase 4 | Text-to-Image Diffusion | 106/106 ✅ | [docs/phase4-diffusion.md](docs/phase4-diffusion.md) |
| Phase 5 | Text-to-3D（Triplane + SDS） | 120/120 ✅ | [docs/phase5-text23d.md](docs/phase5-text23d.md) |
| Phase 6 | VLM + 长上下文 + 最终集成 | 141/141 ✅ | [docs/phase6-vlm.md](docs/phase6-vlm.md) |

**总计 141/141 测试通过。**

> 📚 完整文档入口见 [docs/index.md](docs/index.md)。

## 目录结构

```
LLMInnovation/
├── plan.md                  — 9 章研究计划（架构选型、100B 路线、分阶段实施）
├── README.md                — 本文件
├── docs/                    — 每个 Phase 的使用文档（入口：[docs/index.md](docs/index.md)）
├── src/
│   ├── tokenizer/           — BPE 分词器 (vocab.js, bpe.js)
│   ├── nn/                  — 神经网络原语 (rmsnorm, swiglu, rope, attention, optimizer)
│   ├── training/            — 训练基建 (checkpoint, scheduler, zero3, distributed)
│   ├── inference/           — 推理优化 (kv_cache)
│   ├── eval/                — 评估基准 (benchmarks)
│   └── models/
│       ├── text/            — transformer3b/20b/100b
│       ├── mamba/           — selective_scan, mamba_block
│       ├── hybrid/          — jamba
│       ├── diffusion/       — schedule, vae, unet, text2image
│       ├── 3d/              — triplane, sds, text23d
│       └── vlm/             — vision_encoder, vlm
└── test/                    — 每个 Phase 的测试 (node --test)
```

## 快速开始

```bash
# 运行全部测试
node --test test/*.js

# 运行单个 Phase 的测试
node --test test/tokenizer.test.js test/nn.test.js   # Phase 0
node --test test/vlm.test.js                          # Phase 6
```

## 核心设计决策

- **架构**：单 LLM backbone + 模态专用编码器/解码器
- **归一化**：RMSNorm（比 LayerNorm 快 ~20%，无 mean 减法）
- **FFN**：SwiGLU（LLaMA/Mistral 标准）
- **位置编码**：RoPE + NTK 插值（可外推到 200K context）
- **注意力**：GQA（Grouped Query Attention，KV 压缩）
- **分词器**：BPE（UTF-8 字节级，256 初始 token，目标 32K）
- **优化器**：AdamW（β1=0.9, β2=0.95, ε=1e-8）
- **训练**：ZeRO Stage-3 参数分片 + 分布式通信模拟

## 关键约束（写测试/扩展时注意）

1. **vocabSize ≥ 256**：BPE 从 256 字节 token 起步，模型 vocabSize 必须 ≥ 256
2. **因果掩码**：预分配 maxSeqLen×maxSeqLen 一次，每层截取子视图，避免逐层分配 bug
3. **优化器 step**：`update()` 中 step 只递增 1 次（不是每参数递增）
4. **测试导入**：`beforeEach` 需从 `node:test` 显式导入；test/ 下用 `../src/...` 相对路径
5. **`node --test`**：内置测试运行器，无需额外依赖

## 如何新增一个 Phase（开发指南）

按以下顺序推进，每步都有明确的完成判据。遵循「先让测试绿，再写文档」的原则。

### 第 1 步：明确目标与范围

在 `plan.md` 里找到（或新增）对应章节，确认：

- 这个 Phase 要交付哪些**模块**（类/函数）
- 每个模块的**输入/输出维度**（写进文件头注释）
- 依赖哪些已有模块（跨目录 require 用相对路径）

### 第 2 步：创建源码

```bash
# 按领域建目录（已有：tokenizer/nn/training/inference/eval/models/*）
src/models/<domain>/<module>.js
```

代码规范（全项目一致）：

- 纯 JS + `Float32Array`，**不引入任何外部依赖**
- 每个文件头部写清：作用、公式、API 签名、维度
- 统一 `module.exports = { ClassName }` 或 `{ fnName }`
- 跨目录 require 用正确相对路径（如 `models/3d/` 引用 diffusion 用 `../diffusion/...`）

### 第 3 步：写测试

```bash
test/<phase>.test.js
```

测试约定（见「关键约束」），另注意：

- **用微型配置**验证与生产配置相同的代码路径（生产配置内存会爆炸）
- 首行显式导入：`const { describe, it, beforeEach } = require('node:test');`
- 路径：`require('../src/...')`（不是 `../../src/...`）
- 覆盖维度正确性、无 NaN、边界条件、序列化往返

### 第 4 步：迭代到全绿

```bash
node --test test/<phase>.test.js
```

- 首次运行通常有失败，逐条看 `not ok` 报错定位
- 数值问题（NaN/溢出）优先查 softmax、指数、矩阵维度
- 修完单文件记得 `node --check` 验证语法

### 第 5 步：全量回归

```bash
node --test test/*.js
```

确认新 Phase 不破坏已有测试（累计 141/141 应只增不减）。

### 第 6 步：写文档 + 更新 README

1. 新建 `docs/phaseN-xxx.md`（参照现有文档结构：测试命令 + 可运行示例 + API 参考 + 配置表 + 坑）
2. 在 README 的「项目状态」表加一行，更新总计
3. 在「目录结构」补新目录
4. 更新 `todo` 任务状态

### 模板

**源码骨架**（`src/models/<domain>/<module>.js`）：

```js
/**
 * <module>.js — 作用一句话
 *
 * 公式 / 数据流 / 参考架构
 * 配置示例（生产 + 微型）
 */
class Foo {
  constructor(config) { /* 校验维度、初始化权重 */ }
  forward(x) { /* 纯计算，返回 Float32Array */ }
  countParams() { /* 统计参数量 */ }
}
module.exports = { Foo };
```

**测试骨架**（`test/<phase>.test.js`）：

```js
const assert = require('assert');
const { describe, it, beforeEach } = require('node:test');

let Foo = null;
beforeEach(() => {
  ({ Foo } = require('../src/models/<domain>/<module>.js'));
});

describe('Foo', () => {
  it('维度正确', () => {
    const f = new Foo({ /* 微型配置 */ });
    const out = f.forward(/* 输入 */);
    assert.strictEqual(out.length, /* 期望 */);
  });
});
```

### 检查清单

- [ ] 无外部依赖，纯 JS + Float32Array
- [ ] 文件头注释含 API 签名与维度
- [ ] 测试用微型配置，全绿
- [ ] 全量回归 `node --test test/*.js` 只增不减
- [ ] `docs/phaseN-xxx.md` 已建，README 状态表已更新
