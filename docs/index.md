# LLMInnovation 文档导航

这里是 LLMInnovation 项目的全部文档入口。建议按 Phase 顺序阅读，从 Phase 0 开始逐步构建完整技术栈。

## 总览

| 文档 | 说明 |
|---|---|
| [README](../README.md) | 项目总览：状态表、目录结构、快速开始、核心设计决策、关键约束、新增 Phase 指南 |
| [plan.md](../plan.md) | 9 章研究计划：架构选型、100B 路线、多模态管线、分阶段实施路线图 |

## 分阶段文档（按实施顺序）

| Phase | 内容 | 文档 | 测试 |
|---|---|---|---|
| Phase 0 | 基础设施：BPE 分词器 + NN 原语 + 训练基建 | [phase0-infra.md](phase0-infra.md) | 41/41 ✅ |
| Phase 1 | 3B 基线 Transformer | [phase1-3b.md](phase1-3b.md) | 56/56 ✅ |
| Phase 2 | 20B 模型 + ZeRO-3 + 分布式 | [phase2-20b-zero3.md](phase2-20b-zero3.md) | 72/72 ✅ |
| Phase 3 | Mamba + Jamba 混合架构 | [phase3-mamba-jamba.md](phase3-mamba-jamba.md) | 89/89 ✅ |
| Phase 4 | Text-to-Image Diffusion | [phase4-diffusion.md](phase4-diffusion.md) | 106/106 ✅ |
| Phase 5 | Text-to-3D（Triplane + SDS） | [phase5-text23d.md](phase5-text23d.md) | 120/120 ✅ |
| Phase 6 | VLM + 长上下文 + 最终集成 | [phase6-vlm.md](phase6-vlm.md) | 141/141 ✅ |

**总计 141/141 测试通过。**

## 各 Phase 核心模块速查

### Phase 0 — 基础设施
- `tokenizer/vocab.js` — `Vocab`（BPE 词表）
- `tokenizer/bpe.js` — `BPETokenizer`（训练/编码/解码）
- `nn/rmsnorm.js` — `rmsNorm` / `rmsNormBatch`
- `nn/swiglu.js` — `swiglu`
- `nn/rope.js` — `applyRoPE`（+ NTK 插值）
- `nn/attention.js` — `multiHeadAttention` / `matmul`
- `nn/optimizer.js` — `AdamW`
- `training/scheduler.js` — `LRScheduler`
- `training/checkpoint.js` — `saveCheckpoint` / `loadCheckpoint`
- `training/distributed.js` — `allReduce` / `allGather` / `reduceScatter` / `broadcast`

### Phase 1 — 3B 基线
- `models/text/transformer3b.js` — `TransformerModel` + `CONFIG_3B`

### Phase 2 — 20B + 训练优化
- `models/text/transformer20b.js` — `CONFIG_20B` / `createTransformer20B`
- `training/zero3.js` — `ZeRO3Shard` / `ZeRO3Context`

### Phase 3 — Mamba/Jamba
- `models/mamba/selective_scan.js` — `selectiveScanSequential` / `selectiveScanParallel`
- `models/mamba/mamba_block.js` — `MambaBlock`
- `models/hybrid/jamba.js` — `JambaModel`

### Phase 4 — Diffusion
- `models/diffusion/diffusion_schedule.js` — `generateSchedule` / `addNoise` / `reverseStep` / `ddimStep`
- `models/diffusion/vae.js` — `VAE`
- `models/diffusion/unet.js` — `UNet`
- `models/diffusion/text2image.js` — `Text2Image`

### Phase 5 — Text-to-3D
- `models/3d/triplane.js` — `Triplane`
- `models/3d/sds.js` — `SDS`
- `models/3d/text23d.js` — `Text23D`

### Phase 6 — VLM + 长上下文
- `models/vlm/vision_encoder.js` — `VisionEncoder`
- `models/vlm/vlm.js` — `VLM`
- `models/text/transformer100b.js` — `MoERouter` / `CONFIG_100B` / `estimateParams`
- `inference/kv_cache.js` — `KVCache` / `quantizeInt8` / `dequantizeInt8`
- `eval/benchmarks.js` — `perplexity` / `clozeAccuracy` / `repetitionRate` / `diversity` / `runBenchmarks`

## 常见任务指引

- **快速跑通测试** → [README 快速开始](../README.md)
- **新加一个 Phase** → [README「如何新增一个 Phase」](../README.md)
- **了解架构选型理由** → [plan.md](../plan.md)
- **查某个 API 的签名和坑** → 对应 Phase 文档的「API」与「坑」小节

## 目录结构

```
LLMInnovation/
├── README.md                — 项目总览 + 新增 Phase 指南
├── plan.md                  — 9 章研究计划
├── docs/
│   ├── index.md             — 本导航首页
│   ├── phase0-infra.md
│   ├── phase1-3b.md
│   ├── phase2-20b-zero3.md
│   ├── phase3-mamba-jamba.md
│   ├── phase4-diffusion.md
│   ├── phase5-text23d.md
│   └── phase6-vlm.md
├── src/                     — 全部源码
└── test/                    — 全部测试
```
