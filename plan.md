# plan.md — LLM 自研开发计划：从底层到百亿级多模态

> **项目**: LLMInnovation — 从零自研多模态大语言模型研发平台
>
> **定位**: 不基于任何开源模型（不做 LoRA/微调），从 token 分词到训练管线全部自研。
> 类比关系：mini-engine 里的 `MiniGPT.js` 从零手写 Transformer 的哲学，扩展到完整的大模型研发流水线。
>
> **版本**: v0.1 | 更新日期: 2026-08-26
>
> **核心原则**:
> 1. **不依赖外部预训练模型**——从 embedding 层到输出层全部自研
> 2. **训练数据自处理**——不直接下载别人的训练集，而是构建可复现的数据管线
> 3. **架构多样性**——同时探索 Transformer 及其替代架构（Mamba/RWKV/线性注意力）
> 4. **渐进式规模**——从 3B（CPU 可跑）到 20B（单卡 GPU）到 100B+（集群），每一步都有明确的验证标准
> 5. **多模态覆盖**——text-to-text 为基座，逐步扩展 image/video/3D/音频

---

## 第 0 章：现有模型架构全景（先讨论，再设计）

> 这一章不做自研计划，纯粹梳理"当前业界已有什么"——理解现状是设计未来的前提。

---

### 0.1 Transformer 系谱（2017 → 2024）

```
原始 Transformer (Vaswani 2017)
  │
  ├── GPT 系（decoder-only, causal LM）
  │   ├── GPT-1/2/3        — dModel 2048, 96 层, 175B
  │   ├── GPT-4            — 推测 MoE, 1-2T（推测）
  │   ├── LLaMA 1/2/3      — RoPE + SwiGLU + 无 QKV bias
  │   ├── Mistral            — GQA + Sliding Window + No bias
  │   └── Qwen 2             — SwiGLU + GQA + RoPE + MoE 版本
  │
  ├── T5 / BART 系（encoder-decoder）
  │   └── BART / T5-XXL    — 11B, 22 层 encoder + 22 层 decoder
  │
  └── PaLM 系（decoder-only, 工业级）
      └── PaLM 2           — 340B, GShard 分片训练

核心创新时间线:
  2017  Transformer (scaled dot-product attention)
  2020  GPT-3 (scale law: 参数量 × 数据量 × 计算量的乘积)
  2021  SwiGLU (Shazeer, 用 Swish-Gated Linear Unit 替代原始 FFN)
  2021  ALiBi (Press et al., 无需位置编码, 用线性偏置)
  2021  RoPE (Su et al., 旋转位置编码, 相对位置敏感)
  2022  FlashAttention (Dao et al., IO-aware, O(n²) 但常数极小)
  2022  GQA (Ainslie et al., 分组查询注意力)
  2022  LoRA (Hu et al., 低秩适配, 不是自研范畴但值得了解)
  2023  RLHF (InstructGPT, PPO 优化人类偏好)
  2023  DPO (Direct Preference Optimization, 无 reward model)
  2023  Mamba (Gu et al., SSM + 选择性状态, 线性时间)
  2023  RWKV v4/v5       (线性注意力 + RNN 等价性)
  2023  MoE (GShard/Moja, 稀疏激活, 2-10% 激活率)
  2024  DeepSeek-V2        — MoE + MQA + Grouped-MLP
  2024  Llama 3.1          — 128K context, GQA, SwiGLU
  2024  Qwen 2.5           — 128K context, MoE 版本 128 专家
```

---

### 0.2 关键架构组件对比

#### 0.2.1 位置编码

| 方法 | 公式/机制 | 优势 | 劣势 | 代表模型 |
|---|---|---|---|---|
| **Sinusoidal** (2017) | sin/cos 固定频率 | 无需训练, 外推到长序列 | 高频分辨率低, 短序列浪费 | GPT-2 |
| **Learned** | embedding 可训练 | 学到最优 | 无法外推到训练外长度 | T5 |
| **ALiBi** (2021) | attention logits 加线性偏置 `-m * |i-j|` | 无需学习, 长序列外推好 | 超参数 `m` 需要调 | Falcon |
| **RoPE** (2021) | 旋转 query/key 向量, 用角度编码相对位置 | 相对位置信息丰富, 外推好 | 计算稍多, NTK-RoPE 微调 | LLaMA, GPT-4 |
| **RoFormer** (2023) | RoPE + 插值, 支持动态长度 | 支持 256K+ context | 需要微调才能外推 | Llama 3 |
| **YaRN** (2023) | RoPE 的 y-axis rescaling | 外推到 100K+ | 实现复杂 | Yi-34B |

**自研选型建议**: RoPE + NTK 插值。原因：数学形式简洁（旋转矩阵在 2D 子空间上操作），LLaMA/Mistral 已验证，NTK 插值可外推到 200K context。

#### 0.2.2 注意力变体

| 方法 | 计算复杂度 | 内存带宽需求 | 代表 |
|---|---|---|---|
| **MHA** (Multi-Head) | O(n²dk) | O(n²dk) — QKV 都是 numHeads 个 | LLaMA 2 |
| **GQA** (Grouped Query) | O(n²dk / numGQAGroups) | O(n²dk / numGQAGroups) — Q 分组共享 KV | Mistral, Llama 3 |
| **MQA** (Multi-Query) | O(n²dk) | O(n²dk) — Q 每个头, KV 只有一个 | DeepSeek V2, PaLM 2 |
| **Sliding Window** | O(n · windowSize) | O(n · windowSize) | Mistral (每 4 层 4096 窗口) |
| **FlashAttention** | O(n²) 但 IO 优化 | O(n) 显存 | 通用加速, 任何模型的底层 |
| **LongContext** (Llama 3) | O(n · windowSize) | 每 8 层 256K 局部 + 4 层全局 | Llama 3.1 128K |

**自研选型建议**：
- 小模型（<10B）：GQA（4 组共享 KV）→ 推理速度最优
- 大模型（>50B）：MQA + FlashAttention → 推理显存最省
- 长上下文（>64K）：滑动窗口 + 部分全局层交替

#### 0.2.3 MLP / FFN 变体

| 方法 | 公式 | 参数 | 优势 |
|---|---|---|---|
| **GELU FFN** | W2 · GELU(W1x) | dModel × 4dModel | 原始 GPT, 简单 |
| **SwiGLU** (2021) | (Swish(W1x) · W2x) W3 | dModel × 8dModel | LLaMA 全系, 比 GELU 快 15% |
| **Gated MLP** | 同 SwiGLU, 只是激活函数不同 | 同 SwiGLU | 通用 |
| **MoE** (Mixture of Experts) | top-k 路由, 选激活 k 个专家 | numExperts × dModel × 2dModel | 100B+ 模型, 稀疏激活 |
| **Grouped MLP** (DeepSeek V2) | 分组 + SwiGLU, 减少参数 | 约 80B 激活参数 | DeepSeek V2, 高效 |

**自研选型建议**：SwiGLU（小模型） + MoE 8/16（大模型，激活率 6.25%-12.5%）。

#### 0.2.4 归一化

| 方法 | 公式 | 代表 |
|---|---|---|
| **LayerNorm** | 逐维度 z-score | 原始 Transformer |
| **RMSNorm** (2019) | 除以 RMS, 无 mean 减去 | LLaMA, GPT-4, 现代标准 |
| **GroupNorm** | 每组 z-score | 视觉模型 |
| **NormFormer** | 去掉 LayerNorm | 实验性 |

**自研选型建议**：RMSNorm（计算比 LayerNorm 快 20%，现代主流）。

#### 0.2.5 初始化

| 方法 | 权重初始化 | Embedding 初始化 |
|---|---|---|
| **Xavier / Glorot** | Uniform(-√6/(fan_in+fan_out), ...) | Uniform(-0.1, 0.1) |
| **He** | Normal(0, √2/fan_in) | 同上 |
| **LLaMA 风格** | 权重: Normal(0, 0.02), QK 权重: Normal(0, 0.02/√dK) | Uniform(-0.5, 0.5) |

**自研选型建议**：LLaMA 风格——QK 权重缩小 dK 倍避免注意力爆炸。

---

### 0.3 非 Transformer 架构（2023-2024 前沿）

#### 0.3.1 Mamba / SSM（选择性状态空间模型）

**核心思想**：用选择性扫描（selective scan）替代 self-attention，从 O(n²) 降到 O(n) 时间复杂度。

```
RNN: h_t = h_{t-1} × A_t + x_t × B_t        — 隐藏状态递归
SSM: h_t = A·h_{t-1} + B·x_t                — 线性递归
Mamba: h_t = h_{t-1} × A_t + x_t × B_t · C_t  — 选择性参数（数据依赖）
```

| 维度 | Mamba | Transformer |
|---|---|---|
| 时间复杂度 | O(n) | O(n²) |
| 推理显存 | O(n) | O(n²) (KV cache) |
| 并行训练 | O(n) 但 scan 依赖长序列 | O(n²) 但完全并行 |
| 长序列表现 | 极优（1M+ tokens） | 需要 RoPE + 插值 |
| 小模型表现 | 不优于 Transformer | 基准 |
| 训练稳定性 | 需要仔细的初始化 | 标准 |

**代表工作**：Mamba-1B, Mamba-2B, Mamba-13B。Mamba-2 引入 selective scan v2，训练速度提升 5x。

**自研可行性**：可行。核心操作（selective scan）在 tfjs 中可用 tf.while_loop 或手写实现，不依赖 cuDNN。

#### 0.3.2 RWKV（RNN with KV）

**核心思想**：将 Transformer 的 KV 缓存机制融入 RNN，实现"用 RNN 结构获得 Transformer 效果"。

```
a_t = A + 0.06 / (t+1)^0.3          # 衰减矩阵（随时间变化）
k_t = Wk(x_t)                        # 键
v_t = Wv(x_t)                        # 值
KV_t = a_t * KV_{t-1} + (1-a_t) * k_t * v_t  # 递归 KV 累积
y_t = sum(KV_t · queries)            # 输出
```

RWKV v5 在 100K+ context 上表现接近 GPT-3 级 Transformer。

| 维度 | RWKV | Transformer |
|---|---|---|
| 推理 | 纯 RNN, 无需 KV cache | 需要 KV cache |
| 显存 | 常数 O(1) | O(seqLen) |
| 训练 | 顺序, 无并行 | 并行, 快 |
| 长序列 | 极优 | 需要插值 |
| 生态 | 小 | 大 |

**自研可行性**：可行。核心公式简单，纯矩阵运算。

#### 0.3.3 线性注意力（Linear Attention / Performer / Linformer）

**核心思想**：将 softmax(QKᵀ) 分解为 φ(Q) · φ(K)ᵀ 形式，避免 n² 的注意力矩阵。

```
原始: y = softmax(QKᵀ / √dk) · V
线性: y = (φ(Q) · φ(K)ᵀ / (φ(Q)·1) · (φ(K)ᵀ · V))    # Kernel trick
```

| 方法 | Kernel φ | 复杂度 |
|---|---|---|
| **Performant** (2020) | Random Fourier Features | O(n) |
| **Linformer** (2020) | 低秩投影 | O(n) |
| **Linear-attention** (2023) | Elu(Q) + 1, Elu(K) + 1 | O(n) |
| **FNet** (2021) | FFT(Q) 替代 attention | O(n log n) |

**自研可行性**：可行，但效果不如 Transformer/Mamba 接近 SOTA。作为替代方案保留。

#### 0.3.4 混合架构

| 模型 | 架构组合 | 说明 |
|---|---|---|
| **Jamba** (AI21, 2024) | 交替 Mamba 层 + Transformer 层 | 每 2 层交替, 混合优点 |
| **Mamba-Transformer** | Mamba 做粗粒度, Transformer 做精细 | 实验性 |
| **Hyena** (2023) | SSM + 全局卷积 | 长序列极快 |
| **Griffin** (2023) | SSM + attention 混合 | 每层两种机制 |

**自研选型建议**：Phase 3 实现 Jamba 风格混合架构。原因：
- Mamba 层处理"全局依赖"（如整篇文档的主题）
- Transformer 层处理"局部语义"（如一句话内词语搭配）
- 交替结构在长序列 + 短序列上都有好表现

---

### 0.4 推理优化技术全景

| 技术 | 作用 | 加速比 | 自研难度 |
|---|---|---|---|
| **KV Cache** | 增量推理不重算前文 | 100x+ (相对无 cache) | 低 |
| **FlashAttention** | IO 感知 attention 计算 | 2-3x | 中（需手写） |
| **Quantization (INT8/INT4)** | 权重压缩, 减少显存 | 2-4x | 中 |
| **Speculative Decoding** | 小模型猜, 大模型验证 | 2-3x | 高 |
| **Continuous Batching** | 多请求混合处理 | 2-5x | 中 |
| **PagedAttention (vLLM)** | 分页 KV cache, 零碎片 | 2-5x | 高 |
| **Medusa / Parallel Decoding** | 多 token 并行预测 | 3-5x | 中 |

---

### 0.5 多模态架构全景

| 模态 | 核心架构 | 代表模型 |
|---|---|---|
| **Text-to-Text** | Decoder-only Transformer | GPT-4, LLaMA 3 |
| **Text-to-Image** | Diffusion U-Net (3D + 2D) | Stable Diffusion, DALL-E 3 |
| **Text-to-Video** | 3D Diffusion U-Net (时空) | Sora, Runway Gen-2 |
| **Text-to-3D** | Triplane Diffusion + Score Distillation | Tripo, LGM |
| **Vision-Language** | LLM + Vision Encoder 拼接 | CLIP + LLaVA, GPT-4V |
| **Audio-Text** | Audio Codec + LLM | AudioLM, Bark |
| **Embodied AI** | VLA (Vision-Language-Action) | RT-2, Octo |
| **Multi-modal LLM** | 单模型所有模态 | GPT-4o, Gemini Ultra |

---

### 0.6 训练方法论全景

| 阶段 | 目标 | 方法 |
|---|---|---|
| **Pre-training** | 学习世界知识 + 语言规律 | Next-token prediction (因果 LM) |
| **Supervised Fine-tuning (SFT)** | 学习指令遵循 | 人工标注的 (指令, 回答) 对 |
| **RLHF** | 学习人类偏好 | PPO + reward model |
| **DPO** | 同上, 不需要 reward model | 直接优化偏好 |
| **KTO** | 无条件偏好 + 反例 | Kahneman-Tversky |
| **RLAIF** | 用 AI 反馈代替人类 | Constitutional AI, Claude 系列 |
| **Self-play** | 自博弈 + 自我迭代 | AlphaGo, ChatGPT 自我改进 |
| **Continual Learning** | 增量更新, 不遗忘 | 参数高效 + 正则化 |

---

## 第 1 章：自研 100B+ 模型的算法路线图

> 从底层开始，不依赖任何外部预训练权重。

---

### 1.1 Tokenizer（第 1 步，必须先做）

**为什么先做**：Tokenizer 是模型的"输入层"，词表大小直接影响 embedding 维度、训练速度、输出质量。它是唯一一个**不需要 GPU、不需要大规模数据**就能自研的部分。

**方案：BPE（Byte-Pair Encoding）自研实现**

```
流程:
1. 收集种子语料（纯文本，约 100MB，从 Wikipedia / books / code 等来源）
2. 初始化：单个 UTF-8 字节为 token（256 个初始 token）
3. 迭代：找到最高频的相邻 token 对，合并为一个新 token
4. 重复直到词表大小达到目标（32K 或 64K）

关键细节:
- UTF-8 解码为字节序列，BPE 在字节级操作
- 高频 token 对选择用堆（max-heap），O(n log n)
- 合并用并查集（union-find）或 Trie 优化

输出:
- vocab.json: { "byte_0x00": 0, "byte_0x01": 1, ..., "高频 token": 32000 }
- merges.txt: 每行一条合并规则 ["a", "b"]
```

**词表大小决策**：

| 词表大小 | 优点 | 缺点 |
|---|---|---|
| 8K | 序列更短，embedding 层小 | 生僻词碎片化严重 |
| 32K | 平衡 | LLaMA 3 标准 |
| 64K | 更少碎片 | embedding 层大 2 倍 |
| 100K+ | GPT-4 级别 | 内存开销大 |

**选型**：32K 词表，BPE，UTF-8 字节级。

**训练数据（自生成）**：

```
种子语料来源（全部合法/公开）:
- 本地预装的中英文书籍（如项目自带的 .txt）
- 项目自身代码 + 文档（mini-engine 的 ~40 个 JS 文件, ~100K 字符）
- 从 Wikipedia 通过其公开 API 获取的 100MB 纯文本（CC-BY-SA）
- 从 arxiv 公开论文抓取（arxiv.org 允许爬虫）
```

---

### 1.2 Embedding 层

**方案**：Standard Embedding（词表维度 × dModel），无额外创新。

```js
class Embedding {
  constructor(vocabSize, dModel) {
    // Xavier 初始化, 但 embedding 用 Uniform(-0.5, 0.5)
    this.W = tf.variable(tf.randomUniform([vocabSize, dModel], -0.5, 0.5));
  }
  forward(tokenIds) {
    return tf.gather(this.W, tokenIds);  // [batch, seqLen, dModel]
  }
}
```

**关键**：Embedding 权重和输出层权重共享（tie embedding），节省 50% 参数。

---

### 1.3 位置编码：RoPE + NTK 插值

**参考**：LLaMA / Mistral / MiniGPT（本项目已验证）

**实现**：

```js
function applyRoPE(x, posIds, dModel) {
  // x: [batch, seqLen, dModel]
  // posIds: [batch, seqLen]
  // dModel: 隐藏维度
  // RoPE: 每两个维度一组, 2D 子空间旋转
  //   for i in range(dModel // 2):
  //     angle = pos * i / (10000 ** (2*i / dModel))
  //     x[:, :, 2*i]   = x[:, :, 2*i]   * cos(angle) - x[:, :, 2*i+1] * sin(angle)
  //     x[:, :, 2*i+1] = x[:, :, 2*i]   * sin(angle) + x[:, :, 2*i+1] * cos(angle)
  // NTK 插值: base = 10000, scale = 8 → base = 10000 * 8^((scale/1) - 1)
}
```

---

### 1.4 注意力机制：FlashAttention-2 思路（手写）

**不依赖 cuDNN**，用 tfjs 实现。核心优化：

```
标准 Attention:
  scores = Q @ K.T                           # [n, dK]
  scores = softmax(scores / sqrt(dK) + mask)  # [n, n]
  output = scores @ V                         # [n, dK]
  → 需要显存 O(n²) 存 attention matrix

FlashAttention 思路（IO 优化）:
  将序列分块, 块大小 = block_size (通常 128)
  对每一块对 (Qi, Kj):
    partial_scores = Qi @ Kj.T / sqrt(dK)
    partial_scores = softmax(partial_scores)  # 每行独立 softmax
    # 用 online softmax 公式累积:
    #   new_output = softmax_i * V_i + old_output
    // 关键: 不需要存储完整 n×n 矩阵
    → 显存 O(n * block_size), 时间不变 O(n²) 但常数极小
```

**自研难度**：中。核心难点是 tfjs 没有 `flash_attention` 原语，需要手写 block-by-block 的 online softmax。

---

### 1.5 FFN：SwiGLU

```js
function swiglu(x, W1, W2, W3) {
  // W1: [dModel, dFF], W2: [dModel, dFF], W3: [dFF, dModel]
  // output = (swish(W1x) * W2x) @ W3
  const a = tf.sigmoid(tf.linearProject(x, W1));  // swish(x) = x * sigmoid(x)
  const b = tf.linearProject(x, W2);
  return tf.linearProject(tf.mul(a, b), W3);
}
```

**dFF 选择**：

| 模型大小 | dModel | dFF | 说明 |
|---|---|---|---|
| 3B | 2560 | 6912 | dFF = 2.67 × dModel |
| 20B | 6144 | 16384 | dFF = 2.67 × dModel |
| 100B | 8192 | 24576 | dFF = 3 × dModel |
| 400B | 6144 | 32768 | MoE 版本, dFF 更大 |

---

### 1.6 归一化：RMSNorm

```js
function rmsNorm(x, gamma, eps = 1e-6) {
  // x: [..., dModel]
  // rms = sqrt(mean(x², axis=-1))
  // output = x / rms * gamma
  const rms = tf.sqrt(tf.mean(tf.square(x), -1));
  return tf.mul(tf.div(x, rms), gamma);
}
```

---

### 1.7 层级结构

```
Transformer Block (自研版):

  x = input
  x = x + SwiGLU(RMSNorm(x), ...)   # FFN residual
  x = x + MultiHeadAttention(RMSNorm(x), ...)  # Attention residual
  output = x
```

**Pre-Norm vs Post-Norm**：

| 方案 | 公式 | 优势 | 劣势 |
|---|---|---|---|
| **Pre-Norm** | LayerNorm(x) + SubLayer(x) | 梯度稳定, 深层可训 | 训练速度稍慢 |
| **Post-Norm** | SubLayer(x) + LayerNorm | 训练快 | 深层梯度消失 |
| **RMSNorm + Pre-Norm** (LLaMA 风格) | RMSNorm(x) + SubLayer(x) | 现代主流 | 无 |

**选型**：Pre-Norm + RMSNorm。

---

### 1.8 并行训练策略（100B+ 必需）

**数据并行（DP）**：模型副本在多个 GPU 上，梯度平均。
- 100B 参数需要 ~800GB 显存（fp16），单卡 A100 80GB 不够
- 数据并行本身不能解决问题，需要配合张量并行

**张量并行（TP）**：单个模型层切分到多个 GPU。
- `matmul(A, W)` 把 W 按行或列切分
- 每层前需要 AllReduce 聚合

**流水线并行（PP）**：不同层在不同的 GPU 上。
- 每个 GPU 跑模型的 1/PP 层
- 微批次流水，减少 bubble 时间

**专家并行（EP）**：MoE 场景，不同专家在不同 GPU 上。

**自研选型**（100B 目标）：

```
100B 模型配置:
  - 16 层, 每层 8192 dim, 128 个 MoE 专家, top-2 激活
  - 总参数 100B, 激活参数 ~15B
  - 张量并行: TP=2 (每张卡跑 1 个 matmul 分片)
  - 专家并行: EP=8 (8 个 GPU 各跑 16 个专家)
  - 流水线并行: PP=4 (每 4 层一组)
  - 总计: 2 × 8 × 4 = 64 张 GPU
```

**如果资源有限（单卡）的方案**：

| 方案 | 适用场景 | 实现难度 |
|---|---|---|
| **ZeRO-3** (by DeepSpeed 思路) | 单卡 100B+ | 中（需要 checkpointing 优化） |
| **Activation Recomputation** | 减少显存 | 低 |
| **8-bit Adam** (by bitsandbytes) | 优化器显存减半 | 低 |
| **LoRA + 冻结主干** | 微调 100B | 低 |

---

### 1.9 训练数据管线

**自研原则：不直接下载别人的训练集**。

#### 1.9.1 Pre-training 语料（~1T tokens）

| 来源 | 获取方式 | 规模估算 |
|---|---|---|
| Common Crawl | 公开 URL, HTTP 抓取 | 无限 |
| GitHub 公开代码 | GitHub API, 只抓 license 允许的仓库 | ~500B tokens |
| arXiv 论文 | arXiv API 抓取 | ~50B tokens |
| 维基百科 | Wikipedia API (CC-BY-SA) | ~10B tokens |
| 本地书籍 | 项目自带的 .txt | 忽略不计 |
| 合成数据 | 自研代码生成器 | 按需 |
| **总计目标** | | **~1T tokens** |

**数据清洗管线**（自研）：

```
1. 去重: SimHash 或 MinHash
2. 语言检测: fastText (轻量, CPU 可跑)
3. 低质量过滤: 词频熵 < 阈值 → 丢弃
4. 代码识别: 正则 + 缩进模式 → 单独标记
5. 格式化: JSONL, 每行一个 document
```

#### 1.9.2 SFT 数据

```
格式:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

来源:
- 人工撰写（~5000 条高质量, 覆盖 100+ 任务类型）
- 自研数据生成器（用代码生成多样化的指令 + 答案）
- 公开指令数据集（Alpaca / UltraChat, CC 授权）
```

#### 1.9.3 Preference 数据（RLHF/DPO）

```
格式:
{"prompt": "...", "chosen": "...", "rejected": "..."}

来源:
- 人工标注（~5000 对）
- 自研合成（用两个不同策略生成回答，选更好的作为 chosen）
```

---

### 1.10 训练课程（Curriculum）

```
Stage 1: Pre-training (1T tokens)
  - 学习率: 3e-4 → 0 (cosine decay)
  - 优化器: AdamW (beta1=0.9, beta2=0.95, eps=1e-8)
  - 梯度裁剪: 1.0
  - 权重衰减: 0.1
  - Warmup: 2000 steps
  - 目标: loss < 3.0 (perplexity < 20)
  - 预计时间: 3-6 个月 (100B 模型, 64×A100)

Stage 2: SFT (500K instructions)
  - 学习率: 2e-5
  - 训练轮数: 3 epoch
  - 目标: 指令遵循准确率 > 85%

Stage 3: DPO (100K pairs)
  - 学习率: 5e-6
  - 目标: chosen 比 rejected 的 logit 高 2 个 logit 单位
```

---

## 第 2 章：不同参数规模的模型配置

### 2.1 3B 模型（CPU 可跑, 自研基线）

```
配置:
  vocabSize: 32768
  dModel:    2560
  numLayers: 32
  numHeads:  32  (dK = 80)
  numGQA:    8   (每组 4 个 Q 共享 1 组 KV)
  dFF:       6912
  总参数:    ~3.2B
  context:   4096 (短), 可扩展到 8K

显存需求:
  fp16: 6.4 GB  (3B × 2 bytes)
  fp8:  3.2 GB
  fp16 + activation: ~16 GB (A100 40GB 足够)

训练:
  batch_size: 16 (per device)
  seqLen:     2048
  gradient accumulation: 8
  有效 batch: 16 × 8 = 128 tokens/device
  总 token 数: 500B (半 Pre-training)
  预计时间: 2-4 周 (4×A100)
```

---

### 2.2 20B 模型（单卡 GPU 可推理, 8 卡可训练）

```
配置:
  vocabSize: 32768
  dModel:    6144
  numLayers: 48
  numHeads:  24  (dK = 256)
  numGQA:    8   (每组 3 个 Q 共享 1 组 KV)
  dFF:       16384
  总参数:    ~20B
  context:   8192

显存需求:
  fp16: 40 GB  (20B × 2 bytes)
  fp8:  20 GB
  推理:  单卡 A100 80GB 可跑 fp8
  训练:  需要 8×A100 (ZeRO-3) 或 4×H100

训练:
  batch_size: 4
  seqLen:     4096
  有效 batch: 64K tokens
  总 token 数: 2T (完整 Pre-training)
  预计时间: 3-6 个月 (8×A100)
```

---

### 2.3 100B 模型（集群, 工业级）

```
配置:
  vocabSize: 32768
  dModel:    8192
  numLayers: 64
  numExperts: 128  (MoE, top-2 激活)
  numGQA:    64   (MQA 风格)
  dFF:       24576
  激活参数:  ~15B (15% 激活率)
  context:   32768

显存需求:
  fp16: 200 GB (全部参数), 激活 48 GB
  训练: 需要 64×A100 (TP=2, EP=8, PP=4)
  推理: 单卡 H100 80GB 可跑 fp4 (4bit)

训练:
  batch_size: 1
  seqLen:     8192
  有效 batch: 512K tokens (通过 gradient accumulation)
  总 token 数: 5T
  预计时间: 6-12 个月 (64×A100)
```

---

## 第 3 章：多模态扩展

### 3.1 Text-to-Text（基座模型，第 1 步）

- 架构：Decoder-only Transformer（见 第 1 章）
- 训练数据：Pre-training 1T tokens + SFT 500K + DPO 100K
- 输出：`src/models/text/text2text.js`
- 推理 API：`generate(prompt, { maxTokens, temperature })`

---

### 3.2 Text-to-Image（第 2 步，基于 Diffusion）

**架构**：

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│ text 输入     │───▶│ CLIP/Laion Encoder │───▶│ 32 维条件向量 │
└──────────────┘    └───────────────────┘    └──────┬───────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ Text-Conditioned     │
                                          │ 3D U-Net (Diffusion) │
                                          │ 输入: 64×64 noise    │
                                          │ 条件: 文本向量        │
                                          │ 输出: 64×64 latent   │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ VQGAN Decoder        │
                                          │ 64×64 latent → 1024×1024 图像 │
                                          └──────────────────────┘
```

**U-Net 结构**：

```
Encoder:  4 down blocks, channels 128→256→512→1024→2048
Bottleneck: 2× Self-Attention
Decoder:   4 up blocks, channels 2048→1024→512→256→128
输出: 64×64 latent (不是最终图像)

参数: ~850M
```

**Diffusion 采样**：

```
步数: 20 (DDIM)
噪声调度: cosine
损失: L1 loss on noise prediction
训练数据: 100 万 (text, image) 对 (LAION-5B 子集, CC 授权)
预计训练时间: 4-6 周 (8×A100)
```

---

### 3.3 Text-to-Video（第 3 步，3D Diffusion）

**架构**：

```
输入: text + 首帧图像 (可选)
  │
  ▼
3D U-Net:
  - 空间维度 (H, W): 8×8 池化 (和 image 版一致)
  - 时间维度 (T): 时间卷积 + 时间注意力
  - 输出: (T, 64, 64) latent 序列
  │
  ▼
VQGAN Decoder × T 帧
  │
  ▼
输出: T 帧视频 (24 fps, 480p)
```

**关键创新点**：

```
时间注意力 (Temporal Attention):
  - 跨时间步的全局 self-attention
  - 空间 + 时间 混合注意力 (cross-domain attention)

时间卷积 (Temporal Conv):
  - 3D kernel (1, 3, 3): 时间维只有 1
  - 用于捕捉局部运动
```

**训练数据**：

```
来源: 公开视频数据集 (Kinetics-700, Kinetics-400, WebVid-10M)
预处理: 抽取 16 帧 (480p), 压缩到 256×256
规模: 500 万 video clip
授权: 研究用途 (注意: 用于训练有法律灰区)
预计训练时间: 3-6 个月 (64×A100)
```

---

### 3.4 Text-to-3D（第 4 步，Triplane Diffusion）

**参考 newplan.md**：完整管线已设计。核心架构：

```
text → Text Encoder (Transformer encoder-only) → 条件向量
                                         │
                                         ▼
                             Triplane Diffusion U-Net (2D)
                                         │
                                         ▼
                             SDF Query MLP → 3D 几何
                                         │
                                         ▼
                             3DGS Renderer → 实时渲染
```

**新增自研组件**（在 newplan.md 基础上）：

```
1. Text Encoder 扩展到支持更长的输入 (maxSeqLen=128)
2. Triplane Diffusion 升级到 64×64 分辨率 (2x, 参数量 4x)
3. Score Distillation 自研自编码器 (不用外部 CLIP)
```

---

### 3.5 Vision-Language Model（第 5 步，多模态融合）

**架构**：

```
┌──────────────┐         ┌──────────────┐
│ 图像输入      │──→ VLM-Encoder ──→ 16 个视觉 token
│ (224×224)    │  (ViT-B/16, 197M)   (每个 token 768 维)
└──────────────┘         └──────────────┘
                                         │
                                         ▼ (linear projection 到 LLM 的 dModel)
                                  ┌──────────────────┐
                                  │ LLM Backbone     │
                                  │ (Text-to-Text 模型)│
                                  │ 输入: [image_tokens + text_tokens] │
                                  │ 输出: 文本回答    │
                                  └──────────────────┘
```

**训练数据**：

```
格式: {"image": "...", "messages": [...]}
来源: COCO / LAION-CC / Visual Genome (CC 授权)
规模: 100 万 (image, text) 对
```

---

## 第 4 章：非 Transformer 架构自研

### 4.1 Mamba（选择性 SSM）

**核心实现**：

```js
// selective scan (simplified)
class SelectiveScan {
  constructor(dState = 16) {
    this.dState = dState;
  }

  // 对序列 x, 选择性参数 Δ, B, C, A
  forward(x, delta, B, C, A) {
    // h_t = A · h_{t-1} + Δ · B · x_t   (h 是 dState 维隐藏状态)
    // y_t = C · h_t
    // A, B, C, Δ 都是数据相关的（选择性）
    // 关键: 这个递归是数据依赖的, 无法并行化
    // 但可以用 parallel scan (log n 步) 加速
  }
}

class MambaBlock {
  constructor(dModel, dState = 16, dConv = 4) {
    // 步骤:
    // 1. 1D 卷积 (kernel=dConv) → 局部混合
    // 2. 投影 x → Δ, B, C, A (所有数据相关参数)
    // 3. Selective Scan (SSM 核心)
    // 4. 输出投影
  }
}
```

**Mamba Block 与 Transformer Block 参数对比**：

| 维度 | Transformer | Mamba |
|---|---|---|
| 计算 | O(n²) | O(n) |
| 参数 | 4×dModel² (QKV + Wo) + FFN | 4×dModel·dState (S4 参数) + dModel·dState (output) |
| 序列 1M tokens | 不可行 | 可行 |
| 序列 2K tokens | 可行 | 优于 Transformer |

**自研难度**：高。选择性扫描的 parallel scan 实现需要 log n 步递归，在 tfjs 中手写。

---

### 4.2 Jamba 风格混合架构

```
Layer 0:  Mamba Block        (处理全局依赖)
Layer 1:  Transformer Block   (处理局部语义)
Layer 2:  Mamba Block
Layer 3:  Transformer Block
...
Layer 31: Transformer Block

共 32 层, 16 层 Mamba + 16 层 Transformer
总参数: ~3B (Mamba 参数更少)
```

**优势**：长序列上 Mamba 负责全局, Transformer 负责精细局部；短序列上两者配合效果互补。

---

### 4.3 RWKV 5

```js
class RWKVBlock {
  // 核心:
  // a_t = A + 0.06 / (t+1)^0.3       # 时间衰减
  // k_t = Wk(x_t)                     # 键
  // v_t = Wv(x_t)                     # 值
  // KV_t = a_t * KV_{t-1} + (1-a_t) * k_t * v_t
  // y_t = sum(KV_t · query)

  forward(x, KV_prev, t) {
    const a = this.computeA(t);
    const k = this.Wk(x);
    const v = this.Wv(x);
    const KV = a * KV_prev + (1 - a) * k * v;
    const y = this.attention(KV, x);
    return { output: y, KV: KV, t: t + 1 };
  }
}
```

**自研难度**：中。公式直接可翻译为矩阵运算。

---

## 第 5 章：训练基础设施

### 5.1 训练框架自研

**不基于 PyTorch / TF，用 tfjs**（保持 mini-engine 一致）：

```
核心模块:
  optimizer.js         — AdamW (beta1=0.9, beta2=0.95)
  gradient.js          — 梯度计算 (tf.gradientTape 或手动反向)
  checkpoint.js        — 权重保存/恢复
  distributed.js       — 多设备通信 (tfjs 没有 NCCL, 用自定义实现)
  scheduler.js         — 学习率调度 (cosine decay + warmup)
  dataloader.js        — 数据管线 (batching, shuffling, prefetching)
```

**多设备通信**（不依赖 NCCL）：

```js
// 简单实现: 使用 tfjs 的 shared worker 通信
// 每个 GPU 对应一个 worker, 用 MessageChannel 传递梯度

class AllReduce {
  // 梯度平均: 所有设备的梯度相加后除以 numDevices
  reduce(sum) {
    for (const worker of this.workers) {
      sum = worker.postMessage(sum);  // 伪代码
    }
    return sum / this.numWorkers;
  }
}
```

---

### 5.2 分布式训练策略

| 策略 | 实现 | 通信频率 |
|---|---|---|
| **Data Parallel** | 每个 GPU 跑完整模型, 梯度平均 | 每步 |
| **Tensor Parallel** | 矩阵按列切分, 每 GPU 跑一半 | 每步 |
| **Pipeline Parallel** | 模型按层切分, 微批次流水 | 每微批次 |
| **Expert Parallel** | MoE 场景, 专家按 GPU 切分 | 每步 |

**ZeRO-3 简化版（单卡 100B）**：

```
1. 不存完整参数, 只存激活值 + 中间梯度
2. 前向: 从 checkpoint 恢复需要的参数, 用完立即释放
3. 反向: 同样方式, 梯度算完立即做 AllReduce
4. 优化器状态: 存在 CPU 内存 (fp32, 8B 字节 = 800GB, 单机足够)
5. 显存需求从 200GB 降到 48GB (A100 80GB 可跑)
```

---

### 5.3 训练数据管线自研

```js
class DataPipe {
  constructor() {
    this.sources = [];   // 数据源列表
    this.filters = [];   // 清洗步骤
    this.tokenizer = null;
    this.sampler = null; // 混合采样
  }

  // 数据加载: 流式读取 JSONL 文件
  async stream(path) {
    const lines = fs.createReadStream(path)
      .pipe(new Transform({
        transform(chunk, _, cb) {
          const line = chunk.toString().trim();
          if (line) cb(null, JSON.parse(line));
          else cb();
        }
      }));
  }

  // 语言检测
  async detectLanguage(text) {
    // fastText 风格: 字符 n-gram 特征 + 线性分类
    // 自研, 不依赖外部库
  }

  // 去重
  async deduplicate(texts) {
    // SimHash: 对每个文本计算 64-bit 哈希, 汉明距离 < 3 的判为重复
  }

  // 分词
  tokenize(text) {
    return this.tokenizer.encode(text);
  }
}
```

---

## 第 6 章：验证与评估体系

### 6.1 每个阶段的测试

| 阶段 | 测试内容 | 通过标准 |
|---|---|---|
| Tokenizer | encode/decode 可逆, OOV 处理 | 100% 可逆 |
| Embedding | forward 输出维度正确 | 形状匹配 |
| RoPE | 已知位置编码的手算验证 | 误差 < 1e-6 |
| Attention | causal mask 正确, QKᵀ 维度正确 | 测试 mask 每个位置 |
| FFN | SwiGLU 输出非负 (sigmoid 保证) | 无 NaN |
| Block | residual + norm 后形状不变 | 形状匹配 |
| Model | 给定 token 输入, 输出 logits 维度正确 | 维度匹配 |
| 训练 | loss 单调下降 (至少前 1000 steps) | loss 不爆炸 |

### 6.2 评估基准

| 基准 | 类型 | 难度 | 说明 |
|---|---|---|---|
| **MMLU** | 多任务理解 | 中 | 57 个学科, 5-10 题/学科 |
| **HellaSwag** | 常识推理 | 中 | 10 个类别的完形填空 |
| **ARC** | 科学推理 | 高 | 中小学科学问题 |
| **GSM8K** | 数学 | 高 | 小学数学应用题 |
| **HumanEval** | 代码生成 | 高 | Python 函数生成 |
| **Winogrande** | 指代消解 | 高 | 长段落代词指代 |
| **TruthfulQA** | 事实性 | 极高 | 区分事实和错误陈述 |

**目标分数**（3B 模型）：

| 基准 | 目标 | 说明 |
|---|---|---|
| MMLU | >35% | 接近随机水平 (20 类四选一=25%) |
| HellaSwag | >40% | 随机基线 ~30% |
| GSM8K | >10% | 数学能力很弱但非零 |
| HumanEval | >5% | 代码生成很弱 |

---

## 第 7 章：分阶段实施路线图

### Phase 0: 基础设施（2 周）

| 文件 | 内容 |
|---|---|
| `src/tokenizer/bpe.js` | BPE 分词器 |
| `src/tokenizer/vocab.js` | 词表管理 |
| `src/tokenizer/test/` | encode/decode 测试 |
| `src/nn/optimizer.js` | AdamW 优化器 |
| `src/nn/rmsnorm.js` | RMSNorm |
| `src/nn/swiglu.js` | SwiGLU FFN |
| `src/nn/rope.js` | RoPE + NTK 插值 |
| `src/nn/flashattention.js` | FlashAttention 思路实现 |
| `src/training/checkpoint.js` | 训练 checkpoint |
| `src/training/scheduler.js` | 学习率调度 |

---

### Phase 1: 3B 基线模型（4 周）

| 文件 | 内容 |
|---|---|
| `src/models/text/transformer3b.js` | 完整 3B 模型 |
| `src/models/text/transformer3b.test.js` | 端到端测试 |
| `src/data/pretrain.js` | Pre-training 数据管线 |
| `src/train/train_pretrain.js` | 预训练脚本 |
| `src/train/train_sft.js` | 指令微调脚本 |
| `src/eval/mmlu.js` | MMLU 评估脚本 |
| `public/models/3b/` | 训练输出目录 |

---

### Phase 2: 20B 模型 + 训练优化（6 周）

| 文件 | 内容 |
|---|---|
| `src/models/text/transformer20b.js` | 20B 配置 |
| `src/training/zero3.js` | ZeRO-3 显存优化 |
| `src/training/distributed.js` | 多设备通信 |
| `src/train/train_pretrain_20b.js` | 20B 预训练 |

---

### Phase 3: Mamba + Jamba 混合架构（3 周）

| 文件 | 内容 |
|---|---|
| `src/models/mamba/selective_scan.js` | 选择性扫描 |
| `src/models/mamba/mamba_block.js` | Mamba Block |
| `src/models/hybrid/jamba.js` | Jamba 混合架构 |

---

### Phase 4: Text-to-Image Diffusion（6 周）

| 文件 | 内容 |
|---|---|
| `src/models/image/unet3d.js` | 3D U-Net |
| `src/models/image/vqgan.js` | VQGAN 解码器 |
| `src/models/image/train_diffusion.js` | 扩散训练 |

---

### Phase 5: Text-to-3D（6 周，newplan.md 已有详细设计）

| 文件 | 内容 |
|---|---|
| `src/gen/3d/textEncoder.js` | 文本编码器 |
| `src/gen/3d/triplaneDiffusion.js` | Triplane U-Net |
| `src/gen/3d/sdfQuery.js` | SDF 查询 |
| `src/gen/3d/gsRenderer.js` | 3DGS 渲染 |

---

### Phase 6: VLM + 长上下文 + 最终集成（8 周）

| 文件 | 内容 |
|---|---|
| `src/models/vlm/` | 多模态融合 |
| `src/models/text/transformer100b.js` | 100B MoE 配置 |
| `src/eval/` | 全部基准评估 |
| `src/inference/` | 推理优化 (KV cache, 量化) |

---

## 第 8 章：项目目录结构

```
LLMInnovation/
  │
  ├── README.md                — 项目介绍
  ├── plan.md                  — 本文件 (v0.1)
  ├── newplan.md               — Text-to-3D 详细计划 (mini-engine 已有)
  │
  ├── src/
  │   ├── tokenizer/           — BPE 分词器
  │   │   ├── bpe.js
  │   │   ├── vocab.js
  │   │   └── test/
  │   │
  │   ├── nn/                  — 神经网络基础模块
  │   │   ├── rmsnorm.js
  │   │   ├── swiglu.js
  │   │   ├── rope.js
  │   │   ├── attention.js         — Multi-Head + GQA + FlashAttention 思路
  │   │   ├── positionalEncoding.js
  │   │   ├── optimizer.js         — AdamW
  │   │   └── test/
  │   │
  │   ├── models/
  │   │   ├── text/                  — 文本模型
  │   │   │   ├── transformer3b.js
  │   │   │   ├── transformer20b.js
  │   │   │   ├── transformer100b.js (MoE)
  │   │   │   └── test/
  │   │   │
  │   │   ├── mamba/                 — Mamba SSM
  │   │   │   ├── selective_scan.js
  │   │   │   ├── mamba_block.js
  │   │   │   └── test/
  │   │   │
  │   │   ├── hybrid/                — 混合架构
  │   │   │   ├── jamba.js
  │   │   │   └── test/
  │   │   │
  │   │   ├── image/                 — 图像生成
  │   │   │   ├── unet3d.js
  │   │   │   ├── vqgan.js
  │   │   │   └── test/
  │   │   │
  │   │   ├── 3d/                    — 3D 生成 (来自 newplan.md)
  │   │   │   ├── textEncoder.js
  │   │   │   ├── triplaneDiffusion.js
  │   │   │   ├── sdfQuery.js
  │   │   │   ├── gsRenderer.js
  │   │   │   └── test/
  │   │   │
  │   │   └── vlm/                   — 多模态融合
  │   │       ├── vlm.js
  │   │       └── test/
  │   │
  │   ├── training/              — 训练基础设施
  │   │   ├── checkpoint.js
  │   │   ├── scheduler.js
  │   │   ├── distributed.js       — 多设备通信
  │   │   ├── zero3.js             — ZeRO-3 显存优化
  │   │   ├── dataloader.js
  │   │   └── test/
  │   │
  │   ├── data/                  — 数据管线
  │   │   ├── pretrain.js          — Pre-training 数据
  │   │   ├── sft.js               — SFT 数据
  │   │   ├── preference.js        — Preference 数据
  │   │   ├── languageDetect.js    — 语言检测
  │   │   ├── dedup.js             — 去重
  │   │   └── test/
  │   │
  │   ├── inference/             — 推理优化
  │   │   ├── kvcache.js
  │   │   ├── quantize.js          — INT8/INT4 量化
  │   │   ├── speculative.js       — Speculative decoding
  │   │   └── test/
  │   │
  │   └── eval/                  — 评估基准
  │       ├── mmlu.js
  │       ├── hellaswag.js
  │       ├── gsm8k.js
  │       ├── humaneval.js
  │       └── test/
  │
  ├── scripts/
  │   ├── train_pretrain_3b.sh
  │   ├── train_pretrain_20b.sh
  │   ├── train_pretrain_100b.sh
  │   ├── train_sft.sh
  │   ├── eval_mmlu.sh
  │   └── benchmark_inference.sh
  │
  ├── test/                      — 端到端测试
  │   ├── tokenizer.test.js
  │   ├── transformer.test.js
  │   ├── mamba.test.js
  │   ├── jamba.test.js
  │   ├── diffusion.test.js
  │   └── integration.test.js
  │
  └── public/models/             — 训练输出
      ├── 3b/
      ├── 20b/
      ├── 100b/
      ├── image/
      └── 3d/
```

---

## 第 9 章：总结

### 核心设计决策

```
1. 全部自研，不依赖外部预训练权重
   — 从 Tokenizer 到训练管线, 全部自写

2. 训练数据自处理
   — 不下载别人的训练集, 而是构建可复现的数据管线
   — Common Crawl / arXiv / Wikipedia (全部公开/CC 授权)

3. 架构多样性
   — Transformer (基线)
   — Mamba / SSM (线性时间)
   — Jamba (混合)
   — RWKV (RNN 等价)

4. 渐进式规模
   — 3B (CPU 可跑, 自研基线)
   — 20B (单卡推理, 8 卡训练)
   — 100B (集群, 工业级)

5. 多模态扩展
   — Text-to-Text (Phase 1)
   — Text-to-Image (Phase 4)
   — Text-to-3D (Phase 5, 与 mini-engine 集成)
   — VLM (Phase 6)
   — 未来: Text-to-Video / Text-to-Audio

6. 不依赖 PyTorch / TF, 用 tfjs
   — 保持与 mini-engine 一致
   — 浏览器端推理友好
```

### 时间线总览

```
Phase 0: 基础设施          2 周
Phase 1: 3B 基线           4 周
Phase 2: 20B + 优化        6 周
Phase 3: Mamba/Jamba       3 周
Phase 4: Text-to-Image     6 周
Phase 5: Text-to-3D        6 周
Phase 6: VLM + 最终集成     8 周
─────────────────────────────────
总计:                    ~35 周 (约 8 个月)

Phase 1 完成时, mini-engine 就能获得一个 3B 级别自研 LLM 作为 engine.gen 的后端。
Phase 5 完成时, mini-engine 就能获得一个 Text-to-3D 生成器作为 engine.shapeGen 的后端。
```
