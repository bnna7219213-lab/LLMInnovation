# Phase 4 — Text-to-Image Diffusion

DDPM/DDIM 扩散模型：从文本生成图像。核心是 UNet 去噪网络 + VAE 潜空间压缩。

测试：`node --test test/diffusion.test.js`

## Diffusion Schedule（`src/models/diffusion/diffusion_schedule.js`）

```js
const { generateSchedule, addNoise, reverseStep, ddimStep } =
  require('./src/models/diffusion/diffusion_schedule.js');

// 生成 β 调度（返回 { betas, alpha, alphaBar, sigma }）
const sched = generateSchedule('cosine', 1000, { s: 0.008 }); // 或 'linear'

// 前向加噪：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
const xt = addNoise(x0, tArray, noise, sched.alphaBar); // 返回 Float32Array[]

// DDPM 反向采样
const xTm1 = reverseStep(x_t, t, epsilonPred, sched.alpha, sched.alphaBar, sched.sigma, z, nPixels);

// DDIM 步
const ddim = ddimStep(...);
```

**公式**：
- Forward: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`
- Cosine schedule: `ᾱ_t = cos(π/2 · (t+s)/(T+s))²`
- Linear schedule: `β_t = β_min + t/T · (β_max - β_min)`

## VAE（`src/models/diffusion/vae.js`）

把图像压缩到潜空间，UNet 在潜空间中操作（Stable Diffusion 思路）。

```js
const { VAE } = require('./src/models/diffusion/vae.js');

const vae = new VAE({
  inputChannels: 3, latentChannels: 4, inputSize: 64, latentSize: 8,
  blockChannels: [32, 64, 128],
});

// 编码：图像 → (μ, σ)
const { mu, sigma } = vae.encode(image, batch);

// 重参数化：z = μ + σ·ε
const z = vae.reparameterize(mu, sigma, eps);

// 解码：潜变量 → 图像（返回 Float32Array[batch × H×W×C]）
const decoded = vae.decode(z, batch);   // 注意：z 是扁平 Float32Array，不是数组

vae.getLatentShape(); // [latentSize, latentSize, latentChannels]
```

**典型 SD VAE**：256×256 → 32×32×4，从 196608 压到 4096（48x）。

**注意**：`decode(z, batch)` 的 `z` 是扁平 Float32Array（`[batch × latentSize² × latentChannels]`），**不要**包成 `[z]` 数组。

## UNet（`src/models/diffusion/unet.js`）

去噪核心，含文本条件 cross-attention。

```js
const { UNet } = require('./src/models/diffusion/unet.js');

const unet = new UNet({
  latentChannels: 4, latentSize: 8,
  textEmbedDim: 64, textTokenLen: 16,
  blockChannels: [32, 64, 128], numHeads: 4,
});

// forward(x, t, textEmb, batch)：输入都是数组（batch 维度）
// x: Float32Array[]，t: number[]，textEmb: Float32Array[] → 返回 Float32Array[]
const epsPred = unet.forward([z], [500], [textEmb], 1)[0];
```

**结构**：Down Blocks（逐级下采样 + skip）→ Middle（attention + text conditioning）→ Up Blocks（逐级上采样 + 合并 skip）。

**数值稳定性**：cross-attention 的 softmax 对 scores 裁剪到 [-50, 50]，防止 `Math.exp` 溢出产生 NaN。

## Text2Image（`src/models/diffusion/text2image.js`）

完整生成管道：

```js
const { Text2Image } = require('./src/models/diffusion/text2image.js');

const t2i = new Text2Image({
  T: 1000, schedule: 'cosine',
  textEmbedDim: 64, textTokenLen: 16, numHeads: 4,
  vaeConfig: { inputChannels: 3, latentChannels: 4, inputSize: 64, latentSize: 8, blockChannels: [32,64,128] },
});

// 生成图像（DDPM）
const image = t2i.generate('a cat', seed = 42, numSteps = 1000);  // [H×W×3]

// DDIM 快速采样（确定性，50 步）
const image2 = t2i.generateDDIM('a cat', 42, 50);

// 训练一步（去噪 MSE loss）
const loss = t2i.trainStep(images, 'a cat');
```

**流程**：文本编码 → 噪声初始化 → 反向扩散（UNet 逐步去噪）→ VAE 解码 → 图像。

**API**：`encodeText(prompt, tokenLen, embedDim)`、`generate(prompt, seed, numSteps)`、`generateDDIM(prompt, seed, numSteps)`、`trainStep(images, prompt)`

## 关键 bug 记录（已修复）

1. **VAE W_mu 尺寸错误**：原为 `[encLatentCh × latentChannels]`，应为 `[encSize²×encLatentCh × latentSize²×latentChannels]`
2. **softmax 溢出**：attention scores 未裁剪导致 `exp` 溢出 → NaN
3. **decode 调用**：误传 `[z]` JS 数组，应传扁平 `z`

## 测试要点

- VAE 编码-解码维度正确、无 NaN
- UNet 前向维度正确
- 不同 seed 产生不同输出（seed 需足够远，如 1 vs 1000000）
