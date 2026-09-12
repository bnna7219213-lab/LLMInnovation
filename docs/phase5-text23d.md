# Phase 5 — Text-to-3D（Triplane + Score Distillation）

从文本生成 3D 模型。用 Triplane 表示三维场景，通过 Score Distillation Sampling（SDS）用 2D 扩散模型优化。

测试：`node --test test/text23d.test.js`

## Triplane（`src/models/3d/triplane.js`）

用三个正交平面（XY/XZ/YZ）表示三维场景，任意分辨率查询 3D 点。

```js
const { Triplane } = require('./src/models/3d/triplane.js');

const tp = new Triplane({
  planeSize: 64, latentChannels: 8,
  mlpHidden: 64, numMLPLayers: 2,
});

// 查询特征（拼接三个平面的双线性采样）
const feat = tp.query(x, y, z);      // [latentChannels × 3]，坐标 ∈ [-1, 1]

// 前向：特征 → [SDF, R, G, B]
const out = tp.forward(feat);

// 直接查询 SDF + 颜色
const [sdf, r, g, b] = tp.querySDF(x, y, z);

// 体积渲染：沿光线采样
const { color, depth } = tp.volumeRender(rayOrigin, rayDir, numSamples = 32);

// 相机渲染 [imgH × imgW × 3]
const img = tp.renderCamera(imgH, imgW, fov, distance = 3, numSamples = 16);

// 2D 等值面切片（简化 marching cubes）
const surface = tp.getIsoSurface2D(ySlice = 0, resolution = 16);

tp.countParams();
```

**参数量估算**（默认）：3 × 64 × 64 × 8 = 76800，比 SDF 网格（128³=2M）和 NeRF MLP（百万级）更省。

**体积渲染**：沿光线等距采样 → SDF 经 sigmoid 转密度 `σ = σ_sigmoid(K·(-sdf))` → 累积透明度 alpha compositing。

**API**：`query(x,y,z)`、`forward(feat)`、`querySDF(x,y,z)`、`volumeRender(rayOrigin, rayDir, numSamples)`、`renderCamera(imgH, imgW, fov, distance, numSamples)`、`getIsoSurface2D(ySlice, resolution)`、`countParams()`

## SDS（`src/models/3d/sds.js`）

Score Distillation Sampling：用 2D 扩散模型为 3D 模型提供梯度信号。

```js
const { SDS } = require('./src/models/3d/sds.js');

const sds = new SDS({ tMin: 50, tMax: 1000, imgSize: 32, numSamples: 16, fov: Math.PI/4, distance: 3 });

// 单步前向：渲染 + 加噪
const { zt, epsilon, x0, sqrtAlphaBar, sqrtOneMinusAlphaBar } =
  sds.step(triplane, t, textEmb, alpha, alphaBar, viewAngles, randNoise);

// 完整训练步：随机视角 + 时间步 + UNet 预测 + loss
const { loss, view, t, epsilonPred, epsilon } =
  sds.trainStep(triplane, unet, textEmb, alpha, alphaBar);

// 有限差分梯度估计
const grad = sds.finiteDiffGradient(triplane, planeIdx, pixelIdx, channel, eps, unet, textEmb, alpha, alphaBar);
```

**SDS 流程**：
1. 渲染 2D 图像 `x_0 = render(θ, view)`
2. 加噪 `z_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`
3. UNet 预测 `ε_pred`
4. 损失 `L = 0.5·w(t)·||ε_pred - ε||²`，其中 `w(t) = (1-ᾱ_t)·√ᾱ_t/(1-α_t)`

**注意**：`finiteDiffGradient` 会修改并恢复 triplane 参数，内部调用 `trainStep`（涉及随机视角，梯度有噪声，但可累积）。

## Text23D（`src/models/3d/text23d.js`）

完整生成管道：

```js
const { Text23D } = require('./src/models/3d/text23d.js');

const t23d = new Text23D({
  unetConfig: { latentChannels: 3, latentSize: 32, textEmbedDim: 64, textTokenLen: 16,
                blockChannels: [32, 64], numHeads: 2 },
  triplaneConfig: { planeSize: 16, latentChannels: 8, mlpHidden: 16, numMLPLayers: 1 },
  sdsConfig: { tMin: 50, tMax: 500, imgSize: 32, numSamples: 8, fov: Math.PI/4, distance: 3 },
});

// 生成 3D 模型
const { loss, triplane } = t23d.generate('a sphere', numSteps = 50, lr = 0.01, seed = 1,
  (step, loss) => console.log(step, loss));

// 渲染
const img = t23d.render(triplane, imgSize = 32, numSamples = 16);
```

**API**：`encodeText(prompt, tokenLen, embedDim)`、`generate(prompt, numSteps, lr, seed, progressCallback)`、`render(triplane, imgSize, numSamples)`

**优化循环**：每步 `trainStep` 计算 loss + 采样 4 个像素做有限差分梯度更新。

## 关键实现点

- **依赖路径**：`text23d.js` 里 require diffusion 模块用 `../diffusion/...`（不是 `./diffusion/...`）
- **调度 API**：diffusion 模块导出的是函数 `generateSchedule`，不是 `DiffusionSchedule` 类

## 测试要点

- Triplane 查询维度、forward RGB 在 [0,1]、体积渲染无 NaN
- SDS 权重函数不同时间步不同
- 完整 generate 管道 loss 不为 NaN
