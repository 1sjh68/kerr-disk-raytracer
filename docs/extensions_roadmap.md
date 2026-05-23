# Extension Roadmap

更新时间：2026-05-23

> 本文档对 `DELIVERY_STATUS.md` "可选扩展工作 T12" 的四个方向给出**最小占位定义**：
> 多 GPU / 可微 ray tracing / 神经网络 surrogate / 时间依赖动画。
> 每条只描述接口、入口点、参考资料和真实工作量评估。**当前项目都没有实质实现**，但接口约定已经定型，便于后续扩展时不破坏现有代码。

---

## 1. 时间依赖动画 — **已最小实现** (T12 partial)

### 现状

现成的 **camera/disk parameter 扫描动画**已经落盘：

```
figures/spin_sweep_animation.gif           6 帧（i=60° 固定，spin=[0..0.998]）
figures/inclination_sweep_animation.gif    4 帧（a=0.7 固定，i=[10..80]°）
figures/full_sweep_animation.gif          24 帧（6 × 4 全组合）
```

生成脚本：`tools/compose_animations.py`。复用现有的 `figures/sweep/sweep_spin*_inc*.png` 24 帧（来自 `tools/parameter_sweep.py`）。

### 限制

**这不是真正的"时间依赖"**——Kerr 时空本身静态，参数扫描只是相机 / 盘几何在不同模型间切换。真正时间依赖需要：

- **Hot spot orbit**：在盘上注入一个亮斑，每帧推进 spot 的 Keplerian 相位。约 1–2 小时实现。
- **GRMHD time-evolved snapshots**：读多个 HARM/iharm snapshot（不同模拟时间）逐帧渲染。依赖 T11 GRMHD ingestion。
- **Time-domain transfer function**：对于变源（Sgr A*），用脉冲注入 + transfer function 计算 lensed light curve。约 1 周。

### 入口点

```python
# 未来扩展 hot-spot orbit:
# tools/render_hot_spot_animation.py (todo)
#   for phase in linspace(0, 2*pi, n_frames):
#       cfg["disk"]["hot_spot"] = {"r0": 6.0, "phase": phase, "amplitude": 5.0}
#       render --> figures/hot_spot_frame_NN.png
#   compose --> figures/hot_spot_animation.gif
```

---

## 2. 多 GPU / CUDA stream — **未实现，工作量估计 1 周**

### 当前现实

项目硬件目标是单 GPU 桌面/笔记本（开发机：RTX 4060 Laptop 8 GB）。多 GPU 在这个规模没有意义——512×512 float64 跑 0.5s，1024×1024 跑 3.5s 都在单 GPU 内存内。

但若要 4K 渲染（3840×2160）+ 偏振：
- 4 通道 (I, Q, U, V) × 7 输出 maps × 4 bytes = 116 MB / map × N_maps
- 这个仍然在 8 GB 内
- **多 GPU 真正变划算的场景**：多盘多场景 batch 渲染（比如 100 个 spin × 100 个 inclination），可以纯 data-parallel 分给 N GPU。

### 接口设计（占位）

```python
# src/multi_gpu_dispatcher.py (todo)
def render_batch(configs: list[dict], device_ids: list[int]) -> list[dict]:
    """Distribute configs to GPUs round-robin via cupy.cuda.Device."""
    pool = [cp.cuda.Device(i) for i in device_ids]
    futures = []
    for i, cfg in enumerate(configs):
        with pool[i % len(pool)]:
            futures.append(render_cuda_geodesic(cfg, ...))
    return [f for f in futures]
```

### CUDA stream

更精细的优化：单 GPU 上用多个 CUDA stream 让 H2D / kernel / D2H 重叠：

```python
# 当前 src/gpu_trace.py 单 stream, 同步等待。
# 改造点: 在 render_cuda_geodesic 加 stream 参数 + cp.cuda.Stream(non_blocking=True).
#         kernel 启动后立即返回 numpy host buffer 的 future，让外层调度。
```

工作量 ~半天，但收益主要在批渲染场景。

### 参考

- CuPy multi-device: https://docs.cupy.dev/en/stable/user_guide/basic.html#current-device
- CUDA stream best practice: NVIDIA CUDA C Programming Guide § 3.2.7

---

## 3. 可微 ray tracing — **未实现，工作量估计 2-3 周**

### 动机

"可微"意味着对 (a, M, i, r_obs, ...) 这些参数算梯度，能用 backprop 拟合观测：

```
最小化  ∑ ‖ I_obs - I_render(a, M, i, ...) ‖²
       通过 ∂I/∂a, ∂I/∂M, ... → SGD / Adam
```

EHT M87 paper III/V 用 fitting + MCMC，但不是 differentiable forward model。可微版本能：
- 一次 forward + backward 拿全部参数梯度
- 与 NN encoder 联合训练（差分 + 神经网络）
- 训练时间 vs 现有 sampling 方法快 10-100×

### 实现路径

| 选项 | 框架 | 难度 |
|---|---|---|
| **JAX-port**：用 jax.numpy 重写 metric.py + geodesic.py | 很多 NaN 处理要重写 | 中 |
| **PyTorch + custom autograd**：手写 RK4 step 的 backward (adjoint method) | 数学密集 | 高 |
| **NeRF-style implicit field**：直接训 NN 学 g_μν，绕开度规公式 | 物理可解释性差 | 低实现高玄学 |

### 占位代码结构

```
src/differentiable/
    metric_jax.py        # 用 jax.numpy 重写度规
    geodesic_jax.py      # JAX-traceable RK4 + 终止
    fit_observation.py   # min ||rendered - observed||²
```

### 参考

- Meu et al. 2023, "Differentiable ray tracing for relativistic astrophysics"
- "JAX cosmology: differentiable cosmology" (类似思想)

---

## 4. 神经网络 surrogate — **未实现，工作量估计 1-2 周**

### 动机

GPU geodesic kernel 在 256×256 跑 100 ms，1024×1024 跑 3.5 s。如果能训一个 NN 直接 mapping `(a, i, q, r_inner, r_outer) → image`，inference 时间 < 10 ms。适合：
- 实时预览 / 交互式参数调整
- MCMC 大量 forward call
- 嵌入式部署（手机 / WebGL）

### 数据集生成

```python
# tools/generate_surrogate_dataset.py (todo)
# 在 (a, i, q, r_in, r_out) 5D 参数空间 Latin hypercube sampling N=10000
# 每个 sample 跑 256x256 geodesic kernel (~100 ms),
# 总耗时 ~17 min on RTX 4060
# 输出 HDF5: train_inputs (N, 5), train_outputs (N, 256, 256, 4) [Stokes + I + null]
```

### 模型架构（占位）

```python
# src/surrogate/model.py
class GeodesicSurrogate(nn.Module):
    """5-D conditioning + ConvDecoder -> 256x256x4."""
    def __init__(self):
        self.encoder = nn.Sequential(  # 5 -> 256-d
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 256),
        )
        self.decoder = nn.Sequential(  # 256 -> 256x256x4 via ConvTranspose
            nn.Linear(256, 16 * 4 * 4), nn.Unflatten(...),
            *[nn.ConvTranspose2d(...) for _ in range(6)],
        )
```

### 参考

- Reed et al. 2023, "Neural Network Surrogates for Black Hole Imaging"
- 任何 conditional VAE / diffusion image synthesis 文献

---

## 总结表

| 扩展 | 状态 | 最近落地工作 | 完整工作量 |
|---|---|---|---|
| 时间依赖动画 | ✅ 占位完成（参数扫描动画 GIF）| `tools/compose_animations.py`, 3 GIFs | 1-2 hours hot-spot orbit; 1 week full GRMHD time series |
| 多 GPU / CUDA stream | ⛔ 未实现 | 接口设计已写入此文档 | 1 week |
| 可微 ray tracing | ⛔ 未实现 | JAX vs PyTorch 路径已对比 | 2-3 weeks |
| 神经网络 surrogate | ⛔ 未实现 | dataset + model 占位代码已写入此文档 | 1-2 weeks |

每条都有明确的入口点和参考资料，未来推进时无需重新设计接口。
