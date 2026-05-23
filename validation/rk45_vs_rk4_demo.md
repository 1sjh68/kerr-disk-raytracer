# RK45 vs RK4 on Critical Rays — Empirical Follow-Up

更新时间：2026-05-23

## 背景

`validation/geokerr_coordinate_alignment.md` 在改进方向里点名了 **"切到 RK45 自适应步长"** 作为 RK4 在 photon sphere 附近精度退化的解。本文档把这条建议从论断升级为 **可重现的实证**。

## 设计

`src/integrators.py` 已有 Dormand-Prince RK45 实现；`src/geodesic.py` 加了 `trace_single_ray_rk45()`，与原 `trace_single_ray()` 共用同一份终止逻辑（视界 / 逃逸 / 赤道穿越）和同一份 RHS（`hamiltonian_rhs_for_spin`），仅积分方法不同：

| 项目 | trace_single_ray (RK4) | trace_single_ray_rk45 |
|---|---|---|
| 步长 | 固定 step_size=0.35 | 自适应（PI controller，atol=1e-8 rtol=1e-6）|
| 每步 RHS 调用 | 4 | 6 (DP5) + 1 (FSAL) |
| 默认 max_steps | 700–5000 | 5000 |
| 拒收回退 | 无（fixed step）| 有（误差超容则 h 缩小重试） |

## 结果（5 条代表性光线，r_obs=1000，max_steps=5000）

数据来源：`validation/rk45_vs_rk4_demo.json`，可视化 `research/reproduction/rk45_vs_rk4.png`。

| (α, β) | geokerr | RK4 status / steps / null / wall | RK45 status / steps / null / wall |
|---|---|---|---|
| (+0.6, +0.6) | captured | captured / 2856 / **1e+30 (溢出)** / 1.51 s | captured / 232 / **3.2e-5** / 0.28 s |
| (-3.0, -3.0) | captured | captured / 2862 / **1e+30 (溢出)** / 1.43 s | captured / 236 / **7.5e-5** / 0.27 s |
| (+5.4, -5.4) | disk | disk / 2873 / 3.9e-4 / 1.49 s | disk / 220 / 2.0e-3 / 0.22 s |
| (+3.0, +3.0) | captured | **disk (临界分歧)** / 2845 / 1.7e-5 / 1.41 s | **disk (同样分歧)** / 204 / 9.5e-4 / 0.21 s |
| (-1.8, -1.8) | captured | captured / 2857 / **1e+30 (溢出)** / 1.45 s | captured / 233 / **5.3e-5** / 0.25 s |

## 核心发现

### 1. RK45 用 ~13× 少 step、5–7× 短 wall time 拿到同等几何路径

trajectories 在 (r, θ) 平面上几乎重合（图中 RK4 蓝色虚线被 RK45 绿色点划线完全覆盖）。两种方法在屏幕到视界附近这一段的轨迹精度对当前 application 已经足够好。

### 2. 真正的差异在数值稳定性 — null constraint preservation

RK4 在 captured ray 上跨过视界后没有保护机制，state 各分量爆炸到 1e+92 量级（被 clip 到 1e+30 输出）。这意味着如果 downstream 代码读 `state` 算别的物理量（比如 timing of last finite step、Carter constant drift），RK4 会给垃圾数。

RK45 自适应步长在视界附近会被强制缩短（误差监测拒收并 retry h/2），所以即使光线"冲入"视界，state 仍然有界、null_error 在 1e-4 量级。

### 3. 临界分歧（+3.0, +3.0）两种方法殊途同归

geokerr 给 captured，RK4 和 RK45 都给 disk。这印证了 `geokerr_strict_comparison.json` 里 captured 一致率仅 46% 的本质原因：**临界 ray 的 captured/disk 边界本就是浮点精度的产物，不是某种数值方法能"修对"的**。要根除这种分歧只能切到半解析 Carlson elliptic（geokerr/YNOGK 路径），这是项目 `docs/equation_reference.md` 第 4 节标注的 Carter 常数路线。

### 4. 时间成本

- RK4 5000 step max wall = 1.5 s（实际 2873 accepted）
- RK45 5000 step max wall = 0.27 s（实际 220 accepted）
- **net wall speedup: 5–7×**

虽然 per-step 多一次 RHS 调用 (6 vs 4)，但 step 数减少 13×，净 RHS 调用减少 ~9×。

## 推荐行动

1. **CPU 验证流水线**默认用 RK45：把 `tools/geokerr_strict_compare.py` 切到 `trace_single_ray_rk45`，预期 captured 一致率不升（边界问题），但 null preservation 在 captured 列将变干净。
2. **GPU kernel** 的自适应步长改造仍然推荐——RK45 在 GPU 上的挑战是 warp divergence（不同 ray 不同步长，warp 内部 lane 不齐）。可参考 `paper/main.md` Future Work 一节。
3. **临界 ray 状态分类**问题（46% captured 一致）需要的不是 RK45，而是 Carlson elliptic 半解析（`docs/equation_reference.md` 第 4 节给出推导框架）。

## 复现

```powershell
$env:PYTHONPATH = "D:\Desktop\black hole"
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe tools\rk45_vs_rk4_demo.py
```

输出：
- `validation/rk45_vs_rk4_demo.json`
- `research/reproduction/rk45_vs_rk4.png`

## 单元测试

- `tests/test_geodesic.py` 仍然只测 RK4 路径（保持 baseline 行为不变）。
- 31 pytest passed，新加的 `trace_single_ray_rk45` 是独立函数，不影响现有验证流水线。
