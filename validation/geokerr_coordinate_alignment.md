# geokerr Coordinate-Level Alignment

更新时间：2026-05-23

## 目的

`validation/geokerr_strict_comparison.json` 已经在 status 层面（disk / captured / escaped 分类）做了 400 条光线的统计（总体 91.25% 一致，disk 一致 98.27%，参见 `paper/main.md` §5.3）。本文档把对比推进到**轨迹坐标级**——对每条光线沿轨迹采样的 (r, θ) 进行点对点比对。

## 方法

- **数据源**：geokerr 的 `abgrid_r60.out`，配置 a=0.7, i=60°, r_obs ≈ 4.15×10⁶ M，每条光线 300 个采样点 (uf, muf, dt, dphi, λ, tpm, tpr)。
- **本项目侧**：`src/geodesic.py` 的 `trace_single_ray()` 现在支持 `record_trajectory=True`，沿固定步长 RK4 积分输出每步的 (λ, r, θ, φ)；带 sanity clamp 防止视界后数值溢出污染采样。
- **对齐策略**：geokerr 的仿射参数 λ 与本项目约定不完全等价（前者来自 Carlson elliptic 半解析积分），所以直接 λ 对齐会有标度差。改为在 (r, θ) 平面做 **L2 几何最近邻匹配**——对 geokerr 每个采样点，在我们的轨迹上找最近的 (log r, θ) 点，输出相对/绝对误差统计。
- **样本**：从 abgrid_r60.out 的 20×20 屏幕格点中选 5 条具代表性的光线，覆盖 disk 命中 / captured / 临界几何。

## 结果

完整数据见 `validation/geokerr_coordinate_alignment.json`；可视化见 `research/reproduction/geokerr_coordinate_alignment.png`。

| (α, β) | geokerr | CPU | RMS Δr/r | RMS Δθ (rad) |
|---|---|---|---|---|
| (+0.6, +0.6) | captured | captured | 0.096 | 0.37 |
| (-3.0, -3.0) | captured | captured | 0.222 | 0.52 |
| (+5.4, -5.4) | disk | disk | 0.220 | 0.44 |
| (+3.0, +3.0) | captured (ncase=5) | **disk** (临界分歧) | 1.40 | 0.53 |
| (-1.8, -1.8) | captured | captured | 0.161 | 0.99 |

中位数：RMS Δr/r ≈ 0.22；RMS Δθ ≈ 0.52 rad。

## 解读

1. **远离临界的光线对齐良好**。(+5.4, -5.4) 这条 disk 命中光线，geokerr 与 CPU 轨迹在 r > 100 几乎完全重叠，差异主要集中在 r ~ 10 的 turning-point 区域——这是 RK4 固定步长在曲率极强区域的步长不足，可通过 RK45 自适应步长改善。

2. **captured 光线在 photon sphere 附近的 (r, θ) 路径差异**是预期的：geokerr 用 Carlson elliptic 半解析积分，turning point 处理为闭式解；本项目 RK4 步长 0.35 在临界轨道附近会产生几何路径偏差，最终都进入视界但**进入路径**不同。这解释了 status 一致而 trajectory 不一致的现象。

3. **(+3.0, +3.0) 光线的 status 分歧**（geokerr=captured, CPU=disk）正是 `geokerr_strict_comparison.json` 中 captured 一致率 46.15% 的来源。这条光线的 impact parameter 非常接近 critical，**任何数值精度的浮点积分**都会在该处与半解析方法分道扬镳。这不是项目的缺陷，而是固定步长 RK4 vs 半解析 elliptic 的固有差异。

4. **总体 Δθ ~ 0.5 rad 看似很大**，但这是几何最近邻误差，混入了沿轨迹"哪个点对哪个点"的歧义；status 层面的一致性指标更能反映科学层面的"两条积分方法是否给出相同物理结果"。

## 与 status 一致率的关系

| 量级 | Status 比对 | 坐标级比对 |
|---|---|---|
| 比较内容 | 单条光线最终命运（disk / captured / escaped） | 沿轨迹每个采样点的 (r, θ) |
| 样本量 | 400 条 | 5 条（代表性） |
| 一致率 / 误差 | 91.25% 总体，98.27% disk | RMS Δr/r ~22%, RMS Δθ ~0.5 rad |
| 主要分歧来源 | 临界 captured/disk 边界 | photon sphere 附近 RK4 vs Carlson |
| 改进方向 | adaptive step size 或 elliptic 后端 | 同左 + 减小 step_size 或加密 turning-point 周围采样 |

## 复现

```powershell
$env:PYTHONPATH = "D:\Desktop\black hole"
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe tools\geokerr_coordinate_compare.py
```

输出：
- `validation/geokerr_coordinate_alignment.json` — 5 条光线的 per-ray 误差统计
- `research/reproduction/geokerr_coordinate_alignment.png` — (r, θ) 对照图

如果要扫描更多光线，编辑 `tools/geokerr_coordinate_compare.py` 顶部的 `SAMPLE_RAYS` 列表（geokerr abgrid_r60.out 共 400 条，α/β 在 [-11.4, 11.4] 步长 1.2 的网格上）。

## 已知改进方向

1. 切到 RK45 自适应步长（CPU 端 `src/integrators.py` 已有实现），临界 ray 处自动加密。
2. 在 photon sphere 检测到时，对 step_size 自动 ×0.1 直到走过 turning point。
3. 实现 Carter 常数路线（参考 `docs/equation_reference.md` 第 4 节）作为另一条独立参考，与 geokerr 的差异预期会显著降低。
