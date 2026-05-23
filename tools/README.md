# `tools/` Catalogue

> 项目根目录的 `run_*.py`、`validate*.py`、`benchmark.py` 是**主流水线**入口。
> `tools/` 收录所有**辅助 / 实验性 / 一次性**脚本。每条对应一种复现入口。

## 入口一览（按用途分组）

### 用户面入口（一键运行）

| 脚本 | 作用 | 输出 |
|---|---|---|
| `run_ncu_pipeline.ps1` | Windows host **ncu profile 一键流水线**：弹一次 UAC，自动跑 4 个 `--set full` run | `results/ncu_*.{ncu-rep,summary.txt,csv}` |
| `_ncu_full_pipeline.ps1` | 上面那个 admin 提权后真正调用的流水线（不要直接跑） | 同上 |
| `wsl_profile_pipeline.sh` | 在 WSL2 中用 nvcc + ptxas 出 profile 报告（无 ncu） | `results/wsl_profile_report.md` |
| `ncu_pipeline.md` | 上述脚本的使用说明 | — |

### 性能 benchmark / sweep

| 脚本 | 作用 | 输出 |
|---|---|---|
| `benchmark_fastmath.py` | float32 baseline vs `--use_fast_math`，多分辨率 | `results/fastmath_benchmark.json`、`figures/fastmath_speedup.png` |
| `benchmark_opt.py` | float64 baseline vs float64_opt kernel | `results/optimization_benchmark.json` |
| `parameter_sweep.py` | spin × inclination 24 组 GPU geodesic | `results/parameter_sweep.{json,csv}`、`figures/sweep/sweep_*.png` |
| `disk_param_sweep.py` | disk r_outer + emissivity index 各 6 组 | `results/disk_*_sweep.json`、`figures/disk_*_comparison.png` |
| `resolution_sweep.py` | float32/64 多分辨率 kernel 时间 | `results/resolution_sweep_*.json` |
| `stress_test_512.py` / `stress_test_1024.py` | 高分辨率压力测试 | `figures/gpu_geodesic_*_float64.png` |
| `compose_animations.py` | 把 24 帧 sweep 合成 GIF | `figures/*_animation.gif` |
| `render_color_compare.py` | CIE1931 vs approx 颜色对比 + black-body locus | `figures/cie_vs_approx_comparison.png`、`figures/blackbody_locus.png` |

### 验证 / 交叉验证

| 脚本 | 作用 | 输出 |
|---|---|---|
| `validation_suite.py` | Schwarzschild 阴影 / 步长收敛 / 自旋单调 / 倾角 4 项扩展验证 | `validation/extended_validation.json` |
| `geokerr_strict_compare.py` | 与 geokerr 400 ray status 级对比 | `validation/geokerr_strict_comparison.json` |
| `geokerr_coordinate_compare.py` | **坐标级** (r, θ) 对齐（5 ray sample） | `validation/geokerr_coordinate_alignment.{json,md}`、`research/reproduction/geokerr_coordinate_alignment.png` |
| `rk45_vs_rk4_demo.py` | RK4 vs Dormand-Prince RK45 在临界 ray 上对比 | `validation/rk45_vs_rk4_demo.{json,md}`、`research/reproduction/rk45_vs_rk4.png` |
| `analyze_geokerr_compare.py` | 旧版 geokerr 状态汇总分析（保留） | stdout |

### 物理扩展 demo

| 脚本 | 作用 | 输出 |
|---|---|---|
| `polarization_demo.py` | T10 偏振 stub 端到端：48×48 Stokes I, Q, U + EVPA | `figures/polarization_*.png`、`results/polarization_demo.json` |
| `eht_metrics_demo.py` | T11 ring diameter / asymmetry / photon-ring 24 配置 | `results/eht_metrics.{json,_report.md}` |

### NVIDIA 权限 / 驱动设置

| 脚本 | 作用 |
|---|---|
| `enable_gpu_profiling.reg` | 写入 `HKLM\SOFTWARE\NVIDIA Corporation\GPU Profiling\EnableGpuProfiling=1`（host Windows） |
| `enable_gpu_profiling.ps1` | 同上的 PowerShell 版本（带管理员检查）|

### 其他

| 脚本 | 作用 |
|---|---|
| `profile_kernel.py` | 最小 float64 kernel launcher，用于 ncu 单独 profile target |

## 一键复现项目

```powershell
# Windows
.\scripts\run_all.ps1
```

```bash
# WSL / Linux
./scripts/run_all.sh
```

`run_all` 会按顺序跑核心 CPU/GPU 流水线 + 验证 + 本目录下大部分 demo + pytest。
新增 demo 用 `try/catch`（PowerShell）或 `soft_run`（bash）包裹，失败不会终止流水线。

## 直接跑单个

```powershell
# 任意一个脚本，都要先把项目根设进 PYTHONPATH:
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe tools\benchmark_fastmath.py
```

```bash
PYTHONPATH=. python3 tools/benchmark_fastmath.py
```
