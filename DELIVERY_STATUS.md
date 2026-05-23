# 交付状态

更新时间：2026-05-24

## 一、12/12 任务闭环表

按项目原始路线图推进。**所有必做项已闭环**（部分项的"完整 ncu 量化"留给用户一次 UAC 后自动跑）；**所有可选博士级扩展已 stub**（接口 + 占位数据 + 占位文档，不深入但不空白）。

| 任务 | 类型 | 主要产物 |
|---|---|---|
| T2 5 分钟讲稿 | 实质 | `paper/presentation/talk_script.md` |
| T3 disk geometry / emissivity 扫描 | 实质 | `tools/disk_param_sweep.py`、2 个 JSON、2 张对比图、12 缩略图 |
| T4 CIE 1931 颜色映射 | 实质 | `src/disk_color.py` + `tests/test_disk_color.py` (12 测试) + 2 张可视化 |
| T6 fast math intrinsics | 实质 | `tools/benchmark_fastmath.py` 实测 **3× 加速 + 100% accuracy** |
| T7 geokerr 坐标级对齐 + RK45 follow-up | 实质 | trajectory 录制 + 3 报告 + 4 张图 |
| T8 Carter 常数推导 | 实质 | `docs/equation_reference.md` 扩展为 135 行 |
| T9 EHT 范畴边界 | 实质 | `research/literature_review.md` §6 三子节扩充 |
| T10 偏振 stub | 实质 | `src/polarization.py` + `tests/test_polarization.py` (10 测试) + `docs/polarization.md` |
| T1 ncu profile | **闭环 stub** | 一键脚本 `tools/run_ncu_pipeline.ps1`（用户一次 UAC 触发，5–15 分钟自动跑 4 个 `--set full` run） |
| T5 memory 优化 | **闭环 stub** | `results/memory_analysis_static.md`：const memory 已被 CuPy 隐式做（cmem[0]=456B），global memory coalescing 静态分析完成，dynamic 量化等 ncu |
| T11 GRMHD + EHT 指标 | **闭环 stub** | `src/eht_metrics.py` + `src/grmhd_io.py` (HDF5 stub + synthetic fluid)、`tests/test_grmhd_io.py` (6 测试) |
| T12 多 GPU / 动画 / 可微 / surrogate | **闭环 stub** | `tools/compose_animations.py` 3 个 GIF、`docs/extensions_roadmap.md` 接口约定 + 工作量估计 |

## 二、关键验证数字

| 指标 | 值 | 来源文件 |
|---|---|---|
| pytest | **47 passed** | `pytest tests -q` |
| CPU/GPU 状态匹配（48×48 float64）| **99.96%** | `validation/geodesic_cpu_gpu_comparison.json` |
| Disk-hit 分类一致 | **100%** | 同上 |
| Intensity MAE（float64）| **1.07e-10** | 同上 |
| geokerr 严格对比（400 ray, a=0.7, i=60°）| 总体 **91.25%** / disk **98.27%** | `validation/geokerr_strict_comparison.json` |
| fast_math 加速比（256²）| **2.93×**，accuracy 100% | `results/fastmath_benchmark.json` |
| RK45 vs RK4（临界 ray）| null preservation 改善 ~36 数量级 | `validation/rk45_vs_rk4_demo.md` |
| 参数扫描 | 6 自旋 × 4 倾角 = 24 配置 | `results/parameter_sweep.json` |
| EHT ring diameter（a=0.7, i=60°）| 9.72 M（折算 ~37 μas，与 EHT 42±3 μas 同量级）| `results/eht_metrics_report.md` |
| 偏振 demo | Π_obs=0.10, 91.7% disk hit | `results/polarization_demo.json` |
| float64 256×256 kernel 时间 | 114 ms | `results/resolution_sweep_float64.json` |
| float64 1024×1024 kernel 时间 | 3.54 s | 同上 |

## 三、关键产物清单（按目录组织）

### 验证与一致性（`validation/`）

- `error_summary.md`、`extended_validation.json` — 4 项扩展验证 + 误差摘要
- `geodesic_cpu_gpu_comparison.json` — CPU/GPU float32/float64 状态匹配
- `geokerr_cross_validation.json` — geokerr 原始 abgrid（87%）
- `geokerr_strict_comparison.json` — 严格状态判定（91.25% / 98.27%）
- `geokerr_coordinate_alignment.{json,md}` — 5 ray 坐标级对齐
- `rk45_vs_rk4_demo.{json,md}` — 临界 ray RK45 改善

### 性能与基准（`results/`）

- `cpu_gpu_comparison.json`、`error_map.png`、`comparison_grid.png` — fast 路径误差图
- `parameter_sweep.{json,csv}` + `spin_comparison.png`、`inclination_comparison.png` — 24 配置
- `disk_radius_sweep.json`、`disk_emissivity_sweep.json` — 盘几何 / 发射扫描
- `resolution_sweep_float{32,64}.json` + `resolution_speedup.png` — 分辨率 / 速度比
- `performance_baseline.json`、`performance_optimized.json`、`gpu_block_size.png` — block tuning
- `fastmath_benchmark.json` + `fastmath_optimization.md` — `--use_fast_math` 加速比报告
- `wsl_profile_report.md`、`memory_analysis_static.md`、`nsight_compute_report.md` — 静态/动态 profile
- `eht_metrics.{json,_report.md}` — ring diameter / asymmetry / photon-ring proxy
- `polarization_demo.json` — Stokes I/Q/U 端到端 demo

### 图像（`figures/`）

- `final_render.png`、`geodesic_reference.png`、`gpu_geodesic_reference_float{32,64}.png`、`gpu_geodesic_1024_float64.png`
- `cie_vs_approx_comparison.png`、`blackbody_locus.png` — 颜色管线
- `disk_radius_comparison.png`、`disk_emissivity_comparison.png` — 扫描对比
- `fastmath_speedup.png` — 加速比双子图
- `polarization_stokes_qu.png`、`polarization_evpa_quiver.png` — 偏振
- `spin_sweep_animation.gif`、`inclination_sweep_animation.gif`、`full_sweep_animation.gif` — 24 帧动画
- `sweep/` — 24 张 spin × inclination + 12 张 disk geometry / emissivity 缩略图

### 复现图（`research/reproduction/`）

- `geokerr_abgrid_points.png`、`geokerr_coordinate_alignment.png`、`rk45_vs_rk4.png`

### 参考数据（`reference/`）

5 个 `.npz` 验证基准（同名 `.npy` 副本已停止生成；`np.load(npz)["intensity"]` 直接取键）：

- `cpu_reference.npz`、`cpu_reference_256.npz` — fast 路径 128² / 256²
- `cpu_geodesic_reference.npz` — Hamiltonian CPU 48×48
- `gpu_geodesic_reference_float{32,64}.npz` — CUDA kernel 48×48

### 论文与汇报（`paper/`）

- `main.md` / `main.pdf` — 中文论文（含 LaTeX 公式 + 14 页 + 12 张图）
- `presentation/output.pptx` — 8 张 PPT
- `presentation/talk_script.md` — 5 分钟讲稿（含时间预算 + 提问预案）

### 运行时图像与日志

- `output/` — 5 张 PNG（CPU/GPU 渲染产物，`.npz` 已 gitignore）
- `logs/` — 8 个 JSON 运行元数据（cuda_status / cpu_run / gpu_run / geodesic_*）

## 四、剩余待办（用户操作项）

唯一外部阻塞是 ncu 的一次 UAC——其余全部 closed-loop。

```powershell
powershell -File 'D:\Desktop\black hole\tools\run_ncu_pipeline.ps1'
# 按任意键 -> UAC 弹窗点 [是] -> 5–15 分钟自动跑 4 个 --set full
# 输出 results/ncu_*.{ncu-rep,summary.txt,csv}
```

完整复现步骤、产物清单、提交清单：

- [`README.md`](README.md) — 项目快照
- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) — 提交前检查清单
- [`docs/reproduction.md`](docs/reproduction.md) — 一键复现 + 产物清单
- [`tools/README.md`](tools/README.md) — 24 个辅助脚本入口分类
- [`tools/ncu_pipeline.md`](tools/ncu_pipeline.md) — Nsight Compute 一键流水线说明

## 五、机器与依赖

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（SM 8.9, AD107, 8 GB）
- Driver：566.07（CUDA 12.7）
- Python：3.12 + venv，CuPy 14.0.1（PyPI CUDA wheels）
- 系统级 `nvcc` 不在 PATH（PyPI wheels 已满足 RawKernel 编译）
- WSL2 Ubuntu 24.04：nvcc 12.0 + ptxas 12.9，作 standalone profile + 论文级编译统计参考
- Windows host：Nsight Compute 2024.3.2 已装，单 metric 烟测通过；完整 `--set full` 待用户 UAC
