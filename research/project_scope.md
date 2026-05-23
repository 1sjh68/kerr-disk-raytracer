# Project Scope

更新时间：2026-05-21（文档对齐）

## In Scope

- Kerr metric and inverse metric in Boyer-Lindquist coordinates.
- Horizon and ISCO formulas in `G = c = M = 1` units.
- CPU reference modules for geodesic RHS and RK integration (RK4 + RK45).
- **Dual rendering pipelines** with a unified CPU/GPU output contract:
  - **Fast thin-disk MVP** (`run_cpu.py`, `run_gpu.py`): vectorized screen-space thin-disk model for quick images and baseline CUDA validation.
  - **Full Hamiltonian geodesic path** (`run_geodesic_cpu.py`, `run_geodesic_gpu.py`): per-pixel null geodesic tracing with RK4 fixed-step integration; CUDA kernels `kerr_geodesic_kernel` (float32) and `kerr_geodesic_kernel_double` (float64).
- Thin equatorial disk image maps: intensity, redshift, temperature, hit mask, status code, RGB.
- Validation scripts: fast-path (`validate.py`), geodesic path (`validate_geodesic.py`), extended suite, geokerr cross-validation.
- Report-ready figures, performance artifacts, paper PDF and presentation PPT.

## Out of Scope (博士级扩展)

- Full EHT M87* / Sgr A* observational inference.
- GRMHD data ingestion.
- Polarized Stokes transport.
- Synchrotron emission/absorption.
- Production-grade adaptive geodesic integration on GPU.
- Multi-GPU / differentiable tracing / surrogate models.

## Current Pipeline Boundary

The repository now contains **two complementary render paths**, not a single MVP waiting for geodesic replacement:

| 路径 | 入口 | 用途 |
|------|------|------|
| Fast thin-disk | `run_cpu.py`, `run_gpu.py` | 快速成像、MVP 级 CPU/GPU 一致性（intensity MAE ~2e-10） |
| Full geodesic | `run_geodesic_cpu.py`, `run_geodesic_gpu.py` | 物理测地线参考；float64 @ 48×48 **99.96% 状态匹配（disk-hit 100%）** |

**术语约定**：
- **MVP** 指 fast thin-disk 管线，仍保留用于快速预览与 benchmark。
- **Geodesic path** 指完整 Hamiltonian 逐像素追踪，已是可用的 CPU/GPU 双实现；科学参考应使用 `--precision float64`。
- 产物命名带精度后缀：`gpu_geodesic_image_float32.png` / `_float64.png`（无后缀的旧文件名已废弃）。

**验证分工**：
- `validate.py` → fast thin-disk CPU/GPU 对比 → `validation/error_summary.md`
- `validate_geodesic.py` → geodesic CPU/GPU 对比 → `validation/geodesic_cpu_gpu_comparison.json`

**已知未完成**：geokerr 相同观察者/相机约定下的坐标级轨迹对齐、Carter 常数半解析对照、GPU adaptive 步长、偏振/GRMHD 等扩展。当前 geokerr 原始 abgrid 为 87% 状态一致；a=0.7/i=60 严格状态判定为 91.25% 总体一致、98.27% disk 一致。
