# Nsight Compute Analysis Report

更新时间：2026-05-23

> **当前状态**：
> - WSL2 ncu 路径**不可用**：当前 host driver 566.07 (CUDA 12.7) 在 dxg paravirt 这一层不暴露 GPU performance counter。已验证 ncu 2022.4.1 / 2024.3.2 / 2026.1.1 + setcap CAP_SYS_ADMIN + sudo + 注册表 EnableGpuProfiling=1 + 整机重启的所有组合，全部返回 ERR_NVGPUCTRPERM。NVIDIA 工程师在 2025-06 forum 帖子里给出的修复需要 driver ≥ 576.57 (CUDA 12.9 Update 1)。
> - Windows host ncu 路径**已就绪**：Nsight Compute 2024.3.2 已通过 CUDA 12.6.3 network installer 装入 `C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\`。已验证单 metric (`gpc__cycles_elapsed.avg`) 在 admin 模式下能成功 profile `kerr_geodesic_kernel_double` (3×3 blocks × 16×16 threads, **45.95 M cycles**)。
> - 完整 `--set full` 流水线需要 admin 一次 UAC 同意，由 `tools/run_ncu_pipeline.ps1` 一键触发。
>
> 本报告的 PTX/编译时静态指标（registers/spill/branches）已经是最终数据；动态 metrics (occupancy / cache hit / warp divergence%) 待用户运行一键 ncu 脚本后补全。

## Device
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (SM 8.9, AD107)
- Driver: 566.07 (CUDA 12.7)
- WSL2 toolchain: nvcc 12.0.140
- Windows host ncu: 2024.3.2

## Kernel Under Analysis
- `kerr_geodesic_kernel`        — float32 path
- `kerr_geodesic_kernel_double` — float64 path
- Source: `cuda/kernels.cu` (运行时编译), `cuda/profile_geodesic.cu` (standalone)
- Compile: `nvcc -O3 -arch=sm_89 -lineinfo --ptxas-options=-v`
- Grid: 默认 48×48 (3×3 blocks × 16×16 threads)，扫描到 1024×1024

## Compilation Metrics (ptxas -v, WSL Linux nvcc 12.0)

| Metric | Value | Notes |
|--------|-------|-------|
| Registers per thread | **82** | SM 8.9 寄存器上限 64K → 每 SM 最多约 800 threads 驻留 |
| Stack frame | 32 bytes | 极小，无重 spill |
| Spill stores / loads | 0 / 0 | 寄存器够用，无 local memory 溢出 |
| Constant memory cmem[0] | 456 bytes | 内核参数 |
| Constant memory cmem[2] | 8 bytes | 编译器内部常量 |
| Global memory gmem | 24 bytes | 几乎全部走指针参数 |
| PTX conditional branches | ~910 | per-ray 控制流路径多 |

## Branch / Divergence Indicators (PTX 静态)

Kernel 内有大量 per-ray 控制流：
- 视界命中（horizon）
- 逃逸（escape radius）
- 赤道面穿越 + 线性插值（disk hit）
- NaN/invalid 处理
- Max-steps fallback

相邻像素的光线命运（disk / captured / escaped）很可能不同，因此**显著 warp divergence 几乎必然存在**。具体百分比需 ncu profile 验证。

## Timing (CuPy Events, host Python)

数据源：`results/resolution_sweep_float32.json` / `resolution_sweep_float64.json`

### float32 kernel

| Resolution | Kernel time (ms) | Throughput (Mpix/s) |
|---|---|---|
| 48×48 | 1.40 | 1.6 |
| 64×64 | 1.60 | 2.6 |
| 96×96 | 1.50 | 6.1 |
| 128×128 | 1.80 | 9.1 |
| 160×160 | 2.00 | 12.8 |
| 192×192 | 2.70 | 13.7 |
| 256×256 | 4.00 | 16.4 |

### float64 kernel

| Resolution | Kernel time (ms) | Throughput (Mpix/s) | float64/float32 |
|---|---|---|---|
| 48×48 | 24.30 | 0.1 | 17.4× |
| 64×64 | 28.30 | 0.1 | 17.7× |
| 96×96 | 24.20 | 0.4 | 16.1× |
| 128×128 | 37.60 | 0.4 | 20.9× |
| 160×160 | 48.90 | 0.5 | 24.4× |
| 192×192 | 68.50 | 0.5 | 25.4× |
| 256×256 | 114.30 | 0.6 | 28.6× |

CPU 等效（48×48 float64）：~28–35 s →  **加速比约 1,200–1,400×**。

## Sample Dynamic Metric (admin Windows host, ncu 2024.3.2)

仅做了一个 metric 的烟测，验证 ncu 在 host 真的可用：

```
ncu --metrics gpc__cycles_elapsed.avg --target-processes all \
    .venv/Scripts/python.exe run_geodesic_gpu.py --precision float64
```

```
==PROF== Profiling "kerr_geodesic_kernel_double" - 0: 0%....50%....100% - 1 pass
kerr_geodesic_kernel_double (3, 3, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
  gpc__cycles_elapsed.avg = 45,954,969.67 cycles
```

完整 `--set full` 数据待 `tools/run_ncu_pipeline.ps1` 跑完后补到 `results/ncu_*.summary.txt` / `.csv` / `.ncu-rep`。

## Why WSL2 Path Failed (失败矩阵)

| 尝试 | 结果 |
|---|---|
| ncu 2022.4.1（apt 默认）| ERR_NVGPUCTRPERM |
| ncu 2024.3.2（NVIDIA cuda-wsl repo）| ERR_NVGPUCTRPERM |
| ncu 2026.1.1（最新）| `Cuda driver is not compatible with Nsight Compute`（needs CUDA 13） |
| `setcap cap_sys_admin+ep ncu` | 仍 ERR_NVGPUCTRPERM（dxg paravirt 在内核态拒绝） |
| `sudo ncu` | 仍 ERR_NVGPUCTRPERM |
| Windows 注册表 `EnableGpuProfiling=1` + 整机重启 | 已写入并生效，但对 WSL paravirt 通道无效 |

NVIDIA 工程师 (forum, 2025-06) 给出的 fix：升级 host driver 到 **≥ 576.57 (CUDA 12.9 Update 1)**，比当前 566.07 新约 6 个月。本项目暂不要求升级。

## Recommendations (基于 PTX + ptxas 静态分析)

1. **寄存器压力**：82 regs/thread 限制了每 SM 的 active warp 数（从 64 个理论上限掉到 ~25 个）。可以考虑：
   - 缩减 RK4 中间数组（当前 `state[8]`、`previous[8]`、`hit[8]`、`k1..4[8]`、`tmp[8]`，共 ~7×8 = 56 个 float ≈ 224 字节存储；其中很多被 promote 到寄存器）。
   - RK4 step 改为复用 `tmp[8]` 单一缓冲。
2. **Warp divergence**：per-ray early-exit 是主导成因。可选：
   - 按命运排序光线（内/外屏幕区域聚类），使同一 warp 内 fate 接近。
   - 用 warp-level `__ballot_sync` / `__shfl_sync` 压缩 divergent lanes。
3. **内存**：kernel 几乎全 compute-bound，几乎无 global memory traffic，**带宽不是瓶颈**。
4. **数据类型**：FP64 实测开销 17–29×（理论 64×），说明非 ALU 部分（divergence + 控制流）显著，纯精度提升的边际成本比理论值低。

完整动态验证（occupancy %、L1/L2 hit、scheduler issue rate、stall reasons、warp divergence %）待 `tools/run_ncu_pipeline.ps1` 跑完后补充。

## How to Reproduce / Run Full ncu

WSL 端（不含 ncu，只测 ptxas + nvcc + timing）：
```bash
wsl -d Ubuntu -- bash '/mnt/d/Desktop/black hole/tools/wsl_profile_pipeline.sh'
# 输出：results/wsl_profile_report.md
```

Windows host 端（完整 ncu）：
```powershell
powershell -File 'D:\Desktop\black hole\tools\run_ncu_pipeline.ps1'
# 弹一次 UAC -> 点 是 -> 5–15 分钟自动跑 4 个 --set full
# 输出：results/ncu_float32_48.{ncu-rep,summary.txt,csv}
#       results/ncu_float32_128.{ncu-rep,summary.txt,csv}
#       results/ncu_float64_48.{ncu-rep,summary.txt,csv}
#       results/ncu_float64_128.{ncu-rep,summary.txt,csv}
```
