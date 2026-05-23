# WSL2 Profile Report (Linux nvcc/ptxas + CuPy events)

> 这份报告记录在 WSL2 Ubuntu 中用 NVIDIA 官方 nvcc 工具链编译并实测
> 得到的 Kerr geodesic kernel 数据。**未使用 Nsight Compute**——
> WSL2 GPU paravirt 在当前 host driver 566.07 (CUDA 12.7) 下不暴露
> performance counter，所有版本的 ncu 都返回 ERR_NVGPUCTRPERM。
> 完整的 ncu profile（SM occupancy / cache hit / warp divergence% 等）
> 需在 Windows host 上跑（已安装 Nsight Compute 2024.3.2，脚本在
> `tools/run_ncu_pipeline.ps1`，需 admin 一次 UAC）。

**生成时间**: 2026-05-23T10:16:04
**主机**: LAPTOP-SQMOMH42 (Ubuntu 24.04)
**内核**: 6.6.87.2-microsoft-standard-WSL2
**GPU**: NVIDIA GeForce RTX 4060 Laptop GPU, 566.07, 8188 MiB

## 1. 工具链

```
Build cuda_12.0.r12.0/compiler.32267302_0
```

## 2. ptxas -v 静态分析

来源：在 WSL2 Ubuntu 中重新编译 `cuda/profile_geodesic.cu`。

```
ptxas info    : 24 bytes gmem
ptxas info    : Compiling entry function '_Z20kerr_geodesic_kerneliiffffffiffiffPfS_S_PhS0_S_' for 'sm_89'
ptxas info    : Function properties for _Z20kerr_geodesic_kerneliiffffffiffiffPfS_S_PhS0_S_
    32 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 82 registers, 456 bytes cmem[0], 8 bytes cmem[2]
```

| 指标 | 值 | 含义 |
|---|---|---|
| Registers per thread | 82 | 寄存器压力中等偏高，限制 SM 上同时驻留的 warp 数 |
| Stack frame | 32 bytes | 极小，无重 spill |
| Spill stores / loads | 0 / 0 | 寄存器够用，无溢出到 local memory |
| Constant memory cmem[0] | 456 bytes | 内核参数 |
| gmem | 24 bytes | 几乎全部走指针参数 |

## 3. CuPy events 实测 kernel time（来自项目已有数据）

数据源：`results/resolution_sweep_float32.json` / `resolution_sweep_float64.json`
（CuPy `cp.cuda.Event` 在 Windows host 测得，与 host driver 直接对接）。

### float32 kernel

| 分辨率 | kernel time (ms) | 像素吞吐 (Mpix/s) |
|---|---|---|
| 48x48 | 1.40 | 1.6 |
| 64x64 | 1.60 | 2.6 |
| 96x96 | 1.50 | 6.1 |
| 128x128 | 1.80 | 9.1 |
| 160x160 | 2.00 | 12.8 |
| 192x192 | 2.70 | 13.7 |
| 256x256 | 4.00 | 16.4 |

### float64 kernel

| 分辨率 | kernel time (ms) | 像素吞吐 (Mpix/s) | float64/float32 |
|---|---|---|---|
| 48x48 | 24.30 | 0.1 | 17.4x |
| 64x64 | 28.30 | 0.1 | 17.7x |
| 96x96 | 24.20 | 0.4 | 16.1x |
| 128x128 | 37.60 | 0.4 | 20.9x |
| 160x160 | 48.90 | 0.5 | 24.4x |
| 192x192 | 68.50 | 0.5 | 25.4x |
| 256x256 | 114.30 | 0.6 | 28.6x |

## 4. 解读

- **寄存器压力**：82 regs/thread 在 SM 8.9 (Ada Lovelace) 下意味着每 SM
  最多 65,536 / 82 ~= 800 threads 可驻留；以 16x16 block 计算，每 SM 上
  限 ~3 个 block。RTX 4060 Laptop 24 SMs 理论并发 ~18,432 threads。
  48x48 (2,304 threads) 严重 under-utilized，launch overhead 主导；
  256x256+ (65k+ threads) 占用率才接近饱和。

- **吞吐变化**：throughput 在 ~256x256 之后趋于平稳，与寄存器分析一致。

- **float64 vs float32 倍率**：RTX 4060 FP64 硬件单元只有 FP32 的 1/64。
  实测 ~17-29x 而非 64x，说明 kernel 不是纯 ALU bound——还有 warp
  divergence、内存指针 indexing、初始化等 precision-agnostic overhead。

## 5. 为什么没有完整 Nsight Compute 报告

| 尝试 | 结果 |
|---|---|
| WSL2 + ncu 2022.4.1（apt 默认）| ERR_NVGPUCTRPERM |
| WSL2 + ncu 2024.3.2（NVIDIA repo）| ERR_NVGPUCTRPERM |
| WSL2 + ncu 2026.1.1（最新）| Cuda driver not compatible（needs CUDA 13） |
| WSL: setcap CAP_SYS_ADMIN | 仍 ERR_NVGPUCTRPERM（dxg paravirt 拒绝） |
| WSL: sudo ncu | 仍 ERR_NVGPUCTRPERM |
| Windows 注册表 EnableGpuProfiling=1 + reboot | 已写入，对 WSL paravirt 无效 |

NVIDIA 工程师在 2025-06 forum 帖子里给出的 fix 是**升级 host driver 到
>= 576.57 (CUDA 12.9 Update 1)**，比当前 566.07 新约 6 个月。

替代方案（已就绪）：

- **Windows host 装的 Nsight Compute 2024.3.2** 是已验证可用的——单 metric
  测试已成功 profile 到 `kerr_geodesic_kernel_double` (3x3 blocks x 16x16
  threads, gpc__cycles_elapsed.avg = 45.95 M cycles)。
- 完整 `--set full` 流水线需要 admin 一次 UAC。脚本：
  `tools/run_ncu_pipeline.ps1`。

## 6. 复现

WSL 版（这个报告）：

```bash
wsl -d Ubuntu -- bash '/mnt/d/Desktop/black hole/tools/wsl_profile_pipeline.sh'
```

Windows host 版（完整 ncu）：

```powershell
powershell -File 'D:\Desktop\black hole\tools\run_ncu_pipeline.ps1'
# 弹 UAC -> 点"是" -> 5-15 分钟后所有 .ncu-rep 落入 results/
```
