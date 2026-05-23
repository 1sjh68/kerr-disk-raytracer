# 跑完整 Nsight Compute profile

> WSL2 路径已确认在 driver 566.07 / CUDA 12.7 下不可行（详见 `results/wsl_profile_report.md` 的失败矩阵）。本指南只覆盖 Windows host 路径。

## 一键流程

```powershell
powershell -File 'D:\Desktop\black hole\tools\run_ncu_pipeline.ps1'
```

它会：

1. 提示按任意键启动 → 弹一次 UAC（屏幕变暗 + "是/否"对话框）
2. 你点 **是**
3. 自动跑 4 个 `ncu --set full`：
   - float64 @ 48×48
   - float64 @ 128×128
   - float32 @ 48×48
   - float32 @ 128×128
4. 每个 run 导出 `.ncu-rep` + `.summary.txt` + `.csv` 到 `results/`
5. 整体耗时 5–15 分钟（取决于 GPU 状态）

期间不要做其他 GPU 重活，让 ncu 独占 GPU 拿稳定数据。

## 期望产物

```
results/ncu_float64_48.ncu-rep      results/ncu_float64_48.summary.txt   results/ncu_float64_48.csv
results/ncu_float64_128.ncu-rep     results/ncu_float64_128.summary.txt  results/ncu_float64_128.csv
results/ncu_float32_48.ncu-rep      results/ncu_float32_48.summary.txt   results/ncu_float32_48.csv
results/ncu_float32_128.ncu-rep     results/ncu_float32_128.summary.txt  results/ncu_float32_128.csv
```

`.ncu-rep` 是 NVIDIA Nsight Compute UI 可打开的二进制报告（含完整 sections：occupancy / SM utilization / memory chart / source-correlated metrics 等）。

## 内部结构

- `tools/run_ncu_pipeline.ps1` ← 你直接运行这个（普通用户即可）
- `tools/_ncu_full_pipeline.ps1` ← 上面那个 elevate 后调用的，admin 跑的脚本

## 已就绪的依赖

| 依赖 | 路径 / 版本 |
|---|---|
| Nsight Compute | `C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\` |
| Driver | 566.07 (CUDA 12.7) |
| Project venv | `D:\Desktop\black hole\.venv\Scripts\python.exe` |
| Profile target | `run_geodesic_gpu.py --precision {float32,float64} [--resolution N]` |

## 故障排查

**UAC 被拒绝**：脚本会立刻报错退出，重新运行即可。

**`ERR_NVGPUCTRPERM` 重新出现**：以管理员模式跑时不应该出现。如果出现，检查注册表：
```powershell
reg query "HKLM\SOFTWARE\NVIDIA Corporation\GPU Profiling"
# 应该有 EnableGpuProfiling = 0x1
```
如果不在或值不对，导入：
```powershell
reg import 'D:\Desktop\black hole\tools\enable_gpu_profiling.reg'
# 然后重启系统
```

**ncu 找不到 kernel**：检查 `--kernel-name` 是否匹配 `cuda/kernels.cu` 当前的 kernel 命名。当前两个 kernel 是 `kerr_geodesic_kernel`（float32）和 `kerr_geodesic_kernel_double`（float64）。

## 如果只想跑一个，不要全部 4 个

直接调底层 ncu：

```powershell
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.bat'
$py  = 'D:\Desktop\black hole\.venv\Scripts\python.exe'
cd 'D:\Desktop\black hole'

# 必须以 admin 跑
& $ncu --set full --kernel-name kerr_geodesic_kernel_double `
    --launch-skip 0 --launch-count 1 `
    --target-processes all `
    --export 'results\ncu_my_run' --force-overwrite `
    $py run_geodesic_gpu.py --precision float64 --resolution 256
```

## WSL 端工具链报告（不含 ncu）

```bash
wsl -d Ubuntu -- bash '/mnt/d/Desktop/black hole/tools/wsl_profile_pipeline.sh'
# 输出：results/wsl_profile_report.md（ptxas + CuPy events 数据）
```
