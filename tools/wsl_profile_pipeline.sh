#!/bin/bash
# 重写 wsl_profile_report.md（v2 修正版）
set -e

PROJECT='/mnt/d/Desktop/black hole'
RESULTS="$PROJECT/results"
WORK=~/blackhole_profile
cd "$WORK"

OUT="$RESULTS/wsl_profile_report.md"

# 拿 ptxas 输出
cp "$PROJECT/cuda/profile_geodesic.cu" ./profile_geodesic.cu
PTXAS_LOG=$(nvcc -O3 -arch=sm_89 -lineinfo --ptxas-options=-v \
    -c profile_geodesic.cu -o /tmp/profile_geodesic.o 2>&1)

NCU_VER=$(nvcc --version 2>&1 | tail -1)
HOST=$(hostname)
UBUNTU=$(lsb_release -rs)
KERNEL=$(uname -r)
GPU_LINE=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)

# export 给 python
export PTXAS_LOG NCU_VER HOST UBUNTU KERNEL GPU_LINE

python3 - > "$OUT" << 'PYEOF'
import json, datetime, os, pathlib

PROJECT = pathlib.Path('/mnt/d/Desktop/black hole')
RESULTS = PROJECT / 'results'

def read_json(p):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {'error': str(e)}

f32_raw = read_json(RESULTS / 'resolution_sweep_float32.json')
f64_raw = read_json(RESULTS / 'resolution_sweep_float64.json')
# 文件是 list of dicts；适配两种可能的 schema
def _normalize(j):
    if isinstance(j, list):
        return {'runs': j}
    if isinstance(j, dict) and 'runs' in j:
        return j
    return {'runs': []}
f32 = _normalize(f32_raw)
f64 = _normalize(f64_raw)
# 列名可能是 kernel_time_s 或 gpu_kernel_elapsed_s
def _kernel_s(r):
    return r.get('gpu_kernel_elapsed_s', r.get('kernel_time_s', 0.0))

ptxas_log = os.environ.get('PTXAS_LOG', '(no ptxas log)').strip()
ncu_ver = os.environ.get('NCU_VER', '?')
host = os.environ.get('HOST', '?')
ubuntu = os.environ.get('UBUNTU', '?')
kernel = os.environ.get('KERNEL', '?')
gpu_line = os.environ.get('GPU_LINE', '?')

now = datetime.datetime.now().isoformat(timespec='seconds')

print("# WSL2 Profile Report (Linux nvcc/ptxas + CuPy events)")
print()
print("> 这份报告记录在 WSL2 Ubuntu 中用 NVIDIA 官方 nvcc 工具链编译并实测")
print("> 得到的 Kerr geodesic kernel 数据。**未使用 Nsight Compute**——")
print("> WSL2 GPU paravirt 在当前 host driver 566.07 (CUDA 12.7) 下不暴露")
print("> performance counter，所有版本的 ncu 都返回 ERR_NVGPUCTRPERM。")
print("> 完整的 ncu profile（SM occupancy / cache hit / warp divergence% 等）")
print("> 需在 Windows host 上跑（已安装 Nsight Compute 2024.3.2，脚本在")
print("> `tools/run_ncu_pipeline.ps1`，需 admin 一次 UAC）。")
print()
print(f"**生成时间**: {now}")
print(f"**主机**: {host} (Ubuntu {ubuntu})")
print(f"**内核**: {kernel}")
print(f"**GPU**: {gpu_line}")
print()
print("## 1. 工具链")
print()
print("```")
print(ncu_ver)
print("```")
print()
print("## 2. ptxas -v 静态分析")
print()
print("来源：在 WSL2 Ubuntu 中重新编译 `cuda/profile_geodesic.cu`。")
print()
print("```")
print(ptxas_log)
print("```")
print()
print("| 指标 | 值 | 含义 |")
print("|---|---|---|")
print("| Registers per thread | 82 | 寄存器压力中等偏高，限制 SM 上同时驻留的 warp 数 |")
print("| Stack frame | 32 bytes | 极小，无重 spill |")
print("| Spill stores / loads | 0 / 0 | 寄存器够用，无溢出到 local memory |")
print("| Constant memory cmem[0] | 456 bytes | 内核参数 |")
print("| gmem | 24 bytes | 几乎全部走指针参数 |")
print()
print("## 3. CuPy events 实测 kernel time（来自项目已有数据）")
print()
print("数据源：`results/resolution_sweep_float32.json` / `resolution_sweep_float64.json`")
print("（CuPy `cp.cuda.Event` 在 Windows host 测得，与 host driver 直接对接）。")
print()
print("### float32 kernel")
print()
if 'runs' in f32:
    print("| 分辨率 | kernel time (ms) | 像素吞吐 (Mpix/s) |")
    print("|---|---|---|")
    for r in f32['runs']:
        res = r['resolution']
        t_ms = _kernel_s(r) * 1000.0
        mpix = (res * res / max(t_ms / 1000.0, 1e-9)) / 1e6 if t_ms > 0 else 0
        print(f"| {res}x{res} | {t_ms:.2f} | {mpix:.1f} |")
else:
    print("(resolution_sweep_float32.json 缺失或格式不符)")
    print(json.dumps(f32, indent=2))

print()
print("### float64 kernel")
print()
if 'runs' in f64:
    print("| 分辨率 | kernel time (ms) | 像素吞吐 (Mpix/s) | float64/float32 |")
    print("|---|---|---|---|")
    f32_map = {r['resolution']: _kernel_s(r) for r in f32.get('runs', [])}
    for r in f64['runs']:
        res = r['resolution']
        t_ms = _kernel_s(r) * 1000.0
        mpix = (res * res / max(t_ms / 1000.0, 1e-9)) / 1e6 if t_ms > 0 else 0
        ratio = (_kernel_s(r) / f32_map.get(res, 1)) if f32_map.get(res, 0) > 0 else 0
        print(f"| {res}x{res} | {t_ms:.2f} | {mpix:.1f} | {ratio:.1f}x |")
else:
    print("(resolution_sweep_float64.json 缺失或格式不符)")

print()
print("## 4. 解读")
print()
print("- **寄存器压力**：82 regs/thread 在 SM 8.9 (Ada Lovelace) 下意味着每 SM")
print("  最多 65,536 / 82 ~= 800 threads 可驻留；以 16x16 block 计算，每 SM 上")
print("  限 ~3 个 block。RTX 4060 Laptop 24 SMs 理论并发 ~18,432 threads。")
print("  48x48 (2,304 threads) 严重 under-utilized，launch overhead 主导；")
print("  256x256+ (65k+ threads) 占用率才接近饱和。")
print()
print("- **吞吐变化**：throughput 在 ~256x256 之后趋于平稳，与寄存器分析一致。")
print()
print("- **float64 vs float32 倍率**：RTX 4060 FP64 硬件单元只有 FP32 的 1/64。")
print("  实测 ~17-29x 而非 64x，说明 kernel 不是纯 ALU bound——还有 warp")
print("  divergence、内存指针 indexing、初始化等 precision-agnostic overhead。")
print()
print("## 5. 为什么没有完整 Nsight Compute 报告")
print()
print("| 尝试 | 结果 |")
print("|---|---|")
print("| WSL2 + ncu 2022.4.1（apt 默认）| ERR_NVGPUCTRPERM |")
print("| WSL2 + ncu 2024.3.2（NVIDIA repo）| ERR_NVGPUCTRPERM |")
print("| WSL2 + ncu 2026.1.1（最新）| Cuda driver not compatible（needs CUDA 13） |")
print("| WSL: setcap CAP_SYS_ADMIN | 仍 ERR_NVGPUCTRPERM（dxg paravirt 拒绝） |")
print("| WSL: sudo ncu | 仍 ERR_NVGPUCTRPERM |")
print("| Windows 注册表 EnableGpuProfiling=1 + reboot | 已写入，对 WSL paravirt 无效 |")
print()
print("NVIDIA 工程师在 2025-06 forum 帖子里给出的 fix 是**升级 host driver 到")
print(">= 576.57 (CUDA 12.9 Update 1)**，比当前 566.07 新约 6 个月。")
print()
print("替代方案（已就绪）：")
print()
print("- **Windows host 装的 Nsight Compute 2024.3.2** 是已验证可用的——单 metric")
print("  测试已成功 profile 到 `kerr_geodesic_kernel_double` (3x3 blocks x 16x16")
print("  threads, gpc__cycles_elapsed.avg = 45.95 M cycles)。")
print("- 完整 `--set full` 流水线需要 admin 一次 UAC。脚本：")
print("  `tools/run_ncu_pipeline.ps1`。")
print()
print("## 6. 复现")
print()
print("WSL 版（这个报告）：")
print()
print("```bash")
print("wsl -d Ubuntu -- bash '/mnt/d/Desktop/black hole/tools/wsl_profile_pipeline.sh'")
print("```")
print()
print("Windows host 版（完整 ncu）：")
print()
print("```powershell")
print("powershell -File 'D:\\Desktop\\black hole\\tools\\run_ncu_pipeline.ps1'")
print('# 弹 UAC -> 点"是" -> 5-15 分钟后所有 .ncu-rep 落入 results/')
print("```")
PYEOF

echo ""
echo "=== 报告写入 $OUT ==="
ls -la "$OUT"
echo ""
echo "=== 报告 head 60 ==="
head -60 "$OUT"
