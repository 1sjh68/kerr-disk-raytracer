# Reproduction Log

更新时间：2026-05-21（文档同步）

## Local Pipeline

已运行完整一键路径：

```powershell
.\scripts\run_all.ps1
```

一键路径当前包括：

```text
run_cpu.py
run_geodesic_cpu.py
scripts/check_cuda.py
run_gpu.py
run_geodesic_gpu.py --precision float32
run_geodesic_gpu.py --precision float64
render.py
benchmark.py
scripts/run_experiments.py
validate.py
validate_geodesic.py
scripts/summarize_external_demos.py
scripts/parse_geokerr.py
scripts/make_report.py
paper/presentation/build/build_deck.mjs
```

当前测试结果：

```text
19 passed
```

### Hamiltonian geodesic CPU reference

- resolution: `48x48`
- status counts: `disk=2114`, `captured=182`, `escaped=8`, `max_steps=0`, `invalid=0`
- disk hit fraction: `0.9175347222`
- elapsed: 约 `35 s`（本机最近一次日志）

### Geodesic GPU vs CPU（float64 @ 48×48）

- 状态匹配：**99.96%**
- intensity MAE: ~`1.07e-10`
- GPU kernel: 最近直接运行日志约 `0.059 s`；分辨率扫描 warm-up 后约 `0.024 s`

详见 `validation/geodesic_cpu_gpu_comparison.json`。

### CUDA 诊断

- CuPy: `14.0.1`
- PyPI CUDA runtime/NVRTC/ptxas wheel: 已安装
- CUDA device: `NVIDIA GeForce RTX 4060 Laptop GPU`
- driver: `566.07`, driver CUDA: `12.7`
- Windows system `nvcc`: not found（CuPy RawKernel 仍可用）
- WSL2: nvcc 12.0 + g++ 已安装

## Local Artifacts

核心产物见 `docs/reproduction.md` 与 `DELIVERY_STATUS.md`。

## External Demo Attempts

### geokerr

- 官方页面：https://faculty.washington.edu/agol/geokerr/index.html
- 本地源码：`research/repos/geokerr/geokerr_code/geokerr.f`
- status: **success**（Docker + gfortran）
- output: `research/repos/geokerr/geokerr_code/abgrid.out`
- parsed geodesics: `400`
- visualization: `research/reproduction/geokerr_abgrid_points.png`
- 交叉验证：`validation/geokerr_cross_validation.json`（87% 状态一致）

Docker 命令见历史记录；也可直接读取已落盘的 `abgrid.out`。

### Odyssey

- GitHub：https://github.com/hungyipu/Odyssey
- 本地源码：`research/repos/Odyssey`

**Docker 构建（失败）：**

- build log: `research/repos/Odyssey/build_attempt.log`
- failure: `cuda.h: No such file or directory`

**WSL2 构建（成功）：**

- 在 WSL2 Ubuntu + CUDA Toolkit 12.0 下 `make` 成功
- 已运行 128×128 thermal syn image
- 详见 `DELIVERY_STATUS.md`

## Interpretation

- 本地 pipeline 已包含 CPU / geodesic CPU / fast CUDA MVP / **完整 geodesic CUDA kernel（float32 + float64）** / benchmark / validation / 论文 / PPT。
- geokerr 提供外部 Kerr 参考；Odyssey 在 WSL2 可构建，Docker 路径仍受 CUDA 头文件限制。
- 下一步：geokerr 与本地 tracer 在相同 `r_obs` 下做更严格轨迹对比。
