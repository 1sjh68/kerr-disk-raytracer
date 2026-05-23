# 安装

## 本地 Python（CPU）

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows
# source .venv/bin/activate       # Linux / macOS
python -m pip install -r requirements.txt
python -m pytest tests            # 应得 47 passed
```

CPU 路径依赖：NumPy、Matplotlib、pytest、PyYAML、Pillow（GIF 合成）。

## CUDA Python 环境

推荐 Python 3.12 虚拟环境，加装 CUDA 依赖：

```powershell
uv venv .venv --python 3.12
.\.venv\Scripts\python.exe -m ensurepip --upgrade
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt -r requirements-cuda.txt
.\.venv\Scripts\python.exe scripts\check_cuda.py
```

`scripts/check_cuda.py` 写入 `logs/cuda_status.json`，检查：

- CuPy 导入与 CUDA 设备数量
- `nvidia-smi` 输出
- 系统 `nvcc --version`（Windows 上常不可用）
- PyPI 捆绑的 CUDA wheel 路径
- 捆绑的 `ptxas --version`（安装 `nvidia-cuda-nvcc-cu12` 后可用）
- 项目 CUDA 入口的 backend reason 字符串

机器需有 NVIDIA 驱动和至少一块可见 CUDA GPU，`run_gpu.py` / `run_geodesic_gpu.py` 才会走 RawKernel；否则记录 `backend=cpu_fallback` 并保持输出格式可复现。

## Geodesic GPU 入口

完整逐像素 Hamiltonian geodesic 使用独立 kernel：

```powershell
.\.venv\Scripts\python.exe run_geodesic_gpu.py --precision float64
.\.venv\Scripts\python.exe validate_geodesic.py
```

- `--precision float32`：快速预览（256×256 约 4 ms kernel；启用 `fast_math` 可再快 ~3×）
- `--precision float64`：科学参考（48×48 与 CPU 99.96% 状态匹配，disk-hit 分类 100%）

`run_gpu.py` 仍服务 **fast thin-disk MVP**；geodesic 路径走 `run_geodesic_gpu.py`。

## fast_math 加速（可选）

```python
from src.gpu_trace import render_cuda_geodesic
data = render_cuda_geodesic(cfg, resolution=512, precision="float32", fast_math=True)
```

详细基准与 accuracy/speed tradeoff：`results/fastmath_optimization.md`、`tools/benchmark_fastmath.py`。

## Docker

```bash
docker build -t kerr-disk-raytracer .
docker run --rm -v "$PWD/output:/app/output" kerr-disk-raytracer
```

## WSL2 / 系统 nvcc（可选）

- Windows 宿主机 `nvcc` 通常不在 PATH；CuPy RawKernel 编译依赖 PyPI wheels 即可。
- 完整 CUDA Toolkit、`ncu` profiling、Odyssey 构建可在 WSL2 Ubuntu（nvcc 12.0）中进行。
- 当前 host driver 566.07 (CUDA 12.7) 下 WSL2 ncu 路径不可行（详见 `results/wsl_profile_report.md`）；推荐使用 host Windows 路径：`tools/run_ncu_pipeline.ps1`。
- Nsight Compute 权限：见 `tools/enable_gpu_profiling.ps1` / `enable_gpu_profiling.reg`。

## 完整 Nsight Compute Profile（host Windows，需一次 UAC）

```powershell
powershell -File 'tools\run_ncu_pipeline.ps1'
# 弹 UAC 点 [是] -> 5–15 分钟自动跑 4 个 --set full
# 输出 results/ncu_*.{ncu-rep,summary.txt,csv}
```

## PPT 构建（可选）

一键脚本末尾调用 Node 构建 PPT：

```powershell
node paper/presentation/build/build_deck.mjs
```

需先在 `paper/presentation/build/` 执行 `npm install`。
