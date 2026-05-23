# 复现指南

更新时间：2026-05-24

## 一键运行

**Windows**（推荐，含 float32/float64 geodesic + 8 个本轮新增 demo）：

```powershell
.\scripts\run_all.ps1
```

**Linux / WSL**：

```bash
./scripts/run_all.sh
```

脚本优先使用本地 `.venv`；可通过环境变量 `PYTHON` 覆盖解释器。本轮新增的 8 个 demo 用 `try/catch`（PowerShell）或 `soft_run`（bash）包裹，失败不会终止主流水线。

## `scripts/run_all.*` 步骤一览

```text
# 核心 CPU/GPU 渲染
run_cpu.py
run_geodesic_cpu.py
scripts/check_cuda.py
run_gpu.py
run_geodesic_gpu.py --precision float32
run_geodesic_gpu.py --precision float64
render.py

# 基准 + 验证
benchmark.py
scripts/run_experiments.py
validate.py
validate_geodesic.py
scripts/summarize_external_demos.py
scripts/parse_geokerr.py
scripts/make_report.py                       # 生成 paper/main.pdf

# 本轮新增（2026-05 收尾，soft-fail 包裹）
tools/benchmark_fastmath.py                  # T6: --use_fast_math 加速比
tools/disk_param_sweep.py                    # T3: r_outer + emissivity 扫描
tools/render_color_compare.py                # T4: CIE1931 vs approx 颜色对比
tools/geokerr_coordinate_compare.py          # T7: 坐标级对齐
tools/rk45_vs_rk4_demo.py                    # T7 follow-up
tools/eht_metrics_demo.py                    # T11: ring diameter / asymmetry
tools/polarization_demo.py                   # T10: Stokes I/Q/U + EVPA
tools/compose_animations.py                  # T12: 24 帧扫描合成 GIF

# 测试 + PPT
pytest tests -q
node paper/presentation/build/build_deck.mjs # 需 Node.js
```

## 手动逐步运行

```bash
# 必跑（< 5 分钟）
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests -q       # 应得 47 passed
python run_cpu.py                            # ~5 s, 写 reference/cpu_reference{,_256}.npz
python run_geodesic_cpu.py                   # ~30 s, 写 reference/cpu_geodesic_reference.npz

# GPU 加速（需 CuPy + RTX/CUDA）
python scripts/check_cuda.py                 # 输出 CuPy / CUDA 版本与设备信息
python run_gpu.py
python run_geodesic_gpu.py --precision float64
python validate.py                           # CPU/GPU 一致性
python validate_geodesic.py                  # geodesic CPU/GPU 一致性
```

## 预期产物

### 图像

- `output/cpu_image.png`、`output/gpu_image.png`
- `output/cpu_geodesic_image.png`
- `output/gpu_geodesic_image_float32.png`、`output/gpu_geodesic_image_float64.png`
- `figures/geodesic_reference.png`、`figures/final_render.png`
- `figures/gpu_geodesic_reference_float32.png`、`figures/gpu_geodesic_reference_float64.png`

### 参考数组（`.npz` 包含 intensity / redshift / temperature / hit_mask / rgb / status_code / null_error 等键）

- `reference/cpu_reference.npz`、`reference/cpu_reference_256.npz`
- `reference/cpu_geodesic_reference.npz`
- `reference/gpu_geodesic_reference_float32.npz`
- `reference/gpu_geodesic_reference_float64.npz`

> 历史版本同时保存了同名的 `.npy` 单键副本（与 `.npz` 重复），现已停止生成并加入 `.gitignore`。
> 取单个键直接 `np.load("reference/cpu_reference.npz")["intensity"]`。

### 日志与验证

- `logs/cuda_status.json`、`logs/cpu_run.json`、`logs/gpu_run.json`
- `logs/gpu_geodesic_run_float32.json`、`logs/gpu_geodesic_run_float64.json`
- `results/cpu_gpu_comparison.json`、`results/error_map.png`、`results/comparison_grid.png`
- `validation/error_summary.md`、`validation/extended_validation.json`
- `validation/geodesic_cpu_gpu_comparison.json`、`validation/geokerr_cross_validation.json`
- `validation/geokerr_strict_comparison.json`

### 本轮新增产物（2026-05 收尾）

- `results/fastmath_benchmark.json` + `figures/fastmath_speedup.png`（T6）
- `results/disk_radius_sweep.json` + `results/disk_emissivity_sweep.json` + 2 张 sweep 对比图（T3）
- `figures/cie_vs_approx_comparison.png` + `figures/blackbody_locus.png`（T4）
- `validation/geokerr_coordinate_alignment.{json,md}` + `research/reproduction/geokerr_coordinate_alignment.png`（T7）
- `validation/rk45_vs_rk4_demo.{json,md}` + `research/reproduction/rk45_vs_rk4.png`（T7）
- `results/eht_metrics.json` + `results/eht_metrics_report.md`（T11）
- `results/polarization_demo.json` + `figures/polarization_stokes_qu.png` + `figures/polarization_evpa_quiver.png`（T10）
- `figures/spin_sweep_animation.gif`、`figures/inclination_sweep_animation.gif`、`figures/full_sweep_animation.gif`（T12）

### 论文与汇报

- `paper/main.md`（中文 + LaTeX 公式）
- `paper/main.pdf`（由 `scripts/make_report.py` 拼接，含中文字体 fallback）
- `paper/presentation/output.pptx`、`paper/presentation/talk_script.md`

### 完整 Nsight Compute Profile（需 Windows admin + 一次 UAC）

```powershell
powershell -File 'tools\run_ncu_pipeline.ps1'
# 弹 UAC 点 [是] -> 5–15 分钟自动跑 4 个 --set full
# 输出 results/ncu_*.{ncu-rep,summary.txt,csv}
```

完整清单与验证数字见仓库根目录 [`DELIVERY_STATUS.md`](../DELIVERY_STATUS.md) 与 [`SUBMISSION_CHECKLIST.md`](../SUBMISSION_CHECKLIST.md)。

## 已知限制

- CUDA 不可用时 `run_gpu.py` / `run_geodesic_gpu.py` 自动回退 CPU，记录在对应 `logs/*.json`。
- geokerr Docker demo 已成功；Odyssey Docker 因缺 CUDA 头文件失败，WSL2 + CUDA Toolkit 下可成功（见 `research/reproduction_log.md`）。
- 参数扫描图（24 张）由 `tools/parameter_sweep.py` 生成，路径记录在 `results/parameter_sweep.json`。
- WSL2 ncu 路径在当前 driver 566.07（CUDA 12.7）下不可行，详见 `results/wsl_profile_report.md`；Windows host 路径已就绪。

## 外部 demo

```bash
python scripts/summarize_external_demos.py   # 汇总 geokerr + Odyssey 可用性
python scripts/parse_geokerr.py              # 解析 abgrid.out 并交叉验证
```

详情：`validation/external_cross_validation.md`、`validation/geokerr_strict_comparison.json`、`validation/geokerr_coordinate_alignment.md`。
