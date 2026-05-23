# Research Questions

更新时间：2026-05-21（文档同步）

1. 如何在 Kerr spacetime 中稳定计算 null geodesic，并控制 null constraint drift？
2. 如何建立 CPU 高精度基准，使最终图像可追溯到 metric、相机初始化、盘面命中和红移计算？
3. 如何让 GPU 输出与 CPU 输出使用同一数据契约：`intensity`、`redshift`、`temperature`、`hit_mask`、`rgb`？
4. 如何在 GPU 上处理自适应步长造成的 warp divergence？
5. 如何把 Novikov-Thorne 薄盘、简化 emissivity law、Doppler beaming、gravitational redshift 和 tone mapping 分层实现？

## 当前回答

### Q1–Q3：已实现并验证

- CPU 侧：`src/geodesic.py` + `run_geodesic_cpu.py`，固定步长 RK4 Hamiltonian null geodesic。
- GPU 侧：`cuda/kernels.cu` 中 `kerr_geodesic_kernel`（float32）与 `kerr_geodesic_kernel_double`（float64），一像素一线程。
- 数据契约统一：`intensity`、`redshift`、`temperature`、`hit_mask`、`status_code`、`null_error`、`rgb`。
- **float64 @ 48×48**：CPU/GPU 状态匹配 **99.96%**，disk-hit 计数完全一致；intensity MAE ~1e-10（见 `validation/geodesic_cpu_gpu_comparison.json`）。
- float32 版状态匹配约 **99.91%**，仍有少量 captured/escaped/invalid 边界分歧；科学参考路径应使用 float64。

### Q4：刻意延后

- MVP 采用 **固定步长 RK4**，避免 adaptive RK45 在 GPU 上的 warp divergence。
- `src/integrators.py` 保留 RK45 供 CPU 单光线实验；GPU adaptive 步长列为 Future Work。

### Q5：分层已实现

- 物理层：`src/disk.py`（Novikov–Thorne、power-law emissivity、Doppler、ISCO 内边界）。
- 追踪层：geodesic 命中 + 红移因子 `g`，亮度 `I_obs = g³ I_emit`。
- 显示层：`temperature_to_rgb` + Reinhard tone mapping + 可选 bloom（`src/render.py`）。

### 仍开放

- Carter 常数半解析路径作为独立高精度对照。
- geokerr 轨迹级严格对齐（原始 abgrid 87% 状态一致；a=0.7/i=60 严格状态判定 91.25% 总体一致、98.27% disk 一致）。
- GPU adaptive 步长 / 分桶策略。
