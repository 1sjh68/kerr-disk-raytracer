# Kerr 黑洞吸积盘 GPU 光线追踪研究空白与项目定位

更新时间：2026-05-21（文档对齐）

> **实施状态（2026-05-21）**：下文 P0–P2 路线图已基本完成（含完整 geodesic CUDA kernel、float64 99.96% 状态匹配、参数扫描、论文/PPT）。P3 博士级扩展仍未开始。本文档保留原始研究定位与风险分析供引用。

## 1. 核心定位

本项目的合理定位不是“再写一个完整 GRRT 生产代码”，而是：

> 构建一个从 CPU 高精度基准到 CUDA/GPU 并行实现的可复现 Kerr thin-disk ray tracer，并用明确的误差指标证明 GPU 图像与 CPU 参考一致。

这个定位夹在两个现有生态之间：

- 上游：geokerr/YNOGK 等半解析 geodesic 代码，精度高但不直接服务教学型 GPU 管线。
- 下游：grtrans/ipole/GYOTO 等科学 GRRT 框架，功能强但依赖重、学习曲线陡、GPU 主线不统一。

本项目的创新点应放在“透明、可验证、CPU/GPU 同构、报告级可复现”。

## 2. 已有工作的不足

### 2.1 geokerr / YNOGK

优点：

- Kerr photon orbit 计算精度高。
- 对 turning point 和 elliptic integrals 的处理成熟。

不足：

- Fortran 代码和老式输入输出对现代教学/工程复现不友好。
- 不是 GPU-first 结构。
- 不直接输出本项目需要的 `intensity/redshift/hit_mask/rgb` 统一数据包。

本项目补位：

- 保留半解析结果作为 validation reference。
- 用 Python/CUDA 重新组织数据结构和测试入口。

### 2.2 Odyssey

优点：

- 公开 GPU-based GRRT。
- CUDA kernel 组织和任务分层非常接近本项目目标。

不足：

- GPL-3.0 限制直接复用。
- 项目较旧，现代 Python 工程、CI、图像误差验证和报告管线不足。
- 自适应 RK 在 GPU 上容易产生 warp divergence，初学者很难证明 CPU/GPU 一致。

本项目补位：

- 先实现 fixed-step RK4 GPU MVP，控制误差来源。
- 在 `results/cpu_gpu_comparison.json` 中明确记录误差、运行时间和设备信息。

### 2.3 grtrans / ipole

优点：

- 科学问题完整，尤其是偏振 GRRT。
- 有 EHT 相关应用语境。

不足：

- 物理模型和依赖复杂，不适合作为从零实现的第一目标。
- 重点是 radiative transfer，不是 CPU/GPU 数值一致性教学。

本项目补位：

- 以 optically thick thin disk 为第一物理对象。
- 先把 geodesic、disk hit、redshift 和 image maps 做可验证，再扩展偏振。

### 2.4 GYOTO

优点：

- 框架成熟，可扩展 metric/object。
- 文档和示例相对完整。

不足：

- 构建链、插件机制和 FITS 生态较重。
- 不是一个轻量 GPU ray tracer 教学模板。

本项目补位：

- 输出更轻量的 `run_cpu.py`、`run_gpu.py`、`benchmark.py`、`validate.py`。

## 3. 本项目的可发表/可汇报贡献点

### 3.1 CPU/GPU 同构验证管线

贡献：

- 同一配置驱动 CPU 和 GPU。
- 输出统一数组：`intensity`、`redshift`、`hit_mask`、`rgb`。
- 自动生成 MSE、MAE、最大误差、相对误差和误差热力图。

价值：

- 比单纯展示漂亮图片更有工程可信度。
- 直接回应“GPU 加速是否改变物理结果”。

### 3.2 从 Schwarzschild 到 Kerr 的分层测试

贡献：

- `a=0` Schwarzschild 极限作为第一门。
- `a=0.5` 和 `a=0.9` 验证 frame dragging 与 ISCO 变化。
- `i=20 deg` 与 `i=80 deg` 验证倾角对 lensing 和 Doppler asymmetry 的影响。

价值：

- 测试不是随机参数扫描，而是围绕物理退化极限和高风险区域组织。

### 3.3 Thin disk 物理和渲染分离

贡献：

- 发射模型接口支持 `power_law` 与 `novikov_thorne`。
- 渲染后处理独立：tone mapping、gamma、bloom、temperature color map。

价值：

- 避免把数值积分误差和审美调色混在一起。

### 3.4 GPU divergence 的可测量讨论

贡献：

- baseline 使用固定步长 RK4，保证线程路径更一致。
- optimized 版本再加入 early termination、block size sweep、constant memory。
- 报告中记录速度与误差 tradeoff。

价值：

- 比直接使用 adaptive RK45 更容易解释 GPU 行为。

## 4. 不应声称的内容

为了保持报告严谨，本项目不能声称：

- 已经复现 EHT M87* 或 Sgr A* 的真实观测图像。
- thin disk 模型足以代表真实低辐射效率吸积流。
- 只凭最终图片即可证明 geodesic 正确。
- CUDA 版本如果没有和 CPU map 做误差比较就可称为物理可靠。
- 直接达到 geokerr/YNOGK 的半解析精度，除非完成逐轨迹交叉验证。

## 5. 最小可交付 MVP

MVP 必须包含以下证据链：

1. `docs/physics_model.md`：公式和单位系统。
2. `src/metric.py`：Kerr metric、horizon、ISCO。
3. `src/geodesic.py` + `src/integrators.py`：单光线积分和事件检测。
4. `run_cpu.py`：生成 `output/cpu_image.png` 和 `logs/cpu_run.json`。
5. `src/gpu_trace.py` + `cuda/kernels.cu`：CuPy RawKernel GPU 接口，并保留明确的 CPU fallback。
6. `validate.py`：生成 `validation/error_summary.md`。
7. `results/cpu_gpu_comparison.json`：CPU/GPU 对比，若无 CUDA 则退化为 CPU/fallback 对比。
8. `paper/main.pdf` 或可构建报告源：说明方法、结果、限制。

当前本机 CUDA runtime 已可用，`run_gpu.py` 走 CuPy RawKernel；如果换到无 CUDA 环境，仍可完成 CPU/fallback 验证，但报告必须把状态写清楚。

## 6. 关键科学风险与控制

| 风险 | 影响 | 控制方式 |
|---|---|---|
| 初始相机 tetrad 写错 | 图像整体错误但肉眼难发现 | 单光线 null constraint 和 Schwarzschild 对称性测试 |
| 固定步长穿过盘面但漏检 | hit mask 错误 | 用符号变化 + 线性插值/二分 event refinement |
| 靠图片验收 | 错误不易定位 | 保存 raw maps 和 JSON 误差 |
| GPU adaptive step divergence | 性能不可预测 | MVP 固定步长，优化阶段再分桶/early termination |
| 薄盘模型被误解成真实 EHT 模型 | 报告科学性下降 | 明确 thin disk vs GRMHD/GRRT 限制 |
| 许可证污染 | 交付不可发布 | 外部 GPL 代码只审计和黑箱验证，不复制 |

## 7. 后续路线图

### P0：当前应完成

- 完成 `research/literature_review.md`、`research/github_repo_audit.md`、`research/research_gap.md`。
- 搭建项目目录和 README。
- 写 CPU MVP：低分辨率 Kerr thin disk render。
- 写验证脚本：metric、ISCO、redshift、image map error。

### P1：GPU 最小闭环

- 写 `cuda/kernels.cu` 和 `src/gpu_trace.py`。
- 若有 CUDA：运行真实 GPU。
- 若无 CUDA：用相同输出格式的 CPU fallback 维持验证链。

### P2：论文级结果

- 参数扫描：spin、inclination、resolution、block size。
- 生成 final render、comparison grid、speedup chart。
- 写报告 PDF 和 8-10 页 PPT。

### P3：博士级扩展

- 偏振 parallel transport。
- Stokes 参数。
- synchrotron emission/absorption。
- GRMHD 数据读入。
- differentiable ray tracing。

## 8. 项目一句话创新点

本项目的创新点是把 Kerr 黑洞薄盘 ray tracing 从“能画图”推进到“有 CPU 高精度基准、有 GPU 同构输出、有误差热力图、有性能数据、有复现实验脚本”的完整工程化闭环。
