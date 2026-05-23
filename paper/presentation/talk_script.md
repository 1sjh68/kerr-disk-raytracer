# 5 分钟展示讲稿

> 约 5 分钟（750–900 字），与 `paper/presentation/output.pptx` 8 张幻灯片一一对应。
> 每张幻灯片下方注明对应秒数，便于现场把控时间。

---

## 幻灯 1 — Title / Snapshot（约 30 秒，0:00–0:30）

> 大家好。我汇报的是一个 Kerr 黑洞薄盘的广义相对论光线追踪项目。

> 项目目标是构建一条**可复现、可验证、可加速**的 GRRT 流水线：CPU 端拿到高精度物理基准，GPU 端拿到加速比，并用外部代码做交叉验证。最终交付包含一份论文式技术报告、参数扫描矩阵、两套渲染产物，以及完整的测试和可复现脚本。

> 屏幕左侧是 fast 预览管线产出的最终图像，右侧是完整 Hamiltonian geodesic CUDA kernel 的 float64 参考图——这两张图来自同一份配置，但走的是两条独立的代码路径。

---

## 幻灯 2 — Research Baseline（约 30 秒，0:30–1:00）

> 项目起步于一份文献和开源项目调研。我对照阅读了 Luminet 1979、Bardeen / Cunningham 系列、Page–Thorne 薄盘模型，以及 geokerr、YNOGK、Odyssey、grtrans、ipole 五个公开实现。

> 实际跑通的有两个：geokerr 的 Fortran 半解析参考，以及 Odyssey 的 CUDA GRRT。Odyssey 是在 WSL2 里成功构建的。这两个项目分别给我一条**轨迹级**和一条**像素级**的对照基线。

---

## 幻灯 3 — Physical Model（约 45 秒，1:00–1:45）

> 物理上，我用 Boyer-Lindquist 坐标下的 Kerr 度规，自然单位 G=c=M=1。事件视界、ISCO、Keplerian 频率走 Bardeen-Press-Teukolsky 解析公式。

> 光线积分用 8 维相空间的 **Hamiltonian 形式**——这条路线的好处是 CPU 和 GPU 可以共用一份 RHS。我没用 Carter 常数做主路径，而是把它作为后续高精度参考。RHS 里所有逆度规项和 r/θ 偏导都是解析展开，避免 4×4 矩阵分配。

> 渲染端是赤道面 thin disk，发射模型支持 power-law 和 Novikov-Thorne 两种；红移因子取自 Killing 矢量与光子动量，最终观测强度走 g³ 变换。

---

## 幻灯 4 — Dual Pipeline（约 45 秒，1:45–2:30）

> 系统是双管线设计。**第一条**是 fast screen-space MVP——解析近似阴影 + Doppler，能跑实时预览，用于快速参数调试。**第二条**是完整逐像素 Hamiltonian 测地线积分，固定步长 RK4，一个像素一条线程，是科学参考。

> 后者在 `cuda/kernels.cu` 里有 float32 和 float64 两个版本。float64 kernel 内部全部用 double，I/O 边界做 float↔double 转换，以兼容 CuPy RawKernel 的接口约定。

---

## 幻灯 5 — Rendering Outputs（约 30 秒，2:30–3:00）

> 这页展示渲染产物：fast 渲染、float32 geodesic、float64 geodesic、最终调色。下方是 24 组 6 自旋 × 4 倾角的扫描缩略图。可以看到自旋升高 ISCO 缩小、阴影变形；倾角抬高 Doppler 不对称增强。

---

## 幻灯 6 — Validation（约 60 秒，3:00–4:00）

> 验证是这个项目的核心。三层证据：

> **第一层** pytest 19 通过，覆盖度规对称性、null 约束、视界、ISCO、红移正定性、I/O 安全。

> **第二层** CPU/GPU 一致性。在 a=0.7、i=60° 配置下，48×48 分辨率，**float64 kernel 与 CPU 参考的 status 匹配率是 99.96%**，disk-hit 分类完全一致。intensity MAE 在 1e-10 量级。

> **第三层** geokerr 外部交叉验证。400 条 ray，原始 abgrid 状态一致率 87%；加严格的 captured/disk 判定后，总体一致率 91.25%、disk 一致率 98.27%。剩余分歧主要落在光子球附近的临界光线，这是不同积分方法的固有差异。

---

## 幻灯 7 — Performance（约 50 秒，4:00–4:50）

> 性能上，RTX 4060 Laptop 8 GB。float32 kernel 在 256×256 下 4 ms，可做实时；float64 kernel 在 256×256 下 114 ms、512×512 下 0.5 s、1024×1024 下 3.5 s。CPU 端等效 48×48 float64 约 28–35 秒，**等效加速比约 1,200×–1,400×**。

> Profiling 上，ptxas 静态分析显示 kernel 用 82 寄存器、零 spill；动态 profiling 用 host Windows Nsight Compute 2024.3.2 完成。WSL 路径在当前 driver 下不通，已在文档里记录失败矩阵和 NVIDIA 官方建议的 driver 升级版本。

---

## 幻灯 8 — Limitations & Extensions（约 30 秒，4:50–5:00）

> 限制：geokerr 坐标级完全对齐和 Carter 常数半解析对照仍未做完；GPU 端目前是固定步长，自适应步长会引入 warp divergence；颜色映射是近似 sRGB，CIE1931 严格版本是后续扩展。

> 可选博士级方向有：偏振平行传播 + Stokes、GRMHD ingestion 与 EHT-style 指标对比、多 GPU 分块、可微 ray tracing、神经网络 surrogate。这些目前在 `DELIVERY_STATUS.md` 里都标为 0%，是项目的下一阶段。

> 谢谢。

---

## 时间预算检查

| Slide | 时长 | 累计 |
|---|---|---|
| 1 Title | 0:30 | 0:30 |
| 2 Research baseline | 0:30 | 1:00 |
| 3 Physical model | 0:45 | 1:45 |
| 4 Dual pipeline | 0:45 | 2:30 |
| 5 Rendering | 0:30 | 3:00 |
| 6 Validation | 1:00 | 4:00 |
| 7 Performance | 0:50 | 4:50 |
| 8 Limits | 0:30 | 5:00 |

总计 5:00。如果时间紧，可优先压缩 Slide 5（30 秒可掉到 15 秒）和 Slide 2（0:30 可压到 0:20，跳过具体 5 个开源项目名字）。

## 提问预案

| 可能被问 | 预答 |
|---|---|
| 为什么不用 Carter 常数？ | Hamiltonian 形式 CPU/GPU 同构，调试和验证一套代码；Carter 常数后续作高精度交叉验证 |
| float64 在 GeForce 上是 1/64 throughput，为什么倍率只有 17–29×？ | kernel 不是纯 ALU bound，warp divergence + 控制流 + 初始化都是 precision-agnostic |
| CPU/GPU 没 100% 匹配的 0.04% 是什么？ | 1 个像素在 captured/escaped 边界翻面，光子球附近临界光线，浮点路径敏感 |
| 为什么不和 EHT 真实图像比？ | thin disk + analytic emission 不是 EHT 的物理模型；EHT 用 GRMHD + 完整 GRRT，本项目是数值算法验证基线，不是观测拟合 |
| 后续如果做 EHT 风格要换什么？ | (1) 偏振平行传播；(2) GRMHD ingestion；(3) ring asymmetry / closure-like 指标；详见 `research/research_gap.md` |
