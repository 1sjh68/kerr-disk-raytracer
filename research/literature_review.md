# Kerr 黑洞吸积盘光线追踪文献综述

更新时间：2026-05-21

## 1. 综述结论

本项目的技术路线应拆成三层，而不是一开始直接写 CUDA：

1. **物理基准层**：先用 Boyer-Lindquist 坐标下的 Kerr 度规、事件视界、ISCO、null geodesic 约束、薄盘发射和红移因子建立 CPU 高精度参考。
2. **数值验证层**：用 Schwarzschild 极限、已知 Kerr geodesic 常数、步长收敛、单光线轨迹、CPU/GPU map 误差和文献项目交叉验证来约束正确性。
3. **并行渲染层**：GPU 版本优先实现“一像素一线程”的固定步长或分段固定步长 RK4/RK5 最小版本，再讨论自适应步长带来的 warp divergence。

现有成熟代码的共同教训是：**ray tracing 和 radiative transfer 是两个不同难度层级**。本项目第一阶段应以 optically thick thin disk 成像为核心，保留接口让后续扩展到 polarized GRRT，而不是一开始复刻 grtrans/ipole 的完整偏振输运。

## 2. 经典黑洞成像

### Luminet 1979

Luminet 的 A&A 论文是黑洞吸积盘模拟图像的经典起点，主题是 Schwarzschild 黑洞周围薄吸积盘的视觉外观。该工作给出了三个对本项目仍然有效的验收特征：

- 黑洞阴影必须由被捕获光线形成，而不是人为画黑圆。
- 倾斜薄盘应显示由引力透镜造成的上方/下方弯折像。
- 盘面亮度需要同时包含引力红移、Doppler shift 和观测强度变换。

本项目可将 Luminet 图像作为 Schwarzschild/薄盘渲染的视觉回归基准，但不能把它作为 Kerr 高自旋场景的完整验证。

关键来源：
- Jean-Pierre Luminet, "Image of a spherical black hole with thin accretion disk", Astronomy and Astrophysics 75, 228-235 (1979): https://ui.adsabs.harvard.edu/abs/1979A%26A....75..228L/abstract

### Bardeen, Cunningham 与 Kerr 光线传播

Bardeen、Cunningham 等早期工作建立了 Kerr 光线传播、观察者屏幕坐标、常数运动量和极端 Kerr 光学外观的基本框架。对本项目最重要的是两类结果：

- 用 Carter 常数和能量/角动量参数化 null geodesic，作为数值积分的独立校验。
- 将盘面发射映射到观测屏幕，形成 transfer function 思路。

这些文献建议本项目保留两套描述：

- **ODE/Hamiltonian 路线**：适合 CPU/GPU 同构实现和调试。
- **Carter 常数/半解析路线**：适合作为高精度参考和单光线交叉验证。

关键来源：
- Cunningham & Bardeen, "The optical appearance of a star orbiting an extreme Kerr black hole", ApJ 183, 237 (1973): https://articles.adsabs.harvard.edu/pdf/1973ApJ...183..237C
- Cunningham, "The effects of redshifts and focusing on the spectrum of an accretion disk around a Kerr black hole", ApJ 202, 788 (1975): https://adsabs.harvard.edu/pdf/1975ApJ...202..788C
- Bardeen, "Timelike and null geodesics in the Kerr metric", Les Houches lectures (1973): https://ui.adsabs.harvard.edu/abs/1973blho.conf..215B/abstract

## 3. 薄吸积盘模型

### Novikov-Thorne / Page-Thorne

Novikov-Thorne 薄盘模型和 Page-Thorne 的时均结构公式仍是 Kerr 薄盘基准。对本项目而言，需要优先实现以下最小集合：

- 单位系统：`G = c = M = 1`。
- 事件视界：`r_+ = 1 + sqrt(1 - a^2)`。
- prograde/retrograde ISCO 半径。
- 盘面内边界默认设为 ISCO，外边界可配置。
- 简化发射律 `I_emit(r) = r^-q` 先作为渲染主线，Novikov-Thorne flux/temperature 作为第二层物理模型。

重要边界条件：

- Novikov-Thorne 假设几何薄、光学厚、近似稳态、ISCO 处无 torque；这不适合直接解释厚盘、RIAF 或 EHT 真实源的全部辐射。
- EHT 风格图像通常依赖 GRMHD + GRRT；薄盘图像只能作为算法验证和可视化基线。

关键来源：
- Page & Thorne, "Disk-Accretion onto a Black Hole. Time-Averaged Structure of Accretion Disk", ApJ 191, 499-506 (1974): https://articles.adsabs.harvard.edu/pdf/1974ApJ...191..499P
- Thorne, "Disk-Accretion onto a Black Hole. II. Evolution of the Hole", ApJ 191, 507-519 (1974): https://ui.adsabs.harvard.edu/abs/1974ApJ...191..507T/abstract

## 4. 公共 Kerr geodesic 代码

### geokerr

Dexter & Agol 的 geokerr 是本项目最重要的半解析参考之一。它使用 Carlson elliptic integrals 计算 Kerr photon orbits，核心价值不是渲染效果，而是：

- 高精度 geodesic 参考；
- turning point 处理经验；
- Fortran 代码和示例输入可用于少量轨迹交叉验证。

本项目不应直接照搬 Fortran 结构，但应复刻其验证方式：固定屏幕点、输出常数和轨迹，和 CPU ODE 积分器逐点比较。

关键来源：
- Dexter & Agol, "A Fast New Public Code for Computing Photon Orbits in a Kerr Spacetime", ApJ 696, 1616 (2009): https://arxiv.org/abs/0903.0620
- geokerr release page: https://faculty.washington.edu/agol/geokerr/index.html

### YNOGK

YNOGK 继承 geokerr 思路，但用 Weierstrass/Jacobi elliptic functions 和参数 `p` 表示坐标与仿射参数。它的优势是 turning point 信息不必由用户预先指定，许多成像和 observer-emitter 问题可化为 root-finding。

本项目可借鉴：

- 将 geodesic 状态和 conserved quantities 明确分离；
- 把“求交点”建模为 root-finding 或 event detection，而不是只靠固定步长穿越检测；
- 用 toy problems 做单元测试，而不是只看最终图片。

关键来源：
- Yang & Wang, "YNOGK: A new public code for calculating null geodesics in the Kerr spacetime", ApJS 207, 6 (2013): https://arxiv.org/abs/1305.1250
- OSTI 摘要与 DOI 信息: https://www.osti.gov/biblio/22118709

## 5. GRRT 与 GPU 代码

### Odyssey

Odyssey 是公开的 GPU-based GRRT 代码，采用 CUDA C/C++，用于 Kerr 度规中的图像和光谱计算。它的 README 显示默认任务包括 Keplerian disk redshift 和 Keplerian shell image，并使用 GPU kernel 内的 ray update loop。

本项目可借鉴：

- `main -> task -> GPUcompute -> kernel -> output` 的工程分层；
- 一像素/一光线并行；
- GPU 端 early termination：进入黑洞、逃逸、命中盘面后停止；
- 常量参数和输出数组的结构化设计。

需要避免直接照搬：

- Odyssey 使用 GPL-3.0；若本项目希望更宽松发布，代码级复制会污染许可证。
- Odyssey 目标是 GRRT，依赖较多物理细节；本项目最小 GPU 版应先对齐 CPU thin disk 数据结构。

关键来源：
- Odyssey paper: https://arxiv.org/abs/1601.02063
- Odyssey GitHub: https://github.com/hungyipu/Odyssey

### grtrans

grtrans 是公开 Kerr metric polarized ray tracing radiative transfer code，使用 geokerr，并支持 thin disk、jet、HARM 等测试问题。它更接近科学生产代码，包含 polarized radiative transfer、OpenMP 并行和 Fortran/Python 调用层。

本项目可借鉴：

- 用 `geodata`、`fluiddata` 等 namelist 风格分离几何与流体参数；
- 输出 debug maps，例如 optical depth 和 ray-averaged quantities；
- 使用标准测试问题生成论文图。

但 grtrans 对入门项目偏重，初期不应把偏振输运作为主线。

关键来源：
- Dexter, "A public code for general relativistic, polarised radiative transfer around spinning black holes" (2016): https://arxiv.org/abs/1602.03184
- grtrans GitHub: https://github.com/jadexter/grtrans

### ipole

ipole 是 covariant polarized radiative transport code，服务于 EHT 类黑洞吸积系统成像。它适合后期扩展 Stokes `I,Q,U,V` 和偏振平行传播，不适合作为当前 thin-disk/GPU MVP 的起点。

关键来源：
- Moscibrodzka & Gammie, "ipole - semianalytic scheme for relativistic polarized radiative transport" (2018): https://arxiv.org/abs/1712.03057
- ipole GitHub: https://github.com/AFD-Illinois/ipole

## 6. EHT 观测论文与验证启发

EHT 的 M87* 和 Sgr A* 论文不能直接作为本项目像素级 benchmark，因为真实 EHT 成像包含 VLBI 采样、校准、成像算法、源时间变化和 GRMHD 模型库。但它们对"验证系统"有三点直接启发：

- 不只验证图像好看，还要验证可观测指标，例如 ring diameter、asymmetry、closure-like robust features。
- 需要区分"物理模型误差"和"数值积分误差"。
- 报告中应明确模型限制：thin disk render 不是 EHT 真实源拟合。

### 6.1 EHT 核心结论与本项目可对照的数字

| 指标 | M87* (2019) | Sgr A* (2022) | 本项目能否对照 |
|---|---|---|---|
| Shadow / ring 直径 | 42 ± 3 μas | 48.7 ± 7 μas | **能**（用 r_obs → ∞ + 黑洞质量换算） |
| 黑洞质量 | 6.5 ± 0.7 × 10^9 M_sun | 4.0(+1.1, -0.6) × 10^6 M_sun | 不直接需要（项目用 G=c=M=1 自然单位） |
| 距离 | 16.8 Mpc | 8.15 kpc | 同上 |
| Ring brightness asymmetry | ≳10:1（南北） | 强但变化快 | **能**（Doppler beaming 主导） |
| 倾角推断 | i ≳ 17° (jet 轴向偏离视线) | ~30° | **能**（已有 i = 10°/30°/60°/80° 扫描） |
| 自旋约束 | 弱，与多种 a 一致 | 弱 | **不能**（thin disk 不足以约束 a） |
| 频率 | 230 GHz, 4 GHz BW | 同上 | **不能**（项目是单频/灰体近似） |
| 时间变源 | 多年间稳定 | 数分钟尺度变化 | **不能**（项目稳态） |

EHT 给出的两条最可对照的指标是 **ring diameter** 和 **ring asymmetry**，因为这两条**对辐射模型相对不敏感**，主要受度规几何和倾角主导。本项目可以做的对照实验：

1. 渲染 a = 0.5 / 0.94（M87 GRMHD 偏好的高自旋）/ Schwarzschild 三组，固定 i = 17°，量化 ring 直径在不同自旋下的变化（应在 ~5 r_g 量级，对应 ~40 μas 配 6.5e9 M_sun）。
2. 用 ring 上 azimuthal flux 分布拟合 sinusoidal asymmetry 系数，对照 EHT 报告的 north–south brightness ratio。

这些对照不是用 thin disk 去复现 GRMHD/GRRT 结果，而是验证项目的**度规几何正确**——shadow 大小、形状、视差应与 Kerr 解析预言一致。

### 6.2 本项目的范畴边界

| 维度 | 本项目（thin disk Kerr ray-tracing） | EHT GRMHD + GRRT |
|---|---|---|
| 几何模型 | 赤道面 thin disk，单光线一次命中即停 | 厚 RIAF / MAD / SANE 流体，多次散射 |
| 流体动力学 | Keplerian 速度场（解析） | full GRMHD，含磁场、压力、加热/冷却 |
| 辐射机制 | 灰体功率律 / Novikov-Thorne 通量 | thermal + non-thermal synchrotron + 自吸收 |
| 偏振 | 无 | full polarized transport (Stokes I,Q,U,V) |
| 频率 | 单频 / 灰体 | 多频，含吸收系数 |
| 时间 | 稳态 | 动态（snapshot, time-averaged） |
| 观测拟合 | 图像几何特征 | full closure phases / amplitudes / Stokes |
| 验证目标 | 数值算法正确性 + 性能 | 物理参数推断（M, a, i, accretion rate） |

**本项目是数值算法验证基线，不是观测拟合工具。** 这条边界要在论文 limitations 一节明确：

> 本项目的薄盘 ray-tracing 结果在度规几何层面（shadow size、shape、light bending）与 GRMHD/GRRT 在同等 inclination 下应当一致，但在辐射强度分布、频谱、偏振结构上，由于发射模型和流体动力学的简化，**不能用作对 EHT 观测数据的物理参数推断**。要做参数推断，需要至少补充以下三项之一：(a) GRMHD ingestion + 完整 transfer function；(b) 偏振 parallel transport；(c) 多频带辐射系数。

### 6.3 关键来源

- EHT Collaboration, "First M87 Event Horizon Telescope Results. I. The Shadow of the Supermassive Black Hole", ApJL 875, L1 (2019): https://arxiv.org/abs/1906.11238
- EHT Collaboration, "First M87 Event Horizon Telescope Results. IV. Imaging the Central Supermassive Black Hole", ApJL 875, L4 (2019): https://arxiv.org/abs/1906.11241
- EHT Collaboration, "First M87 Event Horizon Telescope Results. VI. The Shadow and Mass of the Central Black Hole", ApJL 875, L6 (2019): https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...6E
- EHT Collaboration, "First Sagittarius A* Event Horizon Telescope Results. I", ApJL 930, L12 (2022): https://eventhorizontelescope.org/publications/first-sagittarius-event-horizon-telescope-results-i-the-shadow-of-the-supermassive-black-hole-in-the-center-of-the-milky-way
- EHT Collaboration, "M 87 persistent shadow", A&A 681, A79 (2024): https://ui.adsabs.harvard.edu/abs/2024A&A...681A..79E
- 本项目相关：`research/research_gap.md`、`docs/equation_reference.md` 第 4 节。

## 7. 本项目建议的阅读优先级

| 优先级 | 文献/代码 | 目的 | 本项目动作 |
|---|---|---|---|
| P0 | Luminet 1979 | Schwarzschild thin disk 视觉基准 | 生成 `a=0`、倾角 60-80 deg 的低分辨率图像 |
| P0 | Bardeen/Cunningham/Carter 常数路线 | Kerr null geodesic 参考 | 单光线 conserved quantities 测试 |
| P0 | Page-Thorne / Novikov-Thorne | ISCO 和薄盘发射 | 实现 ISCO、`r^-q`、NT flux 可选 |
| P0 | geokerr | 半解析轨迹参考 | 至少抽 3 条轨迹做交叉验证 |
| P1 | YNOGK | turning point 与 root-finding | event detection 设计参考 |
| P1 | Odyssey | CUDA 并行结构 | 设计 GPU 数据结构和 kernel layout |
| P2 | grtrans | 完整 GRRT 工程组织 | 后期配置与调试 maps 参考 |
| P2 | ipole | 偏振扩展 | 可选扩展工作 |
| P2 | EHT papers | 报告中的观测语境 | 指标讨论，不作像素级 truth |

## 8. 立即落地的技术选择

- CPU 基准：Python + NumPy，先实现 Kerr 度规、Hamiltonian RHS、RK4/RK45、事件检测和薄盘命中。
- 数值变量：Boyer-Lindquist `x = (t, r, theta, phi)` 与 covariant momentum `p = (p_t, p_r, p_theta, p_phi)`；固定 `p_t = -1`，用 null constraint 解 `p_r` 或用局部 tetrad 初始化。
- 渲染模型：第一版采用 optically thick equatorial disk，命中一次即停止；发射强度先用 `r^-q`，再加 Novikov-Thorne temperature。
- GPU MVP：CUDA kernel 输入相机/黑洞/盘参数，输出 `intensity`、`redshift`、`hit_mask`；先固定步长 RK4，后续再做 adaptive 分桶。
- 验证优先级：metric symmetry、horizon/ISCO、null constraint drift、single-ray regression、CPU/GPU array equality、图像误差热力图。

