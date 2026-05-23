# 物理模型

## 单位

所有方程使用几何化自然单位：

$$
G = c = M = 1
$$

自旋参数 $a$ 无量纲，约束 $\lvert a \rvert < 1$。

## 坐标系

实现使用 Boyer–Lindquist 坐标：

$$
x^{\mu} = (t,\, r,\, \theta,\, \phi)
$$

度规符号约定为 $(-, +, +, +)$。

## Kerr 度规

辅助量：

$$
\Sigma = r^{2} + a^{2} \cos^{2}\theta,\qquad
\Delta = r^{2} - 2r + a^{2}.
$$

非零协变分量：

$$
\begin{aligned}
g_{tt} &= -\left(1 - \frac{2r}{\Sigma}\right), \\
g_{t\phi} &= -\frac{2 a r \sin^{2}\theta}{\Sigma}, \\
g_{rr} &= \frac{\Sigma}{\Delta}, \\
g_{\theta\theta} &= \Sigma, \\
g_{\phi\phi} &= \left(r^{2} + a^{2} + \frac{2 a^{2} r \sin^{2}\theta}{\Sigma}\right) \sin^{2}\theta.
\end{aligned}
$$

## 视界

外视界半径：

$$
r_{+} = 1 + \sqrt{1 - a^{2}}.
$$

## ISCO（最内稳定圆轨道）

代码通过 `src.metric.isco_radius` 实现标准的 Bardeen–Press–Teukolsky 公式（顺/逆轨道分别处理）。

## 零光子约束

对光子协变 4-动量 $p_{\mu}$：

$$
H = \tfrac{1}{2}\, g^{\mu\nu}\, p_{\mu}\, p_{\nu} = 0.
$$

测地线右端项采用 Hamiltonian 形式：

$$
\frac{\mathrm{d} x^{\mu}}{\mathrm{d}\lambda} = \frac{\partial H}{\partial p_{\mu}} = g^{\mu\nu}\, p_{\nu},\qquad
\frac{\mathrm{d} p_{\mu}}{\mathrm{d}\lambda} = -\frac{\partial H}{\partial x^{\mu}}.
$$

由于 Kerr 时空稳态轴对称，能量 $E = -p_{t}$ 与轴向角动量 $L_{z} = p_{\phi}$ 沿测地线守恒：

$$
p_{t} = \mathrm{const},\qquad p_{\phi} = \mathrm{const}.
$$

Carter 常数路线作为另一条独立参考列在 [`equation_reference.md`](equation_reference.md) 第 4 节。

## 薄盘

MVP 盘是光学厚的赤道平面：

$$
\theta = \pi/2,\qquad r_{\mathrm{in}} \le r \le r_{\mathrm{out}}.
$$

默认内边界 $r_{\mathrm{in}} = r_{\mathrm{ISCO}}$，默认外边界 $r_{\mathrm{out}} = 28\,M$（可在 `configs/default.yaml` 调）。

## 发射模型

默认局部发射模型为幂律：

$$
I_{\mathrm{em}}(r) = r^{-q}.
$$

也可选近似 Novikov–Thorne 通量：

$$
F(r) \propto r^{-3} \left(1 - \sqrt{r_{\mathrm{in}} / r}\right).
$$

## 红移与观测亮度

对 Keplerian 发射体（角速度 $\Omega$）：

$$
\Omega = \frac{1}{r^{3/2} + a},\qquad
u^{t} = \frac{1}{\sqrt{-(g_{tt} + 2\Omega\, g_{t\phi} + \Omega^{2}\, g_{\phi\phi})}}.
$$

红移因子：

$$
g = \frac{\nu_{\mathrm{obs}}}{\nu_{\mathrm{em}}} = \frac{1}{u^{t} (1 - \Omega\, \lambda)},
$$

其中 $\lambda = p_{\phi} / E$，$E = -p_{t}$。观测强度变换：

$$
I_{\mathrm{obs}} = g^{3}\, I_{\mathrm{em}}.
$$

## 当前实现状态

代码渲染流水线有三条路径：

- **`render_thin_disk_fast`**：快速向量化屏幕空间薄盘模型，常规渲染与参数扫描使用。
- **`render_thin_disk_geodesic_cpu`**：低分辨率逐像素 Hamiltonian 测地线 CPU 参考。
- **`render_cuda_geodesic`**：CuPy RawKernel 路径，源自 `cuda/kernels.cu`，`float32` 用于预览，`float64` 用于科学参考。可选 `fast_math=True` 启用 `--use_fast_math` 编译的 module（≥192² 稳定 ~3× 加速，accuracy 100%）。

主要遗留方向不在 GPU 可用性——完整测地线 CUDA 路径已实现并在 48×48 验证。剩余工作集中在更高保真度模型与优化：

- Carter 常数半解析参考（推导见 `equation_reference.md`）
- GPU 自适应步长（CPU 端 RK45 已实现，详见 `validation/rk45_vs_rk4_demo.md`）
- 完整偏振输运与 Faraday rotation（当前 stub 见 `polarization.md`）
- GRMHD 流场 ingestion（接口 stub 见 `src/grmhd_io.py`）
- geokerr 严格相机约定下的坐标级对齐（已做轻量版，见 `validation/geokerr_coordinate_alignment.md`）
