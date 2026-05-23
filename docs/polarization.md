# Polarization Pipeline (Phase 10 — Minimum Viable)

更新时间：2026-05-23

## 摘要

本项目偏振扩展走 **Walker-Penrose 守恒量 + Connors-Stark 投影**路线，是 EHT 类
偏振成像最常用的简化数学框架。当前实现是**最小可行版本**，覆盖：

- Walker-Penrose 复守恒量（沿 Kerr null geodesic）
- 简化盘磁场模型（toroidal B^φ, Keplerian rotation）
- 在屏幕坐标 (α, β) 下解码观察者 EVPA
- Stokes Q, U 输出 + 端到端 demo

**不覆盖**（明确 limitations）：
- 圆偏振 V（需 Faraday conversion + 完整磁场模型）
- 沿轨迹 radiative transfer（吸收 / Faraday rotation 整合）
- GRMHD 流场 ingestion（需 HARM/iharm reader）
- 频率依赖性 / multi-band（项目目前是单频灰体近似）

要做 EHT-grade，至少补 (a) GRMHD ingestion、(b) full polarized transfer
along geodesic、(c) Faraday rotation。这些被列在 `DELIVERY_STATUS.md` 的可选扩展
扩展 T11 中。

## 数学骨架

### 1. Walker-Penrose 守恒量

对 Kerr null geodesic + 实偏振 4-向量 f（满足 p · f = 0、f · f = 1）：

```
κ_WP = κ_1 + i κ_2 = (A − i B)(r − i a cos θ)
```

其中

```
A = (p^t f^r − p^r f^t) + a sin²θ (p^r f^φ − p^φ f^r)
B = ((r² + a²)(p^φ f^θ − p^θ f^φ) − a (p^t f^θ − p^θ f^t)) sin θ
```

`κ_1` 和 `κ_2` 沿测地线**独立守恒**（Walker & Penrose 1970）。这让我们绕开
显式的 parallel-transport ODE：在 emission point 算一次 κ_WP，就直接知道
observer 端的偏振方向（不必沿轨迹积分 Df^μ/dλ + Γ p f = 0）。

实现：`src.polarization.walker_penrose_complex(state, f, a)`。

### 2. 盘 emission frame 偏振 4-向量

模型：Keplerian fluid + 纯 toroidal magnetic field B^φ。

```
u^μ = u^t (1, 0, 0, Ω)        # Keplerian 4-velocity
b^μ = N (∂_φ + (b · u_cov) u) # toroidal B, normalised so b · u = 0
ω_em = − p · u                # photon energy in fluid frame
k^μ = p^μ / ω_em − u^μ        # photon spatial direction in fluid frame
f_raw = b − (b · k) k          # project out k
f = f_raw / √(f · f)           # final unit polarization
```

物理上对应**光学薄 synchrotron 的电矢量 ⊥ 投影磁场**这一简化（Pacholczyk 1970
Ch. 3）。线偏振分数 Π = 0.1 是 representative ensemble 值。

实现：`src.polarization.emitted_polarization_vector_toroidal(...)`。

### 3. Connors-Stark 屏幕投影

观察者在无穷远处用 Bardeen 屏幕坐标 (α, β)。Bardeen 不变量：

```
λ = − α sin i        # L_z / E
q² = β² + (α² − a²) cos² i
μ = − (λ / sin i + a sin i)   # auxiliary
```

观察者 EVPA：

```
tan(2 χ_obs) = (β κ_2 − μ κ_1) / (β κ_1 + μ κ_2)
```

实现：`src.polarization.decode_observer_evpa(kappa, alpha, beta, a, i_rad)`。

### 4. Stokes 旋转

观察者 frame Stokes Q, U：

```
I_obs = I_em        (灰体强度，本项目就是 I_em = g³ I_emit)
Q_obs = Π · I · cos(2 χ_obs)
U_obs = Π · I · sin(2 χ_obs)
V_obs = 0           (本项目不算圆偏振)
```

实现：`src.polarization.rotate_stokes_to_observer(...)`。

## 端到端 demo

`tools/polarization_demo.py` 跑 48×48 CPU geodesic，逐像素出 Stokes I, Q, U：

```bash
PYTHONPATH=. .venv/Scripts/python.exe tools/polarization_demo.py
```

约 30 秒跑完，输出：
- `results/polarization_demo.json` — 统计摘要
- `figures/polarization_stokes_qu.png` — I, Q, U 三联图
- `figures/polarization_evpa_quiver.png` — 在 Stokes I 底图上叠 EVPA tickmarks

观测样本（a=0.7, i=60°，默认 thin disk）：
- 91.7% (2114/2304) 像素 disk hit
- Π_obs ≈ 0.10 mean = max（与 emission Π 一致——本简化里没有 depolarisation）

## 测试

`tests/test_polarization.py` 10 个测试：

| 测试 | 内容 |
|---|---|
| `test_emitted_stokes_qu_correct_amplitude` | √(Q² + U²) = Π · I |
| `test_emitted_stokes_qu_chi_zero_is_pure_q` | χ=0 → Q=ΠI, U=0 |
| `test_emitted_stokes_qu_chi_quarter_pi_is_pure_u` | χ=π/4 → Q=0, U=ΠI |
| `test_linear_polarization_fraction_invariants` | clamp + zero-I 行为 |
| `test_evpa_round_trip` | EVPA → Stokes → EVPA mod π 守恒 |
| `test_walker_penrose_returns_complex` | κ_WP 是有限 complex |
| `test_walker_penrose_conserved_along_geodesic` | κ_WP 沿轨迹评估稳定 |
| `test_emitted_polarization_orthogonal_to_p` | f · p ≈ 0（在 disk-hit state 上）|
| `test_rotate_stokes_keeps_intensity` | I_obs = I_em 守恒 |
| `test_decode_observer_evpa_returns_finite` | EVPA ∈ [-π/2, π/2]，有限 |

集成进项目测试套件后总数：**41 passed**（19 旧 + 12 disk_color + 10 polarization）。

## 已知 Stub 的局限

1. **不沿轨迹积分 D f^μ / dλ + Γ p f = 0**：靠 Walker-Penrose 常数解析
   解避开。理论上等价；数值上没有累积误差。
2. **磁场模型纯 toroidal**：真实 GRMHD 模拟里磁场结构远更复杂（有 poloidal
   分量、湍流、reconnection 区）。代码里的 `emitted_polarization_vector_toroidal`
   留了接口可以替换。
3. **不算 Faraday rotation**：optically thin synchrotron 假设。在 SgrA* 这种
   多频强 RM 源里这是不正确的。
4. **EVPA 解码使用 Bardeen 大 r_obs 极限**：r_obs → ∞ 假设。当前项目相机
   r_obs = 60 M，这是一个轻微 inconsistency；论文级实现需要在有限 r_obs 下
   完整投影。
5. **demo 图里 EVPA pattern 偏均匀**：因为 Π 是常数 0.1 + 简化磁场，没有
   depolarisation 区。真实 EHT M87 偏振图显示明显的 spiral pattern，要复
   现需上面 1-3 项配合。

## 与 EHT 文献的对照

| 维度 | 本项目 | EHT M87 polarization (paper VIII) |
|---|---|---|
| Polarization formalism | Walker-Penrose + Connors-Stark | Full GRRT polarized transfer (ipole/grtrans) |
| Magnetic field | Toroidal stub | GRMHD-derived (KORAL / HARM)|
| Faraday | None | Full Faraday rotation + conversion |
| Validation | 10 unit tests | Closure phase / amplitude fitting |
| Output | Stokes I, Q, U map (灰度) | Full polarized image with rotation measure |

## 后续

短期（数小时～数天）：
- 在 GPU kernel 里把 κ_WP 计算 inline 化，让偏振 demo 也走 GPU
- 加 EVPA-from-screen 闭式公式的单元测试（与解析极限对比）
- 扩展 toroidal field 模型为 toroidal + poloidal 混合

中期（1-2 周）：
- 实现沿测地线的 parallel-transport ODE，作为 Walker-Penrose 路径的独立校验
- 加 Faraday rotation 沿 geodesic 的积分

长期（数周）：
- 接入 GRMHD 流场（T11）
- 多频带辐射系数 + 频率扫描
- 与 ipole/grtrans 标准测试问题的逐像素对比
