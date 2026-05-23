# Equation Reference

| Quantity | Formula | Implemented In |
|---|---|---|
| Sigma | `r^2 + a^2 cos^2(theta)` | `src/metric.py` |
| Delta | `r^2 - 2r + a^2` | `src/metric.py` |
| Horizon | `1 + sqrt(1 - a^2)` | `src/metric.py` |
| ISCO | Bardeen-Press-Teukolsky formula | `src/metric.py` |
| Null Hamiltonian | `0.5 g^{mu nu} p_mu p_nu` | `src/metric.py` |
| Hamiltonian RHS | `xdot = g^{-1} p`, `pdot = -0.5 d(g^{-1}) p p` | `src/geodesic.py` |
| Keplerian Omega | `1 / (r^(3/2) + a)` | `src/metric.py` |
| Redshift | `1 / (u^t (1 - Omega lambda))` | `src/metric.py` |
| Power-law emissivity | `r^-q` | `src/disk.py` |
| NT-like flux | `r^-3 (1 - sqrt(r_in/r))` | `src/disk.py` |
| Observed intensity | `g^3 I_emit` | `src/disk.py` |

## Validation Checks

- `tests/test_metric.py`: horizon, ISCO and metric inverse identity.
- `tests/test_geodesic.py`: null constraint and single-step geodesic sanity checks.
- `tests/test_redshift.py`: redshift positivity and intensity dependence on `g`.

---

## Carter 常数形式：作为高精度参考

> 本项目的 GPU/CPU 主路径用 **Hamiltonian 形式**（8 维相空间 + RK4），原因是 CPU/GPU 可以共用一份 RHS 实现，便于一致性验证。Carter 常数路线在这里**作为半解析高精度参考**列出，用于：
> 1. 单光线 conserved quantities 校验：积分过程中 E、L_z、Q 应保持守恒。
> 2. 与 geokerr/YNOGK 这类 Carlson elliptic integral 半解析代码做坐标级对比。
> 3. 论文 / 报告里说明本项目主路径与"经典 Kerr null geodesic"在数学上等价。

### 1. Killing 矢量与守恒量

Kerr 度规在 Boyer-Lindquist 坐标下不依赖 `t` 与 `phi`，对应两个 Killing 矢量 `xi_(t)^mu = (1, 0, 0, 0)`、`xi_(phi)^mu = (0, 0, 0, 1)`。沿着 affine 参数 lambda 的零测地线，给出两个守恒量：

```
E   = - g_{mu nu} xi_(t)^mu p^nu  = - p_t                 # photon energy
L_z =   g_{mu nu} xi_(phi)^mu p^nu =   p_phi              # axial angular momentum
```

注意约定：covariant `p_t` 与物理 energy 差一个负号；covariant `p_phi` 等于角动量。本项目把外部观察者初始化时设 `E = -p_t = 1`（在 affine 参数下的归一化），`L_z` 由观察者方向决定。

### 2. Carter 常数

Kerr 时空可分离变量，存在第三个守恒量。零测地线情形（`mu^2 = 0`）的 Carter 常数：

```
Q = p_theta^2 + cos^2(theta) [ -a^2 E^2 + L_z^2 / sin^2(theta) ]
```

`Q + (L_z - a E)^2` 是另一个常用形式（Carter "K"），适合在 spin = 0 极限下退化到 Schwarzschild 的 `L_z^2 / sin^2(theta) + p_theta^2`。两者等价。

### 3. 分离变量后的 1D ODE

四个守恒量（`mu^2 = 0`、`E`、`L_z`、`Q`）允许把 4 维 ODE 系统分离成两个独立的 1D ODE：

```
Sigma^2 (dr/dlambda)^2     = R(r)
Sigma^2 (dtheta/dlambda)^2 = Theta(theta)
```

其中

```
R(r)         = [E (r^2 + a^2) - a L_z]^2  -  Delta * [(L_z - a E)^2 + Q]
Theta(theta) = Q  -  cos^2(theta) [ -a^2 E^2 + L_z^2 / sin^2(theta) ]
```

`phi` 与 `t` 的演化是 r 和 theta 的纯函数：

```
Sigma (dphi/dlambda) = a * [E (r^2 + a^2) - a L_z] / Delta
                       - a E
                       + L_z / sin^2(theta)

Sigma (dt/dlambda)   = (r^2 + a^2) * [E (r^2 + a^2) - a L_z] / Delta
                       - a^2 E sin^2(theta)
                       + a L_z
```

数学上这等价于 Hamiltonian 系统在去掉两个 cyclic 自由度后的简化形式。

### 4. Turning Points 与 elliptic 积分

`R(r) = 0` 的根定义径向 turning points（光子能到达的最小或最大 r）；同样 `Theta(theta) = 0` 给出极向 turning points。这两套 turning points 把光线轨道分类为：

| 类型 | r-动力学 | theta-动力学 |
|---|---|---|
| Plunge | 单调下降到视界 | 振荡或单调 |
| Bound | r 有界，振荡 | 通常振荡 |
| Escape | r 单调到无穷 | 振荡或单调 |

geokerr 与 YNOGK 都把这些 1D ODE 转换成 **Carlson elliptic integrals** 求闭式解，避免数值积分的 `Sigma^2` 奇点和 turning point 处的精度损失。这条半解析路线在临界轨道（光子球附近）显著优于 RK4，是项目里"剩余 8.75% 状态分歧"的主要来源。

### 5. 与 Hamiltonian 路线的等价性

Hamiltonian 和 Carter 路线在**理论上完全等价**，都源自同一份度规。差异只在数值实现：

| 项目 | Hamiltonian (本项目主路径) | Carter (作参考) |
|---|---|---|
| 状态维度 | 8 (`x^mu`, `p_mu`) | 4 (r, theta, phi, t) + 3 守恒量 |
| 数值方法 | RK4 / RK45 固定 / 自适应步长 | Carlson elliptic 半解析 |
| Turning points | 隐式（步长穿越） | 显式（解 R=0, Theta=0）|
| GPU 友好 | 是（CPU/GPU 同构） | 否（per-ray 控制流复杂） |
| 守恒量验证 | 可在 RHS 之外周期性检查 `E`, `L_z`, `Q` 保持 | 自动满足（求闭式解）|
| 临界光线精度 | 中（步长敏感）| 高（Carlson 在 turning 处稳定）|

### 6. 项目中作为参考的用法

```python
# 伪代码：在 CPU geodesic 跑完后，沿轨迹周期性核查守恒量

import numpy as np

def carter_constants(state, a):
    """从 8 维状态向量解出 (E, L_z, Q)。"""
    t, r, theta, phi, pt, pr, ptheta, pphi = state
    E   = -pt
    L_z = pphi
    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2
    Q = ptheta**2 + cos2 * (-a*a*E*E + L_z*L_z / max(sin2, 1e-12))
    return E, L_z, Q

# 期望：在 RK4 沿轨迹积分时，max |Carter(end) - Carter(start)| / |Carter(start)| < 1e-3
# 实测见 validation/error_summary.md（本项目 null Hamiltonian 漂移 ~1e-6 量级）
```

### 7. 进一步阅读

- Carter, "Global structure of the Kerr family of gravitational fields", Phys. Rev. 174, 1559 (1968).
- Bardeen, "Timelike and null geodesics in the Kerr metric", Les Houches lectures (1973).
- Dexter & Agol, "geokerr" (2009): https://arxiv.org/abs/0903.0620
- Yang & Wang, "YNOGK" (2013): https://arxiv.org/abs/1305.1250
- 本项目相关文件：`src/metric.py`（度规与导数）、`src/geodesic.py`（Hamiltonian RHS）、`research/literature_review.md` 第 2 节。
