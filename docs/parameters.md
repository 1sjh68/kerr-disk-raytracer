# 参数说明

主配置文件：[`configs/default.yaml`](../configs/default.yaml)。

`load_config()` 读取该 YAML 后，所有运行入口（`run_cpu.py` / `run_gpu.py` / `run_geodesic_*.py` / `render.py` 等）共享同一份配置；可在 Python 代码层覆盖单项后再传入。

## YAML 字段

| 段 | 键 | 含义 |
|---|---|---|
| `black_hole` | `spin` | 无量纲 Kerr 自旋 $a$，约束 $\lvert a \rvert < 1$ |
| `black_hole` | `mass_unit` | 质量单位（默认 1，自然单位 $G = c = M = 1$） |
| `camera` | `r_obs` | 观察者半径（单位 $M$）|
| `camera` | `inclination_deg` | 观察者倾角（单位：度，0° 表示正面观察）|
| `camera` | `fov_m` | 屏幕视场（单位 $M$）|
| `disk` | `inner_radius` | `isco` 字符串或数值 |
| `disk` | `outer_radius` | 盘外半径（单位 $M$）|
| `disk` | `emissivity_index` | 幂律发射指数 $q$，$I_{\mathrm{em}} \propto r^{-q}$ |
| `disk` | `model` | `power_law` 或 `novikov_thorne` |
| `integration` | `step_size` | 单光线 RK4 积分步长 $h$ |
| `integration` | `max_steps` | 测地线积分最大步数 |
| `integration` | `horizon_epsilon` | 视界 $r_h$ 命中容差 |
| `integration` | `escape_radius` | 逃逸半径 |
| `render` | `resolution` | 图像分辨率（fast 路径）|
| `render` | `geodesic_resolution` | 逐像素 geodesic 验证 / 渲染分辨率（默认 48） |
| `render` | `gamma` | 显示 gamma |
| `render` | `bloom` | 是否启用 bloom 后处理 |
| `render` | `tone_map` | tone mapping 模式，当前实现 `reinhard` |
| `render` | `color_mode` | 盘颜色管线：`approx`（默认，启发式 RGB）或 `cie1931`（Planck 谱 + CIE 1931 标准观察者 + sRGB，详见 `src/disk_color.py`） |
| `render` | `cie_t_min` | `color_mode=cie1931` 时的最低黑体温度（K，默认 3000） |
| `render` | `cie_t_max` | `color_mode=cie1931` 时的最高黑体温度（K，默认 30000） |
| `render` | `cuda_block` | CUDA block size，可写 `[16, 16]` 或 `"16x16"` |

## 仅在代码层暴露的选项（YAML 不含）

| 选项 | 函数 | 说明 |
|---|---|---|
| `fast_math` | `src.gpu_trace.render_cuda_geodesic(..., fast_math=True)` | 切到 `--use_fast_math` 编译的 CUDA module，把 `sinf/cosf/sqrtf` 换成硬件 SFU intrinsics。$\ge 192^{2}$ 稳定 ~$3\times$ 加速，accuracy 100%。基准见 `tools/benchmark_fastmath.py`、报告见 `results/fastmath_optimization.md`。 |
| `precision` | 同上 | `"float32"`（默认）/ `"float64"`（科学参考）/ `"float64_opt"`（合并 metric+导数 实验性，未带来净加速） |
| `record_trajectory` | `src.geodesic.trace_single_ray(..., record_trajectory=True)` | 沿 RK4 步保留 $(\lambda, r, \theta, \phi)$ 轨迹，供坐标级验证用。详见 `validation/geokerr_coordinate_alignment.md`。 |

## 单位与约定

- 自然单位 $G = c = M = 1$，时空长度尺度即 $M$。
- 度规约定 $(-, +, +, +)$。
- 坐标顺序：$(t, r, \theta, \phi)$；4-动量协变 $(p_t, p_r, p_\theta, p_\phi)$。
- 红移因子 $g = \nu_{\mathrm{obs}} / \nu_{\mathrm{em}}$；观测强度 $I_{\mathrm{obs}} = g^{3} I_{\mathrm{em}}$。

## 修改示例

```python
import copy
from src.config import DEFAULT_CONFIG
from src.gpu_trace import render_cuda_geodesic

cfg = copy.deepcopy(DEFAULT_CONFIG)
cfg["black_hole"]["spin"] = 0.998
cfg["camera"]["inclination_deg"] = 80.0
cfg["render"]["color_mode"] = "cie1931"
cfg["render"]["cie_t_max"] = 25000.0

data = render_cuda_geodesic(cfg, resolution=256, precision="float32", fast_math=True)
```

更多参数化扫描脚本：见 [`tools/README.md`](../tools/README.md) 的"性能 benchmark / sweep"章节。
