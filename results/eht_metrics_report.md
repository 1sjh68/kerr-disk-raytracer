# EHT-Style Image Metrics

更新时间：自动生成。详见 `src/eht_metrics.py` 与 `tools/eht_metrics_demo.py`。

## 与 EHT 文献的对照

| 量 | 本项目 (a=0.7, i=60°, 48×48 float64) | EHT M87* (2019) | EHT Sgr A* (2022) |
|---|---|---|---|
| Ring diameter (M units → μas via M=6.5e9 M_sun, D=16.8 Mpc) | 9.72 M | 42 ± 3 μas | 48.7 ± 7 μas |
| South / North asymmetry | 0.66 | ≳ 10 | 强但变化快 |
| Photon-ring peak radius | 4.60 M | shadow radius ≈ 5√3 ≈ 8.66 M (a=0) | 类似 |

**说明**：
- 本项目的 ring diameter 是几何半径中位数 × 2（intensity ≥ 0.5·max 的像素），
  单位是 gravitational radius `M`。换算到 μas：M87* 的 1 M ≈ 3.83 μas
  （M=6.5e9 M_sun, D=16.8 Mpc）。
- 本项目使用 thin-disk + power-law emission，不带磁场动力学，所以
  asymmetry 完全由 Doppler beaming + 视差几何贡献。EHT 的 ≳ 10:1 ratio
  来自 GRMHD 亚相对论流场 + 完整 GRRT。
- Photon-ring peak 在 thin disk 模型下不严格存在；用径向亮度峰值
  作 proxy（Bardeen 1973 photon-sphere ≈ 5.2 M for a=0.7）。

## 24 配置扫描表（来自 luminance proxy）

| spin | inclination (deg) | D_ring [M] | south/north | peak_r [M] |
|---|---|---|---|---|
| +0.000 | 10 | 20.75 | 1.04 | 7.42 |
| +0.000 | 30 | 18.37 | 1.11 | 7.42 |
| +0.000 | 60 | 16.35 | 1.13 | 7.42 |
| +0.000 | 80 | 15.24 | 0.87 | 7.42 |
| +0.300 | 10 | 19.03 | 1.03 | 6.72 |
| +0.300 | 30 | 16.08 | 1.06 | 6.72 |
| +0.300 | 60 | 14.62 | 1.06 | 6.72 |
| +0.300 | 80 | 13.63 | 0.79 | 6.72 |
| +0.500 | 10 | 17.21 | 1.01 | 6.01 |
| +0.500 | 30 | 14.28 | 1.03 | 6.01 |
| +0.500 | 60 | 13.21 | 1.01 | 6.01 |
| +0.500 | 80 | 12.27 | 0.73 | 6.01 |
| +0.700 | 10 | 15.42 | 1.01 | 5.30 |
| +0.700 | 30 | 12.54 | 1.02 | 5.30 |
| +0.700 | 60 | 11.56 | 0.96 | 5.30 |
| +0.700 | 80 | 10.85 | 0.67 | 6.01 |
| +0.900 | 10 | 13.24 | 1.00 | 3.89 |
| +0.900 | 30 | 10.55 | 0.99 | 4.60 |
| +0.900 | 60 | 9.46 | 0.89 | 4.60 |
| +0.900 | 80 | 8.78 | 0.59 | 3.89 |
| +0.998 | 10 | 12.01 | 0.99 | 3.18 |
| +0.998 | 30 | 9.59 | 0.97 | 3.89 |
| +0.998 | 60 | 8.10 | 0.83 | 3.18 |
| +0.998 | 80 | 7.12 | 0.51 | 3.18 |

生成：`PYTHONPATH=. .venv/Scripts/python.exe tools/eht_metrics_demo.py`