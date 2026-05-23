# Kerr 黑洞薄盘 CPU/GPU 光线追踪

可复现的 Kerr 黑洞 thin-disk 广义相对论光线追踪项目：CPU 高精度基准、CuPy/CUDA 并行 geodesic kernel、验证矩阵、性能评估与论文级产物。

## 一句话信号

- **float64 geodesic @ 48×48**：CPU/GPU 状态匹配 **99.96%**，disk-hit 分类 **100%**
- **fast_math float32 @ 256×256**：实测 **2.93× 加速**，accuracy 100%
- **geokerr 严格对比**（a=0.7, i=60°）：总体 **91.25%**、disk **98.27%** 一致
- **测试**：`python -m pytest tests` → **47 passed**
- **论文 / 汇报**：`paper/main.pdf`（中文 + LaTeX 公式）+ 8 张 PPT + 5 分钟讲稿

详细进度与产物清单见 [`DELIVERY_STATUS.md`](DELIVERY_STATUS.md)；提交检查清单见 [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md)。

## 目录结构

```text
black_hole/
├── src/                  物理与渲染核心（14 模块）
│   ├── metric.py             Kerr 度规、ISCO、视界
│   ├── geodesic.py           CPU Hamiltonian 测地线（含 RK4 / RK45）
│   ├── gpu_trace.py          CuPy RawKernel 驱动（含 fast_math 模块）
│   ├── render.py             tone mapping / bloom / 颜色 dispatch
│   ├── disk.py               盘发射模型（power-law / NT）
│   ├── disk_color.py         CIE 1931 + Planck + sRGB 严格颜色管线
│   ├── polarization.py       Walker-Penrose 偏振 stub（Stokes I,Q,U）
│   ├── eht_metrics.py        ring diameter / asymmetry 指标
│   ├── grmhd_io.py           GRMHD HDF5 reader stub + synthetic fluid
│   └── ...                   camera / config / integrators / safe_io
│
├── cuda/                 CUDA C 源码
│   ├── kernels.cu            float32 + float64 + opt + fast thin-disk kernel
│   └── profile_geodesic.cu   standalone 二进制（用于 ncu profile）
│
├── tests/                pytest 套件（47 passed）
│
├── configs/default.yaml  运行时参数（spin / camera / disk / render / color_mode / …）
│
├── tools/                辅助脚本（24 个，详见 tools/README.md）
│   ├── benchmark_fastmath.py      fast_math 加速比基准
│   ├── parameter_sweep.py         spin × inclination 24 组扫描
│   ├── disk_param_sweep.py        r_outer + emissivity 扫描
│   ├── geokerr_*_compare.py       与 geokerr 状态级 / 坐标级对比
│   ├── rk45_vs_rk4_demo.py        临界 ray 上 RK4 vs RK45
│   ├── eht_metrics_demo.py        24 配置 ring diameter / asymmetry
│   ├── polarization_demo.py       Stokes I,Q,U + EVPA 端到端 demo
│   ├── compose_animations.py      参数扫描合成 GIF
│   ├── run_ncu_pipeline.ps1       Windows host ncu 一键流水线（用户入口）
│   ├── _ncu_full_pipeline.ps1     ↑ admin 提权后的内部脚本
│   ├── wsl_profile_pipeline.sh    WSL 端 nvcc + ptxas + CuPy events 报告
│   ├── ncu_pipeline.md            ncu 流水线说明
│   └── README.md                  全部脚本入口分类总览
│
├── scripts/              一键复现入口
│   ├── run_all.ps1 / run_all.sh   把全套渲染 + 验证 + demo + pytest 串起来
│   ├── check_cuda.py              CUDA 环境探测，写 logs/cuda_status.json
│   ├── make_report.py             生成 paper/main.pdf（matplotlib + 中文字体）
│   ├── parse_geokerr.py           解析 geokerr abgrid.out
│   ├── run_experiments.py         性能基准
│   └── summarize_external_demos.py  外部 demo 状态汇总
│
├── docs/                 文档（中文 + LaTeX 公式，详见 docs/README.md）
│   ├── installation.md       Python / CUDA / Docker / WSL2 安装
│   ├── reproduction.md       一键复现 + 产物清单
│   ├── parameters.md         configs/default.yaml 参数表
│   ├── physics_model.md      Kerr 度规 / 测地线 / 薄盘 / 红移
│   ├── equation_reference.md 公式速查 + Carter 常数
│   ├── polarization.md       Walker-Penrose 偏振数学骨架
│   └── extensions_roadmap.md 多 GPU / 可微 / surrogate 路线图
│
├── paper/                论文与汇报
│   ├── main.md / main.pdf    中文论文（含 LaTeX 公式 + 14 页 + 12 张图）
│   └── presentation/         output.pptx + talk_script.md（5 分钟讲稿）
│
├── research/             文献与外部代码调研
│   ├── literature_review.md      文献综述（含 EHT M87*/Sgr A* 范畴边界）
│   ├── github_repo_audit.md      公开实现审计（geokerr / Odyssey / grtrans / ipole）
│   ├── research_gap.md           研究空白
│   ├── reproduction/             外部 demo 复现图
│   └── repos/                    外部 clone 源码（.gitignore）
│
├── reference/            5 个 .npz 验证基准（CPU 256² + 各 geodesic 参考）
├── results/              性能与指标（27 个 json / md / png）
├── validation/           CPU/GPU + geokerr 一致性报告（10 文件）
├── figures/              论文用图 + 24 张扫描缩略图 + 3 个 GIF
├── output/               运行时渲染图像（5 张 PNG）
├── logs/                 运行元数据 JSON（8 文件）
│
├── 入口脚本（项目根）
│   ├── run_cpu.py            fast thin-disk CPU 预览
│   ├── run_gpu.py            fast thin-disk GPU
│   ├── run_geodesic_cpu.py   完整 Hamiltonian 测地线 CPU
│   ├── run_geodesic_gpu.py   完整 Hamiltonian 测地线 GPU（--precision float32/64）
│   ├── render.py             tone mapping / 后处理
│   ├── benchmark.py          CPU vs GPU 基准
│   ├── validate.py           fast 路径 CPU/GPU 一致性
│   └── validate_geodesic.py  测地线路径 CPU/GPU 一致性
│
├── 元信息
│   ├── README.md             本文档
│   ├── DELIVERY_STATUS.md    交付状态 + 关键验证数字
│   ├── SUBMISSION_CHECKLIST.md  提交前检查清单
│   ├── LICENSE / CITATION.cff
│   └── requirements.txt / requirements-cuda.txt / environment.yml / Dockerfile
│
└── git 配置
    ├── .gitignore            排除 .venv / 缓存 / 运行时大产物 / IDE 杂项
    └── .gitattributes        统一行尾（默认 LF；.ps1/.bat CRLF；二进制标记）
```

## 快速运行

```powershell
# Windows 一键完整流水线
.\scripts\run_all.ps1
```

```bash
# Linux / WSL
./scripts/run_all.sh
```

或手动逐步：

```powershell
.\.venv\Scripts\python.exe -m pytest tests           # 47 passed
.\.venv\Scripts\python.exe run_cpu.py
.\.venv\Scripts\python.exe run_geodesic_cpu.py
.\.venv\Scripts\python.exe scripts\check_cuda.py
.\.venv\Scripts\python.exe run_gpu.py
.\.venv\Scripts\python.exe run_geodesic_gpu.py --precision float64
.\.venv\Scripts\python.exe validate.py
.\.venv\Scripts\python.exe validate_geodesic.py
```

完整复现步骤、产物清单、已知限制：[`docs/reproduction.md`](docs/reproduction.md)。

## 文档索引

| 类别 | 文档 |
|------|------|
| 项目入口 | [`docs/installation.md`](docs/installation.md), [`docs/reproduction.md`](docs/reproduction.md) |
| 物理与公式 | [`docs/physics_model.md`](docs/physics_model.md), [`docs/equation_reference.md`](docs/equation_reference.md) |
| 配置 | [`docs/parameters.md`](docs/parameters.md) |
| 物理扩展 | [`docs/polarization.md`](docs/polarization.md), [`docs/extensions_roadmap.md`](docs/extensions_roadmap.md) |
| 工具索引 | [`tools/README.md`](tools/README.md), [`tools/ncu_pipeline.md`](tools/ncu_pipeline.md) |
| 论文 / 汇报 | [`paper/main.pdf`](paper/main.pdf), [`paper/presentation/output.pptx`](paper/presentation/output.pptx), [`paper/presentation/talk_script.md`](paper/presentation/talk_script.md) |
| 状态 / 提交 | [`DELIVERY_STATUS.md`](DELIVERY_STATUS.md), [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) |

## 已知限制

- Windows 宿主机系统级 `nvcc` 通常不在 PATH；CuPy + PyPI CUDA wheels 已满足 RawKernel 编译。WSL2 路径下 ncu 在 driver 566.07 不可行，详见 [`results/wsl_profile_report.md`](results/wsl_profile_report.md)；推荐用 Windows host 路径 [`tools/run_ncu_pipeline.ps1`](tools/run_ncu_pipeline.ps1)。
- geokerr 临界光线 captured/disk 边界为浮点精度本质性差异（46% 一致），需 Carlson 椭圆积分半解析路线根除。
- 扩展工作中：偏振、GRMHD、EHT 风格指标已 stub；完整 polarized radiative transfer、GRMHD time-evolved ingestion、多 GPU、可微 ray tracing、神经网络 surrogate 留作后续（见 [`docs/extensions_roadmap.md`](docs/extensions_roadmap.md)）。

## 引用

见 [`CITATION.cff`](CITATION.cff) 与 [`LICENSE`](LICENSE)。
