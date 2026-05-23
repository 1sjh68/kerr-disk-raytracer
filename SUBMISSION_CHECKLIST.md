# 提交清单（Submission Checklist）

更新时间：2026-05-24

> 本文档面向**项目最终交付/提交**。审稿人/评委收到压缩包或 git 仓库后，按本清单可在 30 分钟内验证项目可复现。

---

## 1. 仓库结构（顶层）

| 类型 | 文件 / 目录 |
|---|---|
| **元信息** | `LICENSE`, `CITATION.cff`, `README.md`, `DELIVERY_STATUS.md` |
| **入口脚本** | `run_cpu.py`, `run_gpu.py`, `run_geodesic_cpu.py`, `run_geodesic_gpu.py`, `render.py`, `benchmark.py`, `validate.py`, `validate_geodesic.py` |
| **环境** | `requirements.txt`, `requirements-cuda.txt`, `environment.yml`, `Dockerfile`, `.dockerignore` |
| **跨平台 git 配置** | `.gitignore`, `.gitattributes` |
| **核心代码** | `src/`（14 模块）, `cuda/`（kernels.cu + profile_geodesic.cu）, `tests/`（11 测试文件，47 passed）, `configs/default.yaml` |
| **辅助 / 复现** | `scripts/`（一键 run_all 等）, `tools/`（20 脚本含 README）|
| **文档** | `docs/`（installation / parameters / physics_model / equation_reference / polarization / extensions_roadmap / reproduction）|
| **科学产物** | `reference/`（5 个 `.npz` 验证基准）, `validation/`（13 个 json/md）, `results/`（24 个 json/md/png 性能与指标）, `figures/`（含 24 张扫描缩略图 + 3 个 GIF + 论文用图）, `logs/`（运行元数据）, `output/`（PNG 渲染产物）|
| **论文 / 汇报** | `paper/main.{md,pdf}`（中文 + LaTeX 公式）, `paper/presentation/{output.pptx, talk_script.md}` |
| **研究文档** | `research/`（literature_review、github_repo_audit、research_gap、project_scope 等 + reproduction/ 文件夹；`research/repos/` 已 gitignore） |

---

## 2. 已被 `.gitignore` 排除（不在仓库里）

- `.venv/`、`__pycache__/`、`.pytest_cache/`、`*.egg-info/`、`build/`、`dist/`
- `.kiro/`、`.cursor/`、`.codex_resources.json`、`.aider*`（AI/IDE 缓存）
- `cuda/profile_geodesic{,.exe}`、`cuda/*.{ptx,o,cubin}`（CUDA 构建产物）
- `paper/presentation/build/node_modules/`、`paper/presentation/preview/`、`paper/presentation/scratch/`
- `research/repos/`（外部 clone：geokerr / Odyssey 源码 + abgrid_r60.out 验证数据，约 12 MB；用 `scripts/summarize_external_demos.py` 重建）
- `reference/*.npy`（与 `.npz` 重复的旧副本；`save_maps_npy` 函数留在 `src/render.py` 但 run_*.py 已不再调用）
- `output/*.npz`（运行时大文件 1+ MB；run 一次即可重生成）
- `results/ncu_*.ncu-rep`（Nsight Compute 二进制报告，跑 `tools/run_ncu_pipeline.ps1` 后产生）
- `*.log`、`*.trace`、`.env`、`*.local.{yaml,json}`、`.DS_Store`、`Thumbs.db`、`desktop.ini`

---

## 3. 跨平台 git 行为（`.gitattributes`）

- 默认文本统一 LF（Linux/macOS 友好；Windows 端 git 自动转 CRLF）
- `*.ps1` / `*.bat` / `*.cmd` / `*.reg` 强制 CRLF
- `*.sh` / `*.cu` / `*.cuh` 强制 LF（Linux/WSL nvcc 兼容）
- 二进制：`*.png`、`*.pdf`、`*.pptx`、`*.npz`、`*.ncu-rep`、`*.exe`、`*.dll` 等
- `*.cu` / `*.cuh` 标记 `linguist-language=cuda`，GitHub UI 正确高亮

---

## 4. 验证步骤（30 分钟内复现）

### 必跑（< 5 分钟）

```bash
# Linux / WSL
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests -q                  # 应得 47 passed
python run_cpu.py                                        # ~5 s, 写 reference/cpu_reference{,_256}.npz
python run_geodesic_cpu.py                               # ~30 s, 写 reference/cpu_geodesic_reference.npz
```

```powershell
# Windows
.\.venv\Scripts\python.exe -m pytest tests -q            # 应得 47 passed
.\.venv\Scripts\python.exe run_cpu.py
.\.venv\Scripts\python.exe run_geodesic_cpu.py
```

### 一键完整流水线（5–15 分钟）

```bash
./scripts/run_all.sh                                     # WSL / Linux
.\scripts\run_all.ps1                                    # Windows
```

会顺序跑：核心 CPU/GPU 渲染 → benchmark → validate → 8 个新 demo（fastmath / disk_param_sweep / CIE 颜色对比 / geokerr 坐标对齐 / RK45 vs RK4 / EHT 指标 / 偏振 demo / 动画合成）→ pytest → PPT deck（如有 Node.js）。

### GPU 加速验证（需 RTX/CUDA）

```bash
python scripts/check_cuda.py                             # 输出 CuPy / CUDA 版本与设备信息
python run_gpu.py                                        # fast thin-disk GPU
python run_geodesic_gpu.py --precision float64           # 完整 geodesic GPU
python validate.py                                       # CPU/GPU 一致性
python validate_geodesic.py                              # geodesic CPU/GPU 一致性
```

### 完整 Nsight Compute Profile（需 Windows admin + 一次 UAC）

```powershell
powershell -File 'tools\run_ncu_pipeline.ps1'
# 按任意键 -> UAC 弹窗点 [是] -> 5–15 分钟自动跑 4 个 --set full
# 产物落入 results/ncu_*.{ncu-rep,summary.txt,csv}
```

---

## 5. 必看文档（按顺序）

1. `README.md`：项目快照、当前能力表、快速运行入口
2. `DELIVERY_STATUS.md`：完整交付状态、本轮新增成果（12/12 闭环）、关键性能数字
3. `paper/main.md` / `paper/main.pdf`：中文论文式技术报告（含 LaTeX 公式 + 14 张图）
4. `paper/presentation/output.pptx` + `paper/presentation/talk_script.md`：5 分钟汇报材料
5. `docs/`：物理模型、参数、Carter 常数推导、偏振、扩展 roadmap、复现说明
6. `tools/README.md`：所有辅助脚本的入口总览（按用途分组）

---

## 6. 关键验证数字（应在评审中可对照）

| 指标 | 值 | 来源文件 |
|---|---|---|
| pytest | **47 passed** | `pytest tests -q` |
| CPU/GPU 状态匹配（48×48 float64） | **99.96%** | `validation/geodesic_cpu_gpu_comparison.json` |
| Disk-hit 分类一致 | **100%** | 同上 |
| Intensity MAE（float64） | **1.07e-10** | 同上 |
| geokerr 严格对比（400 ray, a=0.7, i=60°） | 总体 **91.25%** / disk **98.27%** | `validation/geokerr_strict_comparison.json` |
| fast_math 加速比（256²）| **2.93×**，accuracy 100% | `results/fastmath_benchmark.json` |
| RK45 vs RK4（临界 ray）| null preservation 改善 ~36 数量级 | `validation/rk45_vs_rk4_demo.md` |
| 参数扫描 | 6 自旋 × 4 倾角 = 24 配置 | `results/parameter_sweep.json` |
| EHT ring diameter（a=0.7, i=60°）| 9.72 M（折算 ~37 μas，与 EHT 42±3 μas 同量级）| `results/eht_metrics_report.md` |
| 偏振 demo | Π_obs=0.10, 91.7% disk hit | `results/polarization_demo.json` |

---

## 7. 仓库体量（清理后）

```
~16 MB（除去 .venv、research/repos 等已 gitignore 的目录）
├── research/           ~12.8 MB（含 reproduction/ 三张图 + 12 个 md/json；repos/ 已 gitignore）
├── paper/               ~2.9 MB（main.pdf 1.7 MB + output.pptx 0.3 MB + presentation 杂项）
├── figures/             ~2.9 MB（论文用图 + 24 张扫描 + 3 个 GIF）
├── reference/          ~1.6 MB（5 个 .npz 验证基准）
├── results/             ~0.4 MB
├── output/              ~0.04 MB（PNG only；.npz 已 gitignore 除外）
├── src/                 ~0.2 MB（14 模块）
├── tools/               ~0.1 MB（20 脚本）
├── tests/               ~0.1 MB（11 测试）
└── 其他                 ~0.05 MB（cuda/, scripts/, docs/, configs/, validation/, logs/）
```

`research/repos/`（外部 geokerr + Odyssey 源码 + abgrid_r60.out 验证数据，约 12 MB）需要审稿方按需自行 clone。详见 `research/literature_review.md` §4-§5 的链接，或运行 `scripts/summarize_external_demos.py` 自动汇总。

---

## 8. 提交前最后一遍 checklist（开发者自查）

- [x] `pytest tests -q` 全过（47 passed）
- [x] `run_cpu.py` 清理后跑通且不再生成冗余 `.npy`
- [x] `.gitignore` 覆盖所有运行时产物（reference/*.npy、output/*.npz、ncu-rep、缓存等）
- [x] `.gitattributes` 设好行尾（CRLF for `.ps1/.bat`，LF for `.sh/.cu`，binary 标记）
- [x] 所有 `.md` 文档交叉引用都还有效（`DELIVERY_STATUS.md` 已修正注明运行时产物）
- [x] `paper/main.md` 公式已 LaTeX 化（9 块 + 138 行内）
- [x] `paper/main.pdf` 由 `scripts/make_report.py` 重新生成（中文 + 14 页）
- [x] 顶层 19 个文件全部必要，无多余 / 临时文件
- [x] `tools/` 20 脚本无密码明文（之前已清理过 8 个含 `1234` 的临时脚本）

如果你看到这一行——可以提交了。

---

## 9. 提交格式建议

### A. Git 仓库（推荐）

```bash
cd 'D:\Desktop\black hole'
git init
git add .
git status                    # 检查无大文件意外加入
git commit -m "Initial submission: Kerr thin-disk CPU/GPU ray tracing"
git remote add origin <url>
git push -u origin main
```

### B. 压缩包

```powershell
# 排除 .venv / .git / research/repos / .pytest_cache / __pycache__
$exclude = @('.venv', '.git', '.pytest_cache', '__pycache__', 'research\repos')
Compress-Archive -Path 'D:\Desktop\black hole\*' -DestinationPath 'submission_2026_05_24.zip' -Force
# 或手动剪一下 .venv 之后再压缩
```

最终大小：仓库约 16 MB（无 .venv / 无外部 repos），压缩后约 5–8 MB。
