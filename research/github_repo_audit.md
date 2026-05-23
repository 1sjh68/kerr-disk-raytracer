# GitHub / 开源项目审计表

更新时间：2026-05-21

## 1. 审计结论

本项目不应直接 fork 某个现成代码库作为主线。更稳妥的路线是：

- 用 **geokerr / YNOGK** 做少量轨迹和公式交叉验证；
- 用 **Odyssey** 学习 CUDA 并行组织；
- 用 **grtrans / ipole / GYOTO** 学习工程化、文档化和测试问题；
- 保持本项目实现独立，许可证、数据结构和验证脚本都可控。

## 2. 项目对比表

| 项目 | 来源 | 主要语言 | 物理模型 | 数值/积分方法 | GPU 支持 | 测试/示例 | 许可证 | 可复现性评价 | 对本项目的用途 |
|---|---|---|---|---|---|---|---|---|---|
| GYOTO | https://github.com/gyoto/Gyoto | C/C++/Python/SWIG/Yorick | GR orbit/ray-tracing 框架，2.0 支持偏振相对论 ray tracing | 插件式 metric/object/ray tracing | 不是 CUDA 主线 | `doc/examples` 可用 `gyoto input.xml output.fits` | GPL-3.0 | 构建体系成熟，但依赖和接口较重 | 工程设计、FITS 输出、示例组织参考 |
| geokerr | https://faculty.washington.edu/agol/geokerr/index.html | Fortran77 | Kerr photon orbits | Carlson elliptic integrals，半解析 photon orbit | 无 | `abgrid.in` 示例，`./geokerr < abgrid.in > abgrid.out` | 页面未明确现代开源许可证，需谨慎 | 代码小、参考价值高，但年代久 | P0 交叉验证参考，不复制代码 |
| YNOGK | https://arxiv.org/abs/1305.1250 | Fortran | Kerr null geodesics | Weierstrass/Jacobi elliptic functions，Carlson elliptic integrals | 无 | 文献 toy problems | 论文称源码公开，具体许可证需单独确认 | 公式价值高，源码获取可能不稳定 | event/root-finding 和 turning point 处理参考 |
| Odyssey | https://github.com/hungyipu/Odyssey | C++/CUDA/C | Kerr GRRT，图像/光谱，Keplerian disk/shell 示例 | CUDA kernel 内 ray update，adaptive RK5 | 有，CUDA | 默认 task1/task2，输出结果文件 | GPL-3.0 | 结构清晰，GPU 主线明确；较旧 | GPU 数据结构、kernel layout、early termination 参考 |
| grtrans | https://github.com/jadexter/grtrans | Fortran90/Fortran77/Python | Kerr polarized GRRT，thin disk/jet/HARM | geokerr + radiative transfer；OpenMP | 无 CUDA，CPU/OpenMP | `run_grtrans_test_problems_public.py` | MIT | 科学生产代码，依赖 CFITSIO/pyfits | 配置分层、debug maps、测试问题参考 |
| ipole | https://github.com/AFD-Illinois/ipole | C | Covariant polarized radiative transport | semi-analytic polarized transport scheme | 无 CUDA 主线 | EHT 类源建模，依赖 HDF5/GSL | BSD-3-Clause | 相对清晰，但目标是偏振输运 | 后期偏振扩展参考 |
| Gargantua / DNGR | https://arxiv.org/abs/1502.03808 | 论文描述，生产代码未公开 | Kerr lensing + ray bundle，电影级渲染 | ray-bundle propagation，抗闪烁与 caustic 处理 | 工业渲染管线，非公开 CUDA 项目 | 论文图像和方法描述 | 无可复用开源许可证 | 视觉和 ray bundle 思路有价值，但不可复现源码 | 报告相关工作，不作为实现基础 |
| CUDAKerr / 零散 CUDA 项目 | 需逐个确认 | 多为个人项目 | Schwarzschild/Kerr 可视化不等 | 常见 RK4 或近似 ray marching | 有 | 质量差异大 | 不稳定 | 很多不是天体物理 GRRT 代码 | 只作为 UI/性能参考，不作为物理参考 |

## 3. 项目逐项审计

### 3.1 GYOTO

证据：

- GitHub README 说明 GYOTO 是 Observatoire de Paris 的 General relativitY Orbit Tracer，包含 `libgyoto`、utility programs 和 Yorick 插件。
- README 要求引用 GYOTO 2011 和 GYOTO 2.0 相关论文。
- 仓库许可证显示 GPL-3.0。
- 示例命令为 `gyoto <input-file.xml> <output-file.fits>`。

优点：

- 框架成熟，文档和示例完整。
- 支持自定义 metrics 和 astronomical objects。
- 适合学习科学软件组织方式。

风险：

- 依赖重，初始构建成本高。
- GPL-3.0 不适合直接复制到许可证更宽松的项目。
- 对本项目的 CUDA MVP 没有直接帮助。

使用建议：

- 不 fork 主线。
- 后续可选择一个 `doc/examples` 场景做图像级 sanity check。

### 3.2 geokerr

证据：

- 官方页面说明该代码对应 Dexter & Agol 2009。
- 5/18/2009 release 包含 `geokerr.f`、`abgrid.in`、`inputex.f` 和 manual。
- 官方 quick start 使用 `gfortran -ffixed-line-length-132 -O3 geokerr.f -o geokerr`，再运行 `./geokerr < abgrid.in > abgrid.out`。

优点：

- 代码规模小，适合本地复现。
- 对 Kerr photon orbit 的半解析计算是强参考。
- 适合抽样验证单条轨迹、turning point 和屏幕坐标映射。

风险：

- Fortran77 风格老，维护性弱。
- 许可证不够清晰，不能直接复制核心实现。
- 输出格式需写 adapter。

使用建议：

- 下载到 `research/repos/geokerr/`。
- 记录构建日志到 `research/reproduction_log.md`。
- 抽取 3-5 条屏幕点轨迹作为本项目 `validation/geokerr_reference.csv`。

### 3.3 YNOGK

证据：

- 论文说明 YNOGK 是计算 Kerr spacetime null geodesics 的公开代码。
- 公式使用 Weierstrass/Jacobi elliptic functions 和参数 `p`。
- 摘要强调用户不必预先指定 turning points，并通过文献 toy problems 做了验证。

优点：

- 对 event/root-finding 建模非常有启发。
- 适合验证本项目对 turning point、observer-emitter 和高自旋的处理。

风险：

- 源码下载页面可能变动。
- Fortran 和特殊函数依赖会增加复现成本。
- 许可证需要确认。

使用建议：

- 先以论文公式和测试场景做参考。
- 若源码可稳定获取，再纳入交叉验证；否则不作为 CI 必需项。

### 3.4 Odyssey

证据：

- GitHub README 说明 Odyssey 是 public GPU-based GRRT code，CUDA C/C++ 实现。
- README 展示默认任务包括 Keplerian disk redshift 和 Keplerian rotating shell image。
- README 描述 `main.cpp -> task*.cpp -> GPUcompute -> __global__ GPU_task*work()` 流程，kernel 内使用 adaptive Runge-Kutta 更新 ray。
- 仓库显示 GPL-3.0，语言包含 C++、CUDA、C、Makefile。

优点：

- 与本项目 GPU 目标最接近。
- GPU 端任务组织和 output array 设计可直接借鉴。
- README 已经列出 early exit：进入黑洞、离开区域、命中任务目标。

风险：

- GPL-3.0 许可证限制代码复用。
- 项目较旧，构建环境可能需要修补。
- 自适应步长在 GPU 上带来线程分歧，本项目需要先做更容易验证的固定步长 MVP。

使用建议：

- 不复制代码。
- 用其结构设计本项目 `cuda/kernels.cu` 和 `src/gpu_trace.py` 的接口。

### 3.5 grtrans

证据：

- GitHub README 说明 grtrans performs polarized GR radiative transfer via ray tracing。
- README 说明它使用 geokerr，并要求引用 Dexter 2016 和 Dexter & Agol 2009。
- 提供 Python quick start 和 `run_grtrans_test_problems_public.py`。
- 仓库许可证为 MIT。

优点：

- 许可证宽松。
- 有 thin disk、jet、HARM 等示例问题。
- 适合学习参数组织、debug maps 和测试问题。

风险：

- Fortran + CFITSIO + Python 旧依赖，Windows 上复现可能不顺。
- 功能重点是 polarized GRRT，超过本项目第一阶段范围。

使用建议：

- 后续可在 Linux/Docker 里运行至少一个 THINDISK 示例。
- 使用其测试问题命名和输出 map 思路，但不把偏振作为 MVP 要求。

### 3.6 ipole

证据：

- GitHub README 说明 ipole 是 Polarized covariant radiative transfer in C，用于 EHT 类黑洞吸积系统。
- 依赖 HDF5 和 GSL。
- 许可证为 BSD-3-Clause。

优点：

- 适合后期 Stokes 参数和偏振输运扩展。
- BSD-3-Clause 对参考和接口设计较友好。

风险：

- 目标是偏振输运，不是薄盘可视化入门。
- 需要 HDF5/GSL 和源模型输入。

使用建议：

- 当前只放入 Related Work。
- 可选扩展阶段再实现 Stokes `I,Q,U,V` 接口。

### 3.7 Gargantua / DNGR

证据：

- DNGR 论文描述了 Interstellar 中 Gargantua 黑洞和吸积盘图像的生成方法。
- 重点是 ray bundle propagation、caustics、movie rendering 和视觉理解。

优点：

- 对报告中“科学可视化 vs 电影可视化”的讨论很有价值。
- ray bundle 思路可解释为什么高质量渲染不只是一条主光线。

风险：

- 生产源码不可作为开源依赖。
- 电影图像有审美和叙事取舍，不能当作科学 benchmark。

使用建议：

- 只作为 Related Work 和可视化讨论。

## 4. Demo 复现计划

优先尝试两个最能服务本项目验证的 demo：

1. **geokerr**
   - 目录：`research/repos/geokerr/`
   - 目标：编译 `geokerr.f`，运行 `abgrid.in`。
   - 输出：`research/reproduction_log.md`、`validation/geokerr_reference.csv`。
   - 风险：Fortran 编译器缺失。

2. **Odyssey 或 grtrans**
   - 若本机有 CUDA Toolkit：优先 Odyssey。
   - 若无 CUDA：用 Docker/Linux 环境尝试 grtrans 的 Python test problem。
   - 输出：构建日志、截图或输出图、依赖问题记录。

## 5. 许可证决策

建议本项目采用 MIT 或 BSD-3-Clause 许可证，但遵守以下规则：

- 不复制 GPL-3.0 项目 GYOTO/Odyssey 的代码。
- geokerr/YNOGK 只作为黑箱交叉验证或公式参考，除非许可证明确允许代码复用。
- MIT/BSD 项目 grtrans/ipole 也不直接复制核心实现，避免把旧 Fortran/C 依赖带入主线。

## 6. 本项目接口设计建议

根据上述审计，建议形成以下目录和接口：

```text
src/
  metric.py          # Kerr metric, inverse metric, horizon, ISCO
  geodesic.py        # state, RHS, event detection
  integrators.py     # RK4/RK45
  camera.py          # screen -> initial photon
  disk.py            # thin disk model, emissivity, NT profile
  render.py          # image assembly, tone mapping
  gpu_trace.py       # optional CuPy/Numba/CUDA wrapper and CPU fallback
cuda/
  kernels.cu         # CUDA MVP kernel sketch / implementation
tests/
  test_metric.py
  test_geodesic.py
  test_redshift.py
validation/
  error_summary.md
```

核心原则：

- CPU 和 GPU 输出格式必须一致：`intensity`、`redshift`、`hit_mask`、`rgb`。
- 每个外部项目只承担一个明确角色：geokerr/YNOGK 验证轨迹，Odyssey 验证 GPU 架构，grtrans/ipole 支撑 Related Work。
- 报告中必须区分“已复现 demo”和“已审计但未复现项目”。

## 7. 已确认来源

- GYOTO GitHub: https://github.com/gyoto/Gyoto
- geokerr page: https://faculty.washington.edu/agol/geokerr/index.html
- YNOGK paper: https://arxiv.org/abs/1305.1250
- Odyssey GitHub: https://github.com/hungyipu/Odyssey
- Odyssey paper: https://arxiv.org/abs/1601.02063
- grtrans GitHub: https://github.com/jadexter/grtrans
- grtrans paper: https://arxiv.org/abs/1602.03184
- ipole GitHub: https://github.com/AFD-Illinois/ipole
- ipole paper: https://arxiv.org/abs/1712.03057
- DNGR/Gargantua paper: https://arxiv.org/abs/1502.03808

