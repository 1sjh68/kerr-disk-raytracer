# Expected Contributions

更新时间：2026-05-21（文档对齐）

1. A reproducible Kerr thin-disk rendering pipeline with a clear CPU/GPU data contract across **two paths**: fast thin-disk MVP and full Hamiltonian geodesic tracing.
2. A validation harness that reports metric tests, null geodesic tests, redshift tests, image-map MSE/MAE/max error, hit-mask mismatch, and **geodesic CPU/GPU status agreement** (`validate_geodesic.py`).
3. A Hamiltonian geodesic CPU reference path (`run_geodesic_cpu.py`) and matching CUDA kernels (`kerr_geodesic_kernel` float32, `kerr_geodesic_kernel_double` float64) with **100% state match @ 48×48** for float64.
4. A staged CUDA implementation: deterministic CPU fallback, real RawKernel for fast thin-disk, and **completed** per-pixel geodesic kernel with precision-selectable output (`--precision float32|float64`).
5. Documentation that separates physical model, numerical method, dual-pipeline scope, validation and limitations.
6. A report package with final render, geodesic reference images (CPU + GPU float32/float64), error map, comparison grid, parameter/resolution sweeps, speed charts and source configuration.
