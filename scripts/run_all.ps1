$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Get-Location).Path

$PythonBin = if ($env:PYTHON) {
    $env:PYTHON
} elseif (Test-Path ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} else {
    "python"
}

# === Core CPU/GPU pipelines ============================================
& $PythonBin run_cpu.py
& $PythonBin run_geodesic_cpu.py
& $PythonBin scripts/check_cuda.py
& $PythonBin run_gpu.py
& $PythonBin run_geodesic_gpu.py --precision float32
& $PythonBin run_geodesic_gpu.py --precision float64
& $PythonBin render.py

# === Benchmarks & validation ===========================================
& $PythonBin benchmark.py
& $PythonBin scripts/run_experiments.py
& $PythonBin validate.py
& $PythonBin validate_geodesic.py
& $PythonBin scripts/summarize_external_demos.py
& $PythonBin scripts/parse_geokerr.py
& $PythonBin scripts/make_report.py

# === New (May 2026) — Phase 8/9/10/12 closures ==========================
# Optional; safe to skip on slow/CPU-only hosts.
try {
    & $PythonBin tools/benchmark_fastmath.py            # T6: --use_fast_math 3x speedup
} catch { Write-Host "[skip] benchmark_fastmath.py: $_" }

try {
    & $PythonBin tools/disk_param_sweep.py              # T3: r_out + emissivity sweep
} catch { Write-Host "[skip] disk_param_sweep.py: $_" }

try {
    & $PythonBin tools/render_color_compare.py          # T4: CIE1931 vs approx
} catch { Write-Host "[skip] render_color_compare.py: $_" }

try {
    & $PythonBin tools/geokerr_coordinate_compare.py    # T7: coord-level alignment
} catch { Write-Host "[skip] geokerr_coordinate_compare.py: $_" }

try {
    & $PythonBin tools/rk45_vs_rk4_demo.py              # T7 follow-up
} catch { Write-Host "[skip] rk45_vs_rk4_demo.py: $_" }

try {
    & $PythonBin tools/eht_metrics_demo.py              # T11 EHT metrics
} catch { Write-Host "[skip] eht_metrics_demo.py: $_" }

try {
    & $PythonBin tools/polarization_demo.py             # T10 polarization stub
} catch { Write-Host "[skip] polarization_demo.py: $_" }

try {
    & $PythonBin tools/compose_animations.py            # T12 sweep GIFs
} catch { Write-Host "[skip] compose_animations.py: $_" }

# === Tests =============================================================
& $PythonBin -m pytest tests -q

# === PPT build (optional, Windows host needs Node) ======================
if (Get-Command node -ErrorAction SilentlyContinue) {
    node paper/presentation/build/build_deck.mjs
} else {
    Write-Host "node not found; skipping PPT deck build"
}
