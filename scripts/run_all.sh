#!/usr/bin/env bash
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONPATH="${PYTHONPATH:-$PWD}"

if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="${PYTHON}"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="python.exe"
elif [ -x "/mnt/host/c/Users/oo/AppData/Local/Programs/Python/Python312/python.exe" ]; then
  PYTHON_BIN="/mnt/host/c/Users/oo/AppData/Local/Programs/Python/Python312/python.exe"
else
  PYTHON_BIN="python"
fi

# Helper: run a step but don't kill the pipeline if it fails (used for new
# demos which may be GPU-only or matplotlib-only).
soft_run() {
  local label="$1"; shift
  if "$@"; then
    return 0
  fi
  echo "[skip] ${label}: command failed (non-fatal)"
  return 0
}

# === Core CPU/GPU pipelines =============================================
"${PYTHON_BIN}" run_cpu.py
"${PYTHON_BIN}" run_geodesic_cpu.py
"${PYTHON_BIN}" scripts/check_cuda.py
"${PYTHON_BIN}" run_gpu.py
"${PYTHON_BIN}" run_geodesic_gpu.py --precision float32
"${PYTHON_BIN}" run_geodesic_gpu.py --precision float64
"${PYTHON_BIN}" render.py

# === Benchmarks & validation ============================================
"${PYTHON_BIN}" benchmark.py
"${PYTHON_BIN}" scripts/run_experiments.py
"${PYTHON_BIN}" validate.py
"${PYTHON_BIN}" validate_geodesic.py
"${PYTHON_BIN}" scripts/summarize_external_demos.py
"${PYTHON_BIN}" scripts/parse_geokerr.py
"${PYTHON_BIN}" scripts/make_report.py

# === New (May 2026) — Phase 8/9/10/12 closures =========================
soft_run "benchmark_fastmath"        "${PYTHON_BIN}" tools/benchmark_fastmath.py
soft_run "disk_param_sweep"          "${PYTHON_BIN}" tools/disk_param_sweep.py
soft_run "render_color_compare"      "${PYTHON_BIN}" tools/render_color_compare.py
soft_run "geokerr_coordinate_compare" "${PYTHON_BIN}" tools/geokerr_coordinate_compare.py
soft_run "rk45_vs_rk4_demo"          "${PYTHON_BIN}" tools/rk45_vs_rk4_demo.py
soft_run "eht_metrics_demo"          "${PYTHON_BIN}" tools/eht_metrics_demo.py
soft_run "polarization_demo"         "${PYTHON_BIN}" tools/polarization_demo.py
soft_run "compose_animations"        "${PYTHON_BIN}" tools/compose_animations.py

# === Tests ==============================================================
"${PYTHON_BIN}" -m pytest tests -q

# === PPT build (optional, requires Node on Windows host) ===============
if command -v node >/dev/null 2>&1; then
  node paper/presentation/build/build_deck.mjs || \
    echo "PPT deck build skipped in this shell; use scripts/run_all.ps1 on Windows."
fi
