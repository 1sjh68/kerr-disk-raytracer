"""Minimal float64 kernel launcher for ncu profiling."""
import numpy as np
from src.config import load_config
from src.gpu_trace import render_cuda_geodesic

cfg = load_config()
# Warm-up compilation
render_cuda_geodesic(cfg, resolution=48, precision="float64")
# Actual profile target: higher res = longer kernel = easier to capture
render_cuda_geodesic(cfg, resolution=128, precision="float64")
