"""Stress test: 1024x1024 float64 geodesic."""
import time
import copy

from src.config import DEFAULT_CONFIG
from src.gpu_trace import render_cuda_geodesic

cfg = copy.deepcopy(DEFAULT_CONFIG)
cfg["render"]["geodesic_resolution"] = 1024

t0 = time.perf_counter()
data = render_cuda_geodesic(cfg, resolution=1024, precision="float64")
elapsed = time.perf_counter() - t0

print(f"1024x1024 float64: kernel={data['gpu_kernel_elapsed_s']:.4f}s  wall={elapsed:.4f}s")
print(f"hit_fraction={data['hit_mask'].mean():.4f}  captured={(data['status_code']==2).sum()}  escaped={(data['status_code']==3).sum()}")

# Save the image for reference
from src.render import save_png, upscale_rgb
preview = upscale_rgb(data["rgb"])
save_png("figures/gpu_geodesic_1024_float64.png", preview)
print("Saved figures/gpu_geodesic_1024_float64.png")
