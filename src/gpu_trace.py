from __future__ import annotations

import math
import os
import site
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .camera import screen_grid
from .config import MAX_GEODESIC_RESOLUTION, validate_cuda_block, validate_resolution
from .disk import (
    disk_inner_radius,
    emissivity_power_law,
    novikov_thorne_flux,
    reinhard_tone_map,
    temperature_to_rgb,
)
from .render import render_thin_disk_fast


@lru_cache(maxsize=1)
def configure_cuda_wheel_paths() -> dict[str, Any]:
    """Expose NVIDIA CUDA wheel DLL directories to Windows loaders.

    The project can be prepared with PyPI CUDA runtime/NVRTC wheels even when a
    system CUDA Toolkit is not installed. CuPy still needs those DLL directories
    on PATH before its CUDA backend is exercised.
    """

    candidates: list[Path] = []
    for raw in [*site.getsitepackages(), str(Path(sys.prefix) / "Lib" / "site-packages")]:
        path = Path(raw)
        if path.exists() and path not in candidates:
            candidates.append(path)

    bin_paths: list[str] = []
    cuda_roots: dict[str, str] = {}
    for base in candidates:
        nvidia_root = base / "nvidia"
        for package in ["cuda_runtime", "cuda_nvrtc", "cuda_nvcc"]:
            package_root = nvidia_root / package
            bin_path = package_root / "bin"
            if package_root.exists():
                cuda_roots[package] = str(package_root)
            if bin_path.exists():
                bin_str = str(bin_path)
                bin_paths.append(bin_str)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_str)
                    except OSError:
                        pass

    if "CUDA_PATH" not in os.environ and "cuda_runtime" in cuda_roots:
        os.environ["CUDA_PATH"] = cuda_roots["cuda_runtime"]

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    for bin_str in reversed(bin_paths):
        if bin_str not in path_parts:
            path_parts.insert(0, bin_str)
    os.environ["PATH"] = os.pathsep.join(path_parts)
    return {"cuda_roots": cuda_roots, "bin_paths": bin_paths, "cuda_path": os.environ.get("CUDA_PATH", "")}


def cuda_available() -> tuple[bool, str]:
    try:
        configure_cuda_wheel_paths()
        import cupy as cp  # type: ignore

        count = cp.cuda.runtime.getDeviceCount()
        if count <= 0:
            return False, "cupy_cuda_devices=0"
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return True, f"cupy_cuda_devices={count}; device0={name}"
    except Exception as exc:
        return False, f"cuda_unavailable: {exc.__class__.__name__}: {exc}"


@lru_cache(maxsize=1)
def _cuda_kernel_source() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "cuda" / "kernels.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _cuda_module() -> Any:
    import cupy as cp  # type: ignore
    return cp.RawModule(code=_cuda_kernel_source(), options=("--std=c++11",))


@lru_cache(maxsize=1)
def _cuda_module_fastmath() -> Any:
    """Same kernel source compiled with --use_fast_math.

    Replaces sinf/cosf/sqrtf with their __sinf/__cosf/rsqrtf intrinsic
    counterparts (lower precision, higher throughput). Use with the
    float32 kernel only; the float64 path is unaffected because
    intrinsics are FP32-only.
    """
    import cupy as cp  # type: ignore
    return cp.RawModule(
        code=_cuda_kernel_source(),
        options=("--std=c++11", "--use_fast_math"),
    )


def _emission_model_id(model: str) -> int:
    return 1 if model == "novikov_thorne" else 0


def _emitted_scale(config: dict[str, Any], resolution: int, r_inner: float) -> float:
    a = float(config["black_hole"]["spin"])
    inclination = math.radians(float(config["camera"]["inclination_deg"]))
    fov_m = float(config["camera"]["fov_m"])
    q = float(config["disk"]["emissivity_index"])
    model = str(config["disk"].get("model", "power_law"))
    r_outer = float(config["disk"]["outer_radius"])

    alpha, beta = screen_grid(resolution, fov_m)
    cosi = max(0.12, abs(math.cos(inclination)))
    shadow_radius = 3.0 * math.sqrt(3.0) * (1.0 - 0.035 * a * math.cos(inclination))
    shadow_x = -2.0 * a * math.sin(inclination)
    shadow_y_scale = 1.0 + 0.08 * abs(a) * math.sin(inclination)
    b_shadow = np.sqrt((alpha - shadow_x) ** 2 + (beta * shadow_y_scale) ** 2)
    captured = b_shadow < shadow_radius
    x_disk = alpha + 0.18 * a * beta / max(shadow_radius, 1.0)
    y_disk = beta / cosi
    r_disk = np.sqrt(x_disk * x_disk + y_disk * y_disk)
    disk_mask = (r_disk >= r_inner) & (r_disk <= r_outer) & (~captured)
    if model == "novikov_thorne":
        emitted = novikov_thorne_flux(r_disk, r_inner)
    else:
        emitted = emissivity_power_law(r_disk, q=q)
    return float(np.max(emitted[disk_mask])) if np.any(disk_mask) else 1.0


def _captured_mask(config: dict[str, Any], resolution: int) -> np.ndarray:
    a = float(config["black_hole"]["spin"])
    inclination = math.radians(float(config["camera"]["inclination_deg"]))
    fov_m = float(config["camera"]["fov_m"])
    alpha, beta = screen_grid(resolution, fov_m)
    shadow_radius = 3.0 * math.sqrt(3.0) * (1.0 - 0.035 * a * math.cos(inclination))
    shadow_x = -2.0 * a * math.sin(inclination)
    shadow_y_scale = 1.0 + 0.08 * abs(a) * math.sin(inclination)
    return np.sqrt((alpha - shadow_x) ** 2 + (beta * shadow_y_scale) ** 2) < shadow_radius


def _cuda_block_from_config(config: dict[str, Any], block: tuple[int, int] | None = None) -> tuple[int, int]:
    if block is not None:
        return validate_cuda_block(block)
    raw = config.get("render", {}).get("cuda_block", [16, 16])
    return validate_cuda_block(raw)


def render_cuda(
    config: dict[str, Any],
    resolution: int | None = None,
    block: tuple[int, int] | None = None,
) -> dict[str, Any]:
    configure_cuda_wheel_paths()
    import cupy as cp  # type: ignore

    cfg = config
    res = validate_resolution(resolution or cfg["render"]["resolution"], "resolution")
    a = float(cfg["black_hole"]["spin"])
    inclination = math.radians(float(cfg["camera"]["inclination_deg"]))
    fov_m = float(cfg["camera"]["fov_m"])
    q = float(cfg["disk"]["emissivity_index"])
    model = str(cfg["disk"].get("model", "power_law"))
    r_outer = float(cfg["disk"]["outer_radius"])
    r_inner = disk_inner_radius(a, cfg["disk"].get("inner_radius", "isco"))
    emitted_scale = _emitted_scale(cfg, res, r_inner)

    module = _cuda_module()
    kernel = module.get_function("kerr_thin_disk_kernel")
    intensity = cp.zeros((res, res), dtype=cp.float32)
    redshift = cp.zeros((res, res), dtype=cp.float32)
    temperature = cp.zeros((res, res), dtype=cp.float32)
    hit_mask = cp.zeros((res, res), dtype=cp.uint8)

    cuda_block = _cuda_block_from_config(cfg, block=block)
    grid = ((res + cuda_block[0] - 1) // cuda_block[0], (res + cuda_block[1] - 1) // cuda_block[1])
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    t0 = time.perf_counter()
    start_event.record()
    kernel(
        grid,
        cuda_block,
        (
            np.int32(res),
            np.int32(res),
            np.float32(a),
            np.float32(inclination),
            np.float32(fov_m),
            np.float32(r_inner),
            np.float32(r_outer),
            np.float32(q),
            np.int32(_emission_model_id(model)),
            np.float32(emitted_scale),
            intensity,
            redshift,
            temperature,
            hit_mask,
        ),
    )
    end_event.record()
    end_event.synchronize()
    kernel_elapsed_s = float(cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0)

    intensity_np = cp.asnumpy(intensity).astype(np.float32, copy=False)
    redshift_np = cp.asnumpy(redshift).astype(np.float32, copy=False)
    temperature_np = cp.asnumpy(temperature).astype(np.float32, copy=False)
    hit_mask_np = cp.asnumpy(hit_mask).astype(np.uint8, copy=False)
    base_rgb = temperature_to_rgb(temperature_np, np.where(hit_mask_np, redshift_np, 1.0))
    rgb_linear = base_rgb * intensity_np[..., None]
    rgb = reinhard_tone_map(
        rgb_linear / max(float(np.percentile(rgb_linear, 99.5)), 1.0e-12),
        gamma=float(cfg["render"].get("gamma", 2.2)),
        bloom=bool(cfg["render"].get("bloom", True)),
    )
    rgb[_captured_mask(cfg, res)] *= 0.0
    available, reason = cuda_available()
    return {
        "intensity": intensity_np,
        "redshift": redshift_np,
        "temperature": temperature_np,
        "hit_mask": hit_mask_np,
        "rgb": rgb.astype(np.float32),
        "elapsed_s": float(time.perf_counter() - t0),
        "gpu_kernel_elapsed_s": kernel_elapsed_s,
        "method": "cuda_fast_thin_disk_rawkernel",
        "resolution": res,
        "spin": a,
        "inclination_deg": float(cfg["camera"]["inclination_deg"]),
        "inner_radius": float(r_inner),
        "outer_radius": float(r_outer),
        "cuda_block": f"{cuda_block[0]}x{cuda_block[1]}",
        "backend": "cuda" if available else "cuda_unknown",
        "backend_reason": reason,
    }


def render_cuda_geodesic(
    config: dict[str, Any],
    resolution: int | None = None,
    block: tuple[int, int] | None = None,
    precision: str = "float32",
    fast_math: bool = False,
) -> dict[str, Any]:
    """CuPy RawKernel Hamiltonian geodesic path (one thread per pixel)."""

    configure_cuda_wheel_paths()
    import cupy as cp  # type: ignore

    cfg = config
    res = validate_resolution(
        resolution or cfg["render"].get("geodesic_resolution", 48),
        "geodesic_resolution",
        max_value=MAX_GEODESIC_RESOLUTION,
    )
    a = float(cfg["black_hole"]["spin"])
    inclination = math.radians(float(cfg["camera"]["inclination_deg"]))
    r_obs = float(cfg["camera"]["r_obs"])
    fov_m = float(cfg["camera"]["fov_m"])
    q = float(cfg["disk"]["emissivity_index"])
    model = str(cfg["disk"].get("model", "power_law"))
    r_outer = float(cfg["disk"]["outer_radius"])
    r_inner = disk_inner_radius(a, cfg["disk"].get("inner_radius", "isco"))
    step_size = float(cfg["integration"]["step_size"])
    max_steps = int(cfg["integration"]["max_steps"])
    horizon_epsilon = float(cfg["integration"]["horizon_epsilon"])
    escape_radius = float(cfg["integration"]["escape_radius"])

    module = _cuda_module_fastmath() if fast_math else _cuda_module()
    kernel_name = {
        "float64": "kerr_geodesic_kernel_double",
        "float64_opt": "kerr_geodesic_kernel_double_opt",
    }.get(precision, "kerr_geodesic_kernel")
    kernel = module.get_function(kernel_name)
    intensity = cp.zeros((res, res), dtype=cp.float32)
    redshift = cp.zeros((res, res), dtype=cp.float32)
    temperature = cp.zeros((res, res), dtype=cp.float32)
    hit_mask = cp.zeros((res, res), dtype=cp.uint8)
    status_code = cp.zeros((res, res), dtype=cp.uint8)
    null_error = cp.zeros((res, res), dtype=cp.float32)

    cuda_block = _cuda_block_from_config(cfg, block=block)
    grid = ((res + cuda_block[0] - 1) // cuda_block[0], (res + cuda_block[1] - 1) // cuda_block[1])
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    t0 = time.perf_counter()
    start_event.record()
    kernel(
        grid,
        cuda_block,
        (
            np.int32(res),
            np.int32(res),
            np.float32(a),
            np.float32(inclination),
            np.float32(fov_m),
            np.float32(r_inner),
            np.float32(r_outer),
            np.float32(q),
            np.int32(_emission_model_id(model)),
            np.float32(r_obs),
            np.float32(step_size),
            np.int32(max_steps),
            np.float32(horizon_epsilon),
            np.float32(escape_radius),
            intensity,
            redshift,
            temperature,
            hit_mask,
            status_code,
            null_error,
        ),
    )
    end_event.record()
    end_event.synchronize()
    kernel_elapsed_s = float(cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0)

    intensity_np = cp.asnumpy(intensity).astype(np.float32, copy=False)
    redshift_np = cp.asnumpy(redshift).astype(np.float32, copy=False)
    temperature_np = cp.asnumpy(temperature).astype(np.float32, copy=False)
    hit_mask_np = cp.asnumpy(hit_mask).astype(np.uint8, copy=False)
    status_code_np = cp.asnumpy(status_code).astype(np.uint8, copy=False)
    null_error_np = cp.asnumpy(null_error).astype(np.float32, copy=False)
    # Clamp spurious float32 overflows / inf-inf NaNs near the horizon to a
    # finite sentinel so downstream statistics remain meaningful.
    null_error_np = np.where(np.isfinite(null_error_np), null_error_np, 1.0e6)
    null_error_np = np.clip(null_error_np, 0.0, 1.0e6)

    base_rgb = temperature_to_rgb(temperature_np, np.where(hit_mask_np, redshift_np, 1.0))
    rgb_linear = base_rgb * intensity_np[..., None]
    rgb = reinhard_tone_map(
        rgb_linear / max(float(np.percentile(rgb_linear, 99.5)), 1.0e-12),
        gamma=float(cfg["render"].get("gamma", 2.2)),
        bloom=bool(cfg["render"].get("bloom", True)),
    )
    available, reason = cuda_available()
    return {
        "intensity": intensity_np,
        "redshift": redshift_np,
        "temperature": temperature_np,
        "hit_mask": hit_mask_np,
        "rgb": rgb.astype(np.float32),
        "status_code": status_code_np,
        "null_error": null_error_np,
        "elapsed_s": float(time.perf_counter() - t0),
        "gpu_kernel_elapsed_s": kernel_elapsed_s,
        "method": "cuda_hamiltonian_geodesic_rawkernel",
        "resolution": res,
        "spin": a,
        "inclination_deg": float(cfg["camera"]["inclination_deg"]),
        "inner_radius": float(r_inner),
        "outer_radius": float(r_outer),
        "step_size": step_size,
        "max_steps": max_steps,
        "cuda_block": f"{cuda_block[0]}x{cuda_block[1]}",
        "backend": "cuda" if available else "cuda_unknown",
        "backend_reason": reason,
    }


def render_gpu_or_fallback(config: dict[str, Any], resolution: int | None = None) -> dict[str, Any]:
    """GPU entry point with a deterministic CPU fallback."""

    available, reason = cuda_available()
    if available:
        try:
            return render_cuda(config, resolution=resolution)
        except Exception as exc:
            reason = f"cuda_kernel_failed: {exc.__class__.__name__}: {exc}"

    data = render_thin_disk_fast(config, resolution=resolution)
    data["backend"] = "cpu_fallback"
    data["backend_reason"] = reason
    for key in ["intensity", "redshift", "temperature", "rgb"]:
        data[key] = np.asarray(data[key], dtype=np.float32)
    data["hit_mask"] = np.asarray(data["hit_mask"], dtype=np.uint8)
    return data


def render_gpu_or_fallback_geodesic(
    config: dict[str, Any], resolution: int | None = None, precision: str = "float32"
) -> dict[str, Any]:
    """Geodesic GPU entry point with deterministic CPU fallback."""

    from .render import render_thin_disk_geodesic_cpu

    available, reason = cuda_available()
    if available:
        try:
            return render_cuda_geodesic(config, resolution=resolution, precision=precision)
        except Exception as exc:
            reason = f"cuda_geodesic_kernel_failed: {exc.__class__.__name__}: {exc}"

    data = render_thin_disk_geodesic_cpu(config, resolution=resolution)
    data["backend"] = "cpu_fallback"
    data["backend_reason"] = reason
    for key in ["intensity", "redshift", "temperature", "rgb", "null_error"]:
        if key in data:
            data[key] = np.asarray(data[key], dtype=np.float32)
    for key in ["hit_mask", "status_code"]:
        if key in data:
            data[key] = np.asarray(data[key], dtype=np.uint8)
    return data
