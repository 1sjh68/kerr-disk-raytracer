from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .camera import screen_grid
from .config import MAX_GEODESIC_RESOLUTION, load_config, validate_resolution
from .disk import (
    disk_inner_radius,
    doppler_factor,
    emissivity_power_law,
    novikov_thorne_flux,
    reinhard_tone_map,
    temperature_to_rgb,
    temperature_to_rgb_dispatch,
)
from .geodesic import trace_single_ray


def render_thin_disk_fast(config: dict[str, Any] | None = None, resolution: int | None = None) -> dict[str, np.ndarray | float | str]:
    """Vectorized MVP thin-disk render.

    This is a fast screen-space model used for reproducible project plumbing.
    The metric/geodesic modules provide the high-precision replacement path.
    """

    cfg = load_config() if config is None else config
    a = float(cfg["black_hole"]["spin"])
    inclination_deg = float(cfg["camera"]["inclination_deg"])
    inclination = math.radians(inclination_deg)
    fov_m = float(cfg["camera"]["fov_m"])
    res = validate_resolution(resolution or cfg["render"]["resolution"], "resolution")
    q = float(cfg["disk"]["emissivity_index"])
    model = str(cfg["disk"].get("model", "power_law"))
    r_outer = float(cfg["disk"]["outer_radius"])
    r_inner = disk_inner_radius(a, cfg["disk"].get("inner_radius", "isco"))

    t0 = time.perf_counter()
    alpha, beta = screen_grid(res, fov_m)
    cosi = max(0.12, abs(math.cos(inclination)))
    shadow_radius = 3.0 * math.sqrt(3.0) * (1.0 - 0.035 * a * math.cos(inclination))
    shadow_x = -2.0 * a * math.sin(inclination)
    shadow_y_scale = 1.0 + 0.08 * abs(a) * math.sin(inclination)
    b_shadow = np.sqrt((alpha - shadow_x) ** 2 + (beta * shadow_y_scale) ** 2)
    captured = b_shadow < shadow_radius

    x_disk = alpha + 0.18 * a * beta / max(shadow_radius, 1.0)
    y_disk = beta / cosi
    r_disk = np.sqrt(x_disk * x_disk + y_disk * y_disk)
    phi_disk = np.arctan2(y_disk, x_disk)
    disk_mask = (r_disk >= r_inner) & (r_disk <= r_outer) & (~captured)

    if model == "novikov_thorne":
        emitted = novikov_thorne_flux(r_disk, r_inner)
    else:
        emitted = emissivity_power_law(r_disk, q=q)
    grav = np.sqrt(np.clip(1.0 - 2.0 / np.maximum(r_disk, 2.05), 0.04, 1.0))
    doppler = doppler_factor(r_disk, phi_disk, inclination, a)
    redshift = np.clip(grav * doppler, 0.05, 3.5)
    intensity = np.where(disk_mask, emitted * redshift**3, 0.0)

    ring_width = 0.45 + 0.12 * abs(a)
    photon_ring = np.exp(-((b_shadow - shadow_radius * 1.035) / ring_width) ** 2)
    ring_phi = np.arctan2(beta / cosi, alpha - shadow_x)
    ring_doppler = doppler_factor(np.maximum(r_inner + 0.7, shadow_radius), ring_phi, inclination, a)
    ring = 0.19 * photon_ring * ring_doppler * (b_shadow >= shadow_radius * 0.96)
    ring *= np.clip((r_outer - r_inner) / max(r_outer, 1.0), 0.15, 1.0)
    intensity += ring * np.max(emitted[disk_mask]) if np.any(disk_mask) else ring
    redshift = np.where(intensity > 0.0, np.maximum(redshift, np.clip(ring_doppler * 0.45, 0.05, 2.5)), 0.0)

    secondary = np.exp(-((b_shadow - shadow_radius * 1.35) / 0.85) ** 2)
    secondary *= (beta > -0.25 * shadow_radius)
    secondary *= np.clip(abs(math.sin(inclination)), 0.0, 1.0)
    intensity += secondary * 0.055 * (r_inner ** -q) * np.clip(ring_doppler, 0.2, 2.0)

    hit_mask = intensity > 0.0
    redshift = np.where(hit_mask & (redshift <= 0.0), 0.55, redshift)
    temperature = np.where(hit_mask, np.maximum(intensity, 0.0) ** 0.25, 0.0)
    color_mode = str(cfg["render"].get("color_mode", "approx"))
    cie_t_min = float(cfg["render"].get("cie_t_min", 3000.0))
    cie_t_max = float(cfg["render"].get("cie_t_max", 30000.0))
    base_rgb = temperature_to_rgb_dispatch(
        temperature, np.where(hit_mask, redshift, 1.0),
        mode=color_mode, cie_t_min=cie_t_min, cie_t_max=cie_t_max,
    )
    rgb_linear = base_rgb * intensity[..., None]
    rgb = reinhard_tone_map(
        rgb_linear / max(float(np.percentile(rgb_linear, 99.5)), 1.0e-12),
        gamma=float(cfg["render"].get("gamma", 2.2)),
        bloom=bool(cfg["render"].get("bloom", True)),
    )
    rgb[captured] *= 0.0
    elapsed = time.perf_counter() - t0
    return {
        "intensity": intensity.astype(np.float32),
        "redshift": redshift.astype(np.float32),
        "temperature": temperature.astype(np.float32),
        "hit_mask": hit_mask.astype(np.uint8),
        "rgb": rgb.astype(np.float32),
        "elapsed_s": float(elapsed),
        "method": "fast_thin_disk_screen_model",
        "resolution": res,
        "spin": a,
        "inclination_deg": inclination_deg,
        "inner_radius": float(r_inner),
        "outer_radius": float(r_outer),
    }


STATUS_CODES = {
    "disk": 1,
    "captured": 2,
    "escaped": 3,
    "max_steps": 4,
    "invalid": 5,
}


def render_thin_disk_geodesic_cpu(
    config: dict[str, Any] | None = None,
    resolution: int | None = None,
) -> dict[str, np.ndarray | float | str]:
    """Trace one Hamiltonian geodesic per pixel.

    This renderer is intentionally kept at low default resolution because the
    current RHS uses finite-difference metric derivatives. It provides a real
    geodesic reference artifact while the faster production renderer is still
    being replaced.
    """

    cfg = load_config() if config is None else config
    a = float(cfg["black_hole"]["spin"])
    inclination_deg = float(cfg["camera"]["inclination_deg"])
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
    res = validate_resolution(
        resolution or cfg["render"].get("geodesic_resolution", 32),
        "geodesic_resolution",
        max_value=MAX_GEODESIC_RESOLUTION,
    )

    t0 = time.perf_counter()
    alpha, beta = screen_grid(res, fov_m)
    intensity = np.zeros((res, res), dtype=np.float32)
    redshift = np.zeros((res, res), dtype=np.float32)
    temperature = np.zeros((res, res), dtype=np.float32)
    hit_mask = np.zeros((res, res), dtype=np.uint8)
    status_code = np.zeros((res, res), dtype=np.uint8)
    radius = np.zeros((res, res), dtype=np.float32)
    steps = np.zeros((res, res), dtype=np.uint16)
    null_error = np.zeros((res, res), dtype=np.float32)

    for iy in range(res):
        for ix in range(res):
            hit = trace_single_ray(
                float(alpha[iy, ix]),
                float(beta[iy, ix]),
                a=a,
                inclination_deg=inclination_deg,
                r_obs=r_obs,
                disk_inner=r_inner,
                disk_outer=r_outer,
                emissivity_index=q,
                emission_model=model,
                step_size=step_size,
                max_steps=max_steps,
                horizon_epsilon=horizon_epsilon,
                escape_radius=escape_radius,
            )
            code = STATUS_CODES.get(hit.status, 0)
            status_code[iy, ix] = code
            steps[iy, ix] = min(hit.steps, np.iinfo(np.uint16).max)
            if hit.radius is not None:
                radius[iy, ix] = hit.radius
            safe_null = hit.null_error if np.isfinite(hit.null_error) else 1.0e6
            null_error[iy, ix] = min(float(safe_null), 1.0e6)
            if hit.status == "disk":
                intensity[iy, ix] = hit.intensity
                redshift[iy, ix] = hit.redshift
                temperature[iy, ix] = hit.temperature
                hit_mask[iy, ix] = 1

    color_mode = str(cfg["render"].get("color_mode", "approx"))
    cie_t_min = float(cfg["render"].get("cie_t_min", 3000.0))
    cie_t_max = float(cfg["render"].get("cie_t_max", 30000.0))
    base_rgb = temperature_to_rgb_dispatch(
        temperature, np.where(hit_mask, redshift, 1.0),
        mode=color_mode, cie_t_min=cie_t_min, cie_t_max=cie_t_max,
    )
    rgb_linear = base_rgb * intensity[..., None]
    rgb = reinhard_tone_map(
        rgb_linear / max(float(np.percentile(rgb_linear, 99.5)), 1.0e-12),
        gamma=float(cfg["render"].get("gamma", 2.2)),
        bloom=bool(cfg["render"].get("bloom", True)),
    )
    elapsed = time.perf_counter() - t0
    return {
        "intensity": intensity,
        "redshift": redshift,
        "temperature": temperature,
        "hit_mask": hit_mask,
        "rgb": rgb.astype(np.float32),
        "status_code": status_code,
        "radius": radius,
        "steps": steps,
        "null_error": null_error,
        "elapsed_s": float(elapsed),
        "method": "hamiltonian_geodesic_cpu",
        "resolution": res,
        "spin": a,
        "inclination_deg": inclination_deg,
        "inner_radius": float(r_inner),
        "outer_radius": float(r_outer),
        "step_size": step_size,
        "max_steps": max_steps,
    }


def save_png(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(rgb, 0.0, 1.0))


def upscale_rgb(rgb: np.ndarray, min_size: int = 384) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = max(1, int(math.ceil(min_size / max(height, width))))
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def save_maps_npz(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in data.items()
        if isinstance(value, np.ndarray)
        and key in {"intensity", "redshift", "temperature", "hit_mask", "rgb", "status_code", "radius", "steps", "null_error"}
    }
    np.savez_compressed(path, **arrays)


def save_maps_npy(prefix: str | Path, data: dict[str, Any]) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        if (
            isinstance(value, np.ndarray)
            and key in {"intensity", "redshift", "temperature", "hit_mask", "rgb", "status_code", "radius", "steps", "null_error"}
        ):
            np.save(prefix.with_name(f"{prefix.name}_{key}.npy"), value)


def metadata_from_render(data: dict[str, Any]) -> dict[str, Any]:
    meta = {
        "method": data["method"],
        "resolution": int(data["resolution"]),
        "spin": float(data["spin"]),
        "inclination_deg": float(data["inclination_deg"]),
        "inner_radius": float(data["inner_radius"]),
        "outer_radius": float(data["outer_radius"]),
        "elapsed_s": float(data["elapsed_s"]),
        "intensity_max": float(np.max(data["intensity"])),
        "redshift_min": float(np.min(data["redshift"][data["hit_mask"] > 0])) if np.any(data["hit_mask"]) else 0.0,
        "redshift_max": float(np.max(data["redshift"])),
        "hit_fraction": float(np.mean(data["hit_mask"] > 0)),
    }
    if "status_code" in data:
        status = np.asarray(data["status_code"])
        meta["status_counts"] = {
            name: int(np.sum(status == code))
            for name, code in STATUS_CODES.items()
        }
    if "null_error" in data:
        null_error = np.asarray(data["null_error"], dtype=float)
        meta["null_error_max"] = float(np.max(null_error))
        meta["null_error_mean"] = float(np.mean(null_error))
        if "status_code" in data:
            disk_mask = np.asarray(data["status_code"]) == STATUS_CODES["disk"]
            if np.any(disk_mask):
                meta["disk_null_error_max"] = float(np.max(null_error[disk_mask]))
                meta["disk_null_error_mean"] = float(np.mean(null_error[disk_mask]))
    if "step_size" in data:
        meta["step_size"] = float(data["step_size"])
    if "max_steps" in data:
        meta["max_steps"] = int(data["max_steps"])
    if "gpu_kernel_elapsed_s" in data:
        meta["gpu_kernel_elapsed_s"] = float(data["gpu_kernel_elapsed_s"])
    if "cuda_block" in data:
        meta["cuda_block"] = str(data["cuda_block"])
    return meta


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
