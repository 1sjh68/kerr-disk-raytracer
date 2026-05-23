"""Compute EHT-style metrics on existing geodesic-rendered intensity maps.

Loads reference/gpu_geodesic_reference_float64.npz and the 24 spin x
inclination sweep PNGs (via reading the saved JSON metadata), computes
ring diameter / asymmetry / photon-ring peak for each, and writes
results/eht_metrics.json + a markdown comparison table at
results/eht_metrics_report.md.

Note: this demo does NOT recompute geodesics. It reads previously
saved intensity maps. For new configurations rerun
tools/parameter_sweep.py first.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eht_metrics import summary


# Default config FOV is 24 M; the 48x48 reference uses geodesic_resolution=48
REFERENCE_FOV_M = 24.0


def load_reference() -> tuple[np.ndarray, dict]:
    npz = np.load("reference/gpu_geodesic_reference_float64.npz")
    intensity = np.asarray(npz["intensity"], dtype=float)
    return intensity, {
        "source": "reference/gpu_geodesic_reference_float64.npz",
        "resolution": int(intensity.shape[0]),
        "fov_m": REFERENCE_FOV_M,
        "spin": 0.7,
        "inclination_deg": 60.0,
    }


def load_parameter_sweep_intensities() -> list[dict]:
    """Use the per-config metadata in results/parameter_sweep.json to
    locate intensity arrays — but the sweep only saved the PNG previews
    and per-pixel intensity_max. We recompute summary metrics per-PNG
    using a luminance proxy (sufficient for ring diameter / asymmetry
    geometry; absolute intensity is not preserved through tone mapping)."""
    p = Path("results/parameter_sweep.json")
    if not p.exists():
        return []
    sweep = json.loads(p.read_text(encoding="utf-8"))
    out = []
    try:
        from PIL import Image
    except ImportError:
        return []
    for r in sweep:
        img_path = Path(r.get("image_path", ""))
        if not img_path.exists():
            continue
        img = np.asarray(Image.open(img_path).convert("L"), dtype=float) / 255.0
        out.append({
            "spin": r["spin"],
            "inclination_deg": r["inclination_deg"],
            "image_path": str(img_path),
            "metrics": summary(img, REFERENCE_FOV_M),
        })
    return out


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    print("=== EHT metrics: float64 reference (a=0.7, i=60deg) ===")
    intensity, meta = load_reference()
    ref_metrics = summary(intensity, REFERENCE_FOV_M)
    ref_metrics["meta"] = meta
    for k in ("ring_diameter_M", "photon_ring_peak_radius_M", "intensity_max"):
        print(f"  {k}: {ref_metrics[k]:.4g}")
    asym = ref_metrics["asymmetry"]
    print(f"  asymmetry south/north: {asym['south_over_north']:.3f}")

    print("\n=== Sweep PNG luminance metrics (24 configs) ===")
    sweep_metrics = load_parameter_sweep_intensities()
    for r in sweep_metrics:
        m = r["metrics"]
        print(f"  a={r['spin']:+.3f} i={r['inclination_deg']:.0f}deg  "
              f"D_ring={m['ring_diameter_M']:.2f}M  "
              f"south/north={m['asymmetry']['south_over_north']:.2f}  "
              f"peak_r={m['photon_ring_peak_radius_M']:.2f}M")

    out = {
        "reference": ref_metrics,
        "sweep": sweep_metrics,
    }
    with open("results/eht_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results/eht_metrics.json")

    # Also write a markdown comparison table
    md_lines = [
        "# EHT-Style Image Metrics",
        "",
        "更新时间：自动生成。详见 `src/eht_metrics.py` 与 `tools/eht_metrics_demo.py`。",
        "",
        "## 与 EHT 文献的对照",
        "",
        "| 量 | 本项目 (a=0.7, i=60°, 48×48 float64) | EHT M87* (2019) | EHT Sgr A* (2022) |",
        "|---|---|---|---|",
    ]
    md_lines.append(
        f"| Ring diameter (M units → μas via M=6.5e9 M_sun, D=16.8 Mpc) | "
        f"{ref_metrics['ring_diameter_M']:.2f} M | 42 ± 3 μas | 48.7 ± 7 μas |"
    )
    md_lines.append(
        f"| South / North asymmetry | {ref_metrics['asymmetry']['south_over_north']:.2f} | ≳ 10 | 强但变化快 |"
    )
    md_lines.append(
        f"| Photon-ring peak radius | {ref_metrics['photon_ring_peak_radius_M']:.2f} M | shadow radius ≈ 5√3 ≈ 8.66 M (a=0) | 类似 |"
    )
    md_lines += [
        "",
        "**说明**：",
        "- 本项目的 ring diameter 是几何半径中位数 × 2（intensity ≥ 0.5·max 的像素），",
        "  单位是 gravitational radius `M`。换算到 μas：M87* 的 1 M ≈ 3.83 μas",
        "  （M=6.5e9 M_sun, D=16.8 Mpc）。",
        "- 本项目使用 thin-disk + power-law emission，不带磁场动力学，所以",
        "  asymmetry 完全由 Doppler beaming + 视差几何贡献。EHT 的 ≳ 10:1 ratio",
        "  来自 GRMHD 亚相对论流场 + 完整 GRRT。",
        "- Photon-ring peak 在 thin disk 模型下不严格存在；用径向亮度峰值",
        "  作 proxy（Bardeen 1973 photon-sphere ≈ 5.2 M for a=0.7）。",
        "",
        "## 24 配置扫描表（来自 luminance proxy）",
        "",
        "| spin | inclination (deg) | D_ring [M] | south/north | peak_r [M] |",
        "|---|---|---|---|---|",
    ]
    for r in sweep_metrics:
        m = r["metrics"]
        md_lines.append(
            f"| {r['spin']:+.3f} | {r['inclination_deg']:.0f} | "
            f"{m['ring_diameter_M']:.2f} | "
            f"{m['asymmetry']['south_over_north']:.2f} | "
            f"{m['photon_ring_peak_radius_M']:.2f} |"
        )
    md_lines.append("")
    md_lines.append("生成：`PYTHONPATH=. .venv/Scripts/python.exe tools/eht_metrics_demo.py`")
    md = "\n".join(md_lines)
    Path("results/eht_metrics_report.md").write_text(md, encoding="utf-8")
    print("Saved results/eht_metrics_report.md")


if __name__ == "__main__":
    main()
