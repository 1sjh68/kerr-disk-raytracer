"""Render approx vs CIE 1931 colour comparison.

Outputs:
- figures/cie_vs_approx_comparison.png   side-by-side fast renders
- figures/blackbody_locus.png             chromaticity sweep over T
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CONFIG
from src.render import render_thin_disk_fast
from src.disk_color import temperature_to_srgb


def render_pair(resolution: int = 192) -> tuple[np.ndarray, np.ndarray]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["render"]["resolution"] = resolution
    cfg["render"]["bloom"] = True

    cfg_a = copy.deepcopy(cfg)
    cfg_a["render"]["color_mode"] = "approx"
    out_a = render_thin_disk_fast(cfg_a)

    cfg_b = copy.deepcopy(cfg)
    cfg_b["render"]["color_mode"] = "cie1931"
    cfg_b["render"]["cie_t_min"] = 3500.0
    cfg_b["render"]["cie_t_max"] = 25000.0
    out_b = render_thin_disk_fast(cfg_b)

    return out_a["rgb"], out_b["rgb"]


def main() -> None:
    Path("figures").mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping figures")
        return

    rgb_a, rgb_b = render_pair()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(rgb_a)
    axes[0].set_title("approx (legacy)", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(rgb_b)
    axes[1].set_title("CIE 1931 + Planck + sRGB", fontsize=12)
    axes[1].axis("off")
    fig.suptitle("Disk colour pipeline comparison\n(fast renderer, 192x192, a=0.7, i=60°)", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/cie_vs_approx_comparison.png", dpi=150)
    plt.close()
    print("Saved figures/cie_vs_approx_comparison.png")

    # Black-body locus: T sweep 1500..30000 K
    T = np.linspace(1500.0, 30000.0, 256)
    rgb = temperature_to_srgb(T)
    bar = np.tile(rgb[None, :, :], (50, 1, 1))

    fig2, ax2 = plt.subplots(figsize=(9, 2.5))
    ax2.imshow(bar, aspect="auto", extent=[T[0], T[-1], 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel("Black-body temperature (K)")
    ax2.set_title("CIE 1931 + Planck + sRGB black-body locus")
    plt.tight_layout()
    plt.savefig("figures/blackbody_locus.png", dpi=150)
    plt.close()
    print("Saved figures/blackbody_locus.png")


if __name__ == "__main__":
    main()
