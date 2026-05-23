"""Compose spin/inclination sweep frames into GIF animations.

Uses already-rendered tiles in figures/sweep/ (24 frames from
parameter_sweep.py): 6 spins x 4 inclinations.

Outputs:
  figures/spin_sweep_animation.gif       (i=60° fixed, spin sweep)
  figures/inclination_sweep_animation.gif (a=0.7 fixed, inclination sweep)
  figures/full_grid_animation.gif        (24-frame raster, all combos)

This is a presentation aid, not a physically time-dependent animation
(spacetime is static; only camera/disk parameters vary).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


SPINS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.998]
INCLINATIONS = [10, 30, 60, 80]


def _frame(spin: float, inc: int) -> Path:
    return Path(f"figures/sweep/sweep_spin{spin}_inc{inc}.png")


def make_gif(frames: list[Path], out: Path, duration_ms: int = 500, label_fn=None) -> None:
    images = []
    for f in frames:
        if not f.exists():
            print(f"  ! missing {f}, skipping")
            continue
        im = Image.open(f).convert("RGB")
        images.append(im)
    if not images:
        print(f"  no frames for {out}")
        return
    images[0].save(
        out, format="GIF", save_all=True, append_images=images[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    print(f"  Saved {out} ({len(images)} frames, {out.stat().st_size // 1024} KB)")


def main() -> None:
    Path("figures").mkdir(exist_ok=True)

    print("=== spin sweep (i=60deg) ===")
    spin_frames = [_frame(s, 60) for s in SPINS]
    make_gif(spin_frames, Path("figures/spin_sweep_animation.gif"), duration_ms=600)

    print("\n=== inclination sweep (a=0.7) ===")
    inc_frames = [_frame(0.7, i) for i in INCLINATIONS]
    make_gif(inc_frames, Path("figures/inclination_sweep_animation.gif"), duration_ms=600)

    print("\n=== full grid (24 frames, spin then inclination) ===")
    full = []
    for s in SPINS:
        for i in INCLINATIONS:
            full.append(_frame(s, i))
    make_gif(full, Path("figures/full_sweep_animation.gif"), duration_ms=350)


if __name__ == "__main__":
    main()
