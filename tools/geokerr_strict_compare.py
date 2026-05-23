"""Strict trajectory comparison between geokerr and our CPU tracer.

Determines geokerr ray status empirically from trajectory data:
- captured: ray reaches r < r_horizon + 0.1
- disk: ray crosses equator (muf=0) at r between ISCO and outer_radius
- escaped: ray returns to large r without hitting disk/horizon
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.geodesic import trace_single_ray
from src.metric import horizon_radius, isco_radius
from src.safe_io import read_limited_text


def parse_geokerr_out(path: str | Path) -> tuple[list[dict], dict]:
    lines = read_limited_text(path).strip().splitlines()
    if not lines:
        return [], {}
    header = lines[0].split()
    ngeo = int(header[0])
    mu0 = float(header[1])
    a = float(header[2])
    u0 = float(header[3])
    r_obs_geokerr = 1.0 / u0
    inclination_deg = math.degrees(math.acos(mu0))
    r_h = horizon_radius(a)
    r_in = isco_radius(a)
    r_out = 28.0

    records = []
    idx = 1
    for _ in range(ngeo):
        meta = lines[idx].split()
        idx += 1
        alpha = float(meta[0])
        beta = float(meta[1])
        nup = int(meta[2])
        ncase = int(meta[-1])

        points = []
        for _ in range(nup):
            vals = lines[idx].split()
            idx += 1
            uf = float(vals[0])
            muf = float(vals[1])
            r = 1.0 / uf if uf > 0 else 1e10
            points.append({
                "r": r,
                "muf": muf,
                "theta": math.acos(max(-1.0, min(1.0, muf))),
            })

        # Empirical status determination
        min_r = min(p["r"] for p in points)
        max_r = max(p["r"] for p in points)

        # Check equator crossings
        equator_crossings = []
        for i in range(1, len(points)):
            p1, p2 = points[i-1], points[i]
            if p1["muf"] * p2["muf"] <= 0:  # Crossed equator
                # Interpolate r at crossing
                frac = abs(p1["muf"]) / (abs(p1["muf"]) + abs(p2["muf"]))
                r_cross = p1["r"] + frac * (p2["r"] - p1["r"])
                equator_crossings.append(r_cross)

        # Determine status
        if min_r <= r_h + 0.1:
            gstatus = "captured"
        elif any(r_in <= rc <= r_out for rc in equator_crossings):
            gstatus = "disk"
        elif max_r > 1000:  # Returns to large r
            gstatus = "escaped"
        else:
            gstatus = "max_steps"

        records.append({
            "alpha": alpha,
            "beta": beta,
            "ncase": ncase,
            "gstatus": gstatus,
            "min_r": min_r,
            "max_r": max_r,
            "equator_crossings": equator_crossings,
        })

    meta = {
        "a": a,
        "r_obs_geokerr": r_obs_geokerr,
        "inclination_deg": inclination_deg,
    }
    return records, meta


def compare_with_cpu(records: list[dict], meta: dict) -> dict:
    a = meta["a"]
    inclination_deg = meta["inclination_deg"]
    r_obs = 1000.0
    max_steps = 10000
    step_size = 0.35
    escape_radius = 90.0

    results = []
    agreements = 0
    disk_agree = 0
    captured_agree = 0
    escaped_agree = 0
    total_disk = 0
    total_captured = 0
    total_escaped = 0

    t0 = time.perf_counter()
    for i, rec in enumerate(records):
        hit = trace_single_ray(
            rec["alpha"],
            rec["beta"],
            a=a,
            inclination_deg=inclination_deg,
            r_obs=r_obs,
            step_size=step_size,
            max_steps=max_steps,
            escape_radius=escape_radius,
        )
        cstatus = hit.status
        gstatus = rec["gstatus"]

        match = cstatus == gstatus
        if match:
            agreements += 1

        if gstatus == "disk":
            total_disk += 1
            if match:
                disk_agree += 1
        elif gstatus == "captured":
            total_captured += 1
            if match:
                captured_agree += 1
        elif gstatus == "escaped":
            total_escaped += 1
            if match:
                escaped_agree += 1

        results.append({
            "alpha": rec["alpha"],
            "beta": rec["beta"],
            "ncase": rec["ncase"],
            "geokerr_status": gstatus,
            "cpu_status": cstatus,
            "match": match,
            "geokerr_min_r": rec["min_r"],
            "cpu_final_r": float(hit.state[1]),
        })

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(records)}] {elapsed:.1f}s elapsed")

    n = len(records)
    summary = {
        "a": a,
        "r_obs_geokerr": meta["r_obs_geokerr"],
        "r_obs_cpu": r_obs,
        "inclination_deg": inclination_deg,
        "total_rays": n,
        "overall_agreement": agreements / n if n > 0 else 0,
        "disk_agreement": disk_agree / total_disk if total_disk > 0 else 0,
        "captured_agreement": captured_agree / total_captured if total_captured > 0 else 0,
        "escaped_agreement": escaped_agree / total_escaped if total_escaped > 0 else 0,
        "total_disk": total_disk,
        "total_captured": total_captured,
        "total_escaped": total_escaped,
        "per_ray": results,
    }
    return summary


def main() -> None:
    geokerr_path = "research/repos/geokerr/geokerr_code/abgrid_r60.out"
    print(f"Parsing {geokerr_path} ...")
    records, meta = parse_geokerr_out(geokerr_path)
    print(f"  Found {len(records)} rays")
    print(f"  a={meta['a']}, r_obs_geokerr={meta['r_obs_geokerr']:.1f}, i={meta['inclination_deg']}deg")

    # Show geokerr status distribution
    from collections import Counter
    gcounts = Counter(r["gstatus"] for r in records)
    print(f"  Geokerr status distribution: {dict(gcounts)}")

    print("Running CPU comparison at r_obs=1000 (max_steps=10000) ...")
    summary = compare_with_cpu(records, meta)

    print(f"\n=== Geokerr Strict Comparison ===")
    print(f"Overall agreement: {summary['overall_agreement']*100:.2f}%")
    print(f"Disk agreement:    {summary['disk_agreement']*100:.2f}% ({summary['total_disk']} rays)")
    print(f"Captured agreement: {summary['captured_agreement']*100:.2f}% ({summary['total_captured']} rays)")
    print(f"Escaped agreement:  {summary['escaped_agreement']*100:.2f}% ({summary['total_escaped']} rays)")

    with open("validation/geokerr_strict_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved validation/geokerr_strict_comparison.json")


if __name__ == "__main__":
    main()
