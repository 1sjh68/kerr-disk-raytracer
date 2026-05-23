"""Parse geokerr abgrid.out and run cross-validation with our CPU geodesic tracer."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Allow running from project root without PYTHONPATH
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.geodesic import trace_single_ray
from src.metric import isco_radius
from src.render import write_json
from src.safe_io import read_limited_text


def parse_abgrid_out(path: str | Path, *, max_bytes: int = 32 * 1024 * 1024) -> list[dict]:
    """Parse geokerr abgrid.out standard output."""
    lines = read_limited_text(path, max_bytes=max_bytes).strip().splitlines()
    if not lines:
        return []
    header = lines[0].split()
    ngeo = int(header[0])
    mu0 = float(header[1])
    a = float(header[2])
    u0 = float(header[3])
    r_obs = 1.0 / u0
    inclination_deg = math.degrees(math.acos(mu0))

    records = []
    idx = 1
    for _ in range(ngeo):
        # ALPHA BETA NUP NCASE  (geokerr readme says 4 fields, but output has 5)
        # Observed format: alpha beta nup u0 ncase  (u0 repeated?)
        meta = lines[idx].split()
        idx += 1
        alpha = float(meta[0])
        beta = float(meta[1])
        nup = int(meta[2])
        # skip possible extra field, take last as ncase
        ncase = int(meta[-1])
        points = []
        for _ in range(nup):
            vals = lines[idx].split()
            idx += 1
            uf = float(vals[0])
            muf = float(vals[1])
            dt = float(vals[2])
            dphi = float(vals[3])
            lam = float(vals[4])
            tpm = int(vals[5])
            tpr = int(vals[6])
            points.append({
                "uf": uf, "muf": muf, "dt": dt, "dphi": dphi,
                "lambda": lam, "tpm": tpm, "tpr": tpr,
                "r": 1.0 / uf if uf != 0.0 else float("inf"),
                "theta": math.acos(muf) if abs(muf) <= 1.0 else (0.0 if muf > 0 else math.pi),
            })
        records.append({
            "alpha": alpha, "beta": beta, "nup": nup, "ncase": ncase,
            "points": points,
        })

    return {
        "ngeo": ngeo,
        "mu0": mu0,
        "a": a,
        "u0": u0,
        "r_obs": r_obs,
        "inclination_deg": inclination_deg,
        "rays": records,
    }


def cross_validate(geokerr_path: str, output_dir: str = "validation") -> dict:
    data = parse_abgrid_out(geokerr_path)
    a = data["a"]
    inclination_deg = data["inclination_deg"]
    r_obs = data["r_obs"]
    r_inner = isco_radius(a)
    r_outer = 28.0  # default from our config
    q = 3.0
    model = "power_law"
    step_size = 0.35
    max_steps = 700
    horizon_epsilon = 0.05
    escape_radius = 90.0

    comparisons = []
    status_match = 0
    for ray in data["rays"]:
        alpha = ray["alpha"]
        beta = ray["beta"]
        hit = trace_single_ray(
            alpha, beta, a=a, inclination_deg=inclination_deg,
            r_obs=r_obs, disk_inner=r_inner, disk_outer=r_outer,
            emissivity_index=q, emission_model=model,
            step_size=step_size, max_steps=max_steps,
            horizon_epsilon=horizon_epsilon, escape_radius=escape_radius,
        )
        gk = ray["points"][0] if ray["points"] else None
        comp = {
            "alpha": alpha,
            "beta": beta,
            "ncase": ray["ncase"],
            "our_status": hit.status,
            "our_radius": hit.radius,
            "our_redshift": hit.redshift,
            "our_steps": hit.steps,
            "our_null_error": hit.null_error,
        }
        if gk:
            comp["gk_r"] = gk["r"]
            comp["gk_theta_deg"] = math.degrees(gk["theta"])
            comp["gk_lambda"] = gk["lambda"]
            comp["gk_dt"] = gk["dt"]
            comp["gk_dphi"] = gk["dphi"]
            comp["gk_tpm"] = gk["tpm"]
            comp["gk_tpr"] = gk["tpr"]
        comparisons.append(comp)
        if hit.status == "disk" and gk and abs(gk["muf"]) < 0.1:
            status_match += 1
        elif hit.status != "disk" and (gk is None or abs(gk["muf"]) >= 0.1):
            status_match += 1

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_json(str(out_path / "geokerr_cross_validation.json"), {
        "geokerr_params": {
            "ngeo": data["ngeo"],
            "a": data["a"],
            "r_obs": data["r_obs"],
            "inclination_deg": data["inclination_deg"],
        },
        "total_rays": len(comparisons),
        "status_agreement": status_match / len(comparisons) if comparisons else 0.0,
        "comparisons": comparisons[:50],  # truncate for readability
    })
    return {
        "total": len(comparisons),
        "status_agreement": status_match / len(comparisons) if comparisons else 0.0,
    }


if __name__ == "__main__":
    result = cross_validate("research/repos/geokerr/geokerr_code/abgrid.out")
    print(json.dumps(result, indent=2))
