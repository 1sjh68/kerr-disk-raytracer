from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "reason": "not_found"}
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "available": proc.returncode == 0,
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _cupy_probe() -> dict[str, Any]:
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:
        return {"available": False, "reason": f"{exc.__class__.__name__}: {exc}"}

    out: dict[str, Any] = {"available": True, "version": getattr(cp, "__version__", "unknown")}
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        out["device_count"] = count
        out["devices"] = []
        for idx in range(count):
            props = cp.cuda.runtime.getDeviceProperties(idx)
            name = props.get("name", b"unknown")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            out["devices"].append(
                {
                    "index": idx,
                    "name": name,
                    "total_global_mem_gb": round(float(props.get("totalGlobalMem", 0)) / 1024**3, 3),
                    "major": int(props.get("major", 0)),
                    "minor": int(props.get("minor", 0)),
                }
            )
    except Exception as exc:
        out["runtime_available"] = False
        out["runtime_error"] = f"{exc.__class__.__name__}: {exc}"
    else:
        out["runtime_available"] = out.get("device_count", 0) > 0
    return out


def main() -> None:
    from src.gpu_trace import configure_cuda_wheel_paths, cuda_available

    wheel_paths = configure_cuda_wheel_paths()
    available, reason = cuda_available()
    status = {
        "python": sys.version,
        "cuda_wheel_paths": wheel_paths,
        "cuda_available": available,
        "cuda_reason": reason,
        "cupy": _cupy_probe(),
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvcc": _run(["nvcc", "--version"]),
        "ptxas": _run(["ptxas", "--version"]),
    }
    out_path = ROOT / "logs" / "cuda_status.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
