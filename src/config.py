from __future__ import annotations

from pathlib import Path
from typing import Any

MAX_CONFIG_BYTES = 64 * 1024
MAX_RENDER_RESOLUTION = 4096
MAX_GEODESIC_RESOLUTION = 2048
MAX_INTEGRATION_STEPS = 100_000
MAX_CUDA_BLOCK_THREADS = 1024


DEFAULT_CONFIG: dict[str, Any] = {
    "black_hole": {"spin": 0.7, "mass_unit": 1.0},
    "camera": {
        "r_obs": 60.0,
        "inclination_deg": 60.0,
        "fov_m": 24.0,
    },
    "disk": {
        "inner_radius": "isco",
        "outer_radius": 28.0,
        "emissivity_index": 3.0,
        "model": "power_law",
    },
    "integration": {
        "step_size": 0.35,
        "max_steps": 700,
        "horizon_epsilon": 0.05,
        "escape_radius": 90.0,
    },
    "render": {
        "resolution": 128,
        "geodesic_resolution": 48,
        "gamma": 2.2,
        "bloom": True,
        "tone_map": "reinhard",
    },
}


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return {}
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    try:
        if any(ch in raw for ch in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _finite_float(
    value: Any,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite")
    if min_value is not None and result < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and result > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return result


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _int_range(value: Any, name: str, *, min_value: int, max_value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value and str(result) != str(value).strip():
        raise ValueError(f"{name} must be an integer")
    if result < min_value or result > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}")
    return result


def validate_resolution(
    value: Any,
    name: str = "resolution",
    *,
    max_value: int = MAX_RENDER_RESOLUTION,
) -> int:
    return _int_range(value, name, min_value=1, max_value=max_value)


def validate_cuda_block(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError("render.cuda_block must be formatted as WIDTHxHEIGHT")
        width, height = (_int_range(part, "render.cuda_block", min_value=1, max_value=MAX_CUDA_BLOCK_THREADS) for part in parts)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        width = _int_range(value[0], "render.cuda_block[0]", min_value=1, max_value=MAX_CUDA_BLOCK_THREADS)
        height = _int_range(value[1], "render.cuda_block[1]", min_value=1, max_value=MAX_CUDA_BLOCK_THREADS)
    else:
        raise ValueError("render.cuda_block must be a two-item list or WIDTHxHEIGHT string")
    if width * height > MAX_CUDA_BLOCK_THREADS:
        raise ValueError(f"render.cuda_block must not exceed {MAX_CUDA_BLOCK_THREADS} threads")
    return width, height


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {section: values.copy() if isinstance(values, dict) else values for section, values in config.items()}

    black_hole = cfg.setdefault("black_hole", {})
    camera = cfg.setdefault("camera", {})
    disk = cfg.setdefault("disk", {})
    integration = cfg.setdefault("integration", {})
    render = cfg.setdefault("render", {})
    if not all(isinstance(section, dict) for section in [black_hole, camera, disk, integration, render]):
        raise ValueError("config sections must be mappings")

    black_hole["spin"] = _finite_float(black_hole.get("spin", 0.7), "black_hole.spin", min_value=-0.999999, max_value=0.999999)
    black_hole["mass_unit"] = _finite_float(black_hole.get("mass_unit", 1.0), "black_hole.mass_unit", min_value=1.0e-12, max_value=1.0e12)

    camera["r_obs"] = _finite_float(camera.get("r_obs", 60.0), "camera.r_obs", min_value=2.0 + 1.0e-6, max_value=1.0e9)
    camera["inclination_deg"] = _finite_float(camera.get("inclination_deg", 60.0), "camera.inclination_deg", min_value=0.0, max_value=180.0)
    camera["fov_m"] = _finite_float(camera.get("fov_m", 24.0), "camera.fov_m", min_value=1.0e-9, max_value=1.0e9)

    inner_radius = disk.get("inner_radius", "isco")
    if isinstance(inner_radius, str):
        if inner_radius.lower() != "isco":
            try:
                inner_radius = _finite_float(inner_radius, "disk.inner_radius", min_value=1.0e-9, max_value=1.0e9)
            except ValueError as exc:
                raise ValueError("disk.inner_radius must be 'isco' or a positive finite number") from exc
        else:
            inner_radius = "isco"
    else:
        inner_radius = _finite_float(inner_radius, "disk.inner_radius", min_value=1.0e-9, max_value=1.0e9)
    disk["inner_radius"] = inner_radius
    disk["outer_radius"] = _finite_float(disk.get("outer_radius", 28.0), "disk.outer_radius", min_value=1.0e-9, max_value=1.0e9)
    if isinstance(inner_radius, float) and disk["outer_radius"] <= inner_radius:
        raise ValueError("disk.outer_radius must be greater than disk.inner_radius")
    disk["emissivity_index"] = _finite_float(disk.get("emissivity_index", 3.0), "disk.emissivity_index", min_value=-20.0, max_value=20.0)
    model = str(disk.get("model", "power_law"))
    if model not in {"power_law", "novikov_thorne"}:
        raise ValueError("disk.model must be 'power_law' or 'novikov_thorne'")
    disk["model"] = model

    integration["step_size"] = _finite_float(integration.get("step_size", 0.35), "integration.step_size", min_value=1.0e-9, max_value=1.0e6)
    integration["max_steps"] = _int_range(integration.get("max_steps", 700), "integration.max_steps", min_value=1, max_value=MAX_INTEGRATION_STEPS)
    integration["horizon_epsilon"] = _finite_float(integration.get("horizon_epsilon", 0.05), "integration.horizon_epsilon", min_value=0.0, max_value=1.0e6)
    integration["escape_radius"] = _finite_float(integration.get("escape_radius", 90.0), "integration.escape_radius", min_value=1.0e-9, max_value=1.0e9)

    render["resolution"] = validate_resolution(render.get("resolution", 128), "render.resolution")
    if "geodesic_resolution" in render:
        render["geodesic_resolution"] = validate_resolution(
            render["geodesic_resolution"],
            "render.geodesic_resolution",
            max_value=MAX_GEODESIC_RESOLUTION,
        )
    render["gamma"] = _finite_float(render.get("gamma", 2.2), "render.gamma", min_value=0.01, max_value=10.0)
    render["bloom"] = _bool_value(render.get("bloom", True), "render.bloom")
    tone_map = str(render.get("tone_map", "reinhard"))
    if tone_map != "reinhard":
        raise ValueError("render.tone_map must be 'reinhard'")
    render["tone_map"] = tone_map
    if "cuda_block" in render:
        render["cuda_block"] = list(validate_cuda_block(render["cuda_block"]))

    return cfg


def _merge_config(base: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    merged = {section: values.copy() if isinstance(values, dict) else values for section, values in base.items()}
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a small YAML subset without requiring PyYAML.

    The project config intentionally uses only two-level ``key: value`` mappings.
    If PyYAML is installed, it is used; otherwise this tiny parser is enough for
    the checked-in default file.
    """

    path = Path(path)
    config = {section: values.copy() for section, values in DEFAULT_CONFIG.items()}
    if not path.exists():
        return validate_config(config)
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError(f"config file is too large: {path}")

    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore

    if yaml is not None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"invalid YAML config: {path}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("config root must be a mapping")
        return validate_config(_merge_config(config, data))

    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            current = stripped[:-1]
            config.setdefault(current, {})
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            if current is None:
                config[key.strip()] = _parse_scalar(value)
            else:
                section = config.setdefault(current, {})
                if isinstance(section, dict):
                    section[key.strip()] = _parse_scalar(value)
    return validate_config(config)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out
