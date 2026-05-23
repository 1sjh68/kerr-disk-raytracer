# Validation Error Summary

## Status

- pytest_returncode: `0`
- backend: `cuda`
- backend_reason: `cupy_cuda_devices=1; device0=NVIDIA GeForce RTX 4060 Laptop GPU`

## CPU/GPU Map Metrics

- intensity_mse: `5.974770e-19`
- intensity_mae: `2.296257e-10`
- intensity_max_error: `1.490116e-08`
- intensity_relative_mae: `3.441364e-07`
- hit_mask_mismatch_fraction: `0.000000e+00`
- rgb_mse: `6.995111e-15`
- rgb_mae: `5.790738e-08`
- rgb_max_error: `6.854534e-07`

## Hamiltonian Geodesic CPU Reference

- method: `hamiltonian_geodesic_cpu`
- resolution: `48x48`
- elapsed_s: `34.925`
- hit_fraction: `0.917535`
- status_counts: `{'disk': 2114, 'captured': 182, 'escaped': 8, 'max_steps': 0, 'invalid': 0}`
- disk_null_error_mean: `1.596324e-03`
- disk_null_error_max: `2.416238e-01`

## External Demo Status

- geokerr_available: `True`
- geokerr_parsed_geodesics: `400`
- odyssey_docker_build_succeeded: `False`
- odyssey_wsl2_build_succeeded: `True`
- odyssey_cuda_header_missing: `True`

## Geodesic CPU/GPU (see validate_geodesic.py)

- float64_status_match_fraction: `0.999566`
- float64_intensity_mae: `1.070760e-10`
- float64_redshift_mae: `9.713245e-09`

## Interpretation

- This script validates the **fast thin-disk** CPU/GPU output contract only.
- Full per-pixel Hamiltonian geodesic validation is in `validate_geodesic.py` → `validation/geodesic_cpu_gpu_comparison.json`.
- float64 geodesic kernel @ 48×48 achieves **99.96% status match** with CPU when geodesic comparison JSON is present.
- Use `run_geodesic_gpu.py --precision float64` for scientific reference; float32 is for fast preview.
