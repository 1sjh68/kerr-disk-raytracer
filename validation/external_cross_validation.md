# External Demo and Cross-Validation Status

更新时间：2026-05-21

## geokerr

- status: `success`
- source artifact: `research\repos\geokerr\geokerr_code\abgrid.out`
- parsed_geodesics: `400`
- line_count: `801`
- spin: `0.998`
- alpha_range: `-3.7` to `7.699999999999999`
- beta_range: `-5.7` to `5.699999999999999`
- plot: `research\reproduction\geokerr_abgrid_points.png`
- cross-validation: `validation/geokerr_cross_validation.json`
- state agreement: **87%**（400 条光线；配置 r_obs 与本地默认不同）
- strict status agreement: **91.25%**（a=0.7, i=60°, disk 一致率 98.27%；见 `validation/geokerr_strict_comparison.json`）

## Odyssey

### Docker 构建

- status: `build_attempt_failed`
- build_log: `research\repos\Odyssey\build_attempt.log`
- cuda_header_missing: `True`

```text
./src/main.cpp:26:10: fatal error: cuda.h: No such file or directory
   26 | #include <cuda.h>
      |          ^~~~~~~~
compilation terminated.
make: *** [Makefile:14: cpp] Error 1
```

### WSL2 + CUDA Toolkit

- status: `success`
- output: `research\repos\Odyssey\Output_task2.txt`
- output_lines: `16384`
- note: WSL2 Ubuntu, nvcc 12.0; 128×128 thermal syn image

## Interpretation

- geokerr 提供成功的外部 Kerr 光线参考，已用于轨迹级 sanity check。
- Odyssey 在 WSL2 + CUDA Toolkit 下可完整构建；Docker 一键复现仍需 CUDA 开发镜像。
- 严格状态判定已完成（91.25% overall / 98.27% disk）；相同观察者/相机约定下的坐标级轨迹对齐仍是后续工作。
- 本地 float64 geodesic 与 CPU 参考为 99.96% 状态匹配，disk-hit 计数完全一致。

## 相关脚本

```bash
python scripts/summarize_external_demos.py
python scripts/parse_geokerr.py
```
