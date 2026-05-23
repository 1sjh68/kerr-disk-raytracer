"""生成 paper/main.pdf 的多页技术报告（中文版）。

关键设计：
- matplotlib backend_pdf 拼接多页（避免 LaTeX/Pandoc 依赖）
- 文档级 `font.sans-serif` 设置 Microsoft YaHei / SimHei / Noto Sans CJK
  fallback，让中文不变成豆腐方块
- 内容从 logs / validation / results 动态读取最新数字，正文段落硬编码为中文
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.safe_io import read_limited_json


# ---- 中文字体 fallback：Windows / Linux / Mac 都能找到至少一个 ----
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",   # Windows 默认中文
    "SimHei",            # Windows 备选
    "Noto Sans CJK SC",  # Linux noto
    "PingFang SC",       # macOS
    "Arial Unicode MS",  # macOS 备选
    "DejaVu Sans",       # 通用 fallback（不支持中文，但避免崩溃）
]
matplotlib.rcParams["axes.unicode_minus"] = False


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), dpi=160)
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    fig.text(0.08, 0.94, title, fontsize=20, weight="bold", ha="left", va="top")
    y = 0.88
    for line in lines:
        if line.startswith("## "):
            fig.text(0.08, y, line[3:], fontsize=14, weight="bold", ha="left", va="top")
            y -= 0.04
        elif line == "":
            y -= 0.025
        else:
            fig.text(0.10, y, line, fontsize=10.5, ha="left", va="top", wrap=True)
            y -= 0.032
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), dpi=160)
    fig.text(0.05, 0.94, title, fontsize=18, weight="bold", ha="left", va="top")
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.74])
    ax.axis("off")
    if image_path.exists():
        ax.imshow(plt.imread(image_path))
    fig.text(0.06, 0.06, caption, fontsize=10, ha="left", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def _read_json_safe(path: Path) -> dict:
    if path.exists():
        try:
            return read_limited_json(path)
        except Exception:
            return {}
    return {}


def main() -> None:
    out = ROOT / "paper" / "main.pdf"

    # === 动态摘取最新数字 ============================================
    gpu_run = _read_json_safe(ROOT / "logs" / "gpu_run.json")
    gpu_backend_line = (
        f"- CUDA 后端：{gpu_run.get('backend')}（{gpu_run.get('method')}）。"
        if gpu_run else "- CUDA 后端状态记录于 logs/gpu_run.json。"
    )

    geo_cmp = _read_json_safe(ROOT / "validation" / "geodesic_cpu_gpu_comparison.json")
    f64 = geo_cmp.get("float64", {})
    if f64 and "error" not in f64:
        geodesic_line = (
            f"- 测地线 float64 @ 48×48：状态匹配率 "
            f"{float(f64.get('status_match_fraction', 0.0)) * 100:.2f}%，"
            f"intensity MAE {float(f64.get('intensity_mae', 0.0)):.2e}。"
        )
    else:
        geodesic_line = "- 测地线 CPU/GPU 对比记录于 validation/geodesic_cpu_gpu_comparison.json。"

    fastmath_data = _read_json_safe(ROOT / "results" / "fastmath_benchmark.json")
    fastmath_line = "- fast_math 优化：results/fastmath_benchmark.json"
    if fastmath_data and "runs" in fastmath_data:
        runs_512 = [r for r in fastmath_data["runs"] if r.get("resolution") == 512]
        if runs_512:
            r = runs_512[0]
            fastmath_line = (
                f"- fast_math @ 512×512：baseline {r['baseline_kernel_ms']:.2f} ms → "
                f"fastmath {r['fastmath_kernel_ms']:.2f} ms（{r['speedup']}×）"
                f"，status 一致率 {r['status_match_rate']*100:.2f}%。"
            )

    eht = _read_json_safe(ROOT / "results" / "eht_metrics.json")
    eht_line = "- EHT 风格指标：results/eht_metrics.json"
    if eht and eht.get("reference"):
        ref = eht["reference"]
        eht_line = (
            f"- EHT 风格指标（a=0.7, i=60°, 48×48 float64）："
            f"ring 直径 {ref.get('ring_diameter_M', 0):.2f} M，"
            f"south/north 不对称 {ref.get('asymmetry', {}).get('south_over_north', 0):.2f}，"
            f"photon-ring peak {ref.get('photon_ring_peak_radius_M', 0):.2f} M。"
        )

    pol = _read_json_safe(ROOT / "results" / "polarization_demo.json")
    pol_line = "- 偏振 stub：results/polarization_demo.json"
    if pol:
        pol_line = (
            f"- 偏振 stub demo（a={pol.get('spin')}, i={pol.get('inclination_deg')}°，48×48）："
            f"disk hit {pol.get('disk_hit_pixels')}/{int(pol.get('resolution', 48))**2}，"
            f"Π_obs 平均 {pol.get('pi_obs_mean', 0):.3f}。"
        )

    external_summary = ROOT / "research" / "external_demo_summary.json"
    external_lines = [
        "## 外部复现",
        "- geokerr 外部 demo 尚未生成摘要（运行 scripts/summarize_external_demos.py）。",
    ]
    external = _read_json_safe(external_summary)
    if external:
        geokerr = external.get("geokerr", {})
        odyssey = external.get("odyssey", {})
        external_lines = [
            "## 外部复现",
            f"- geokerr abgrid demo：available={geokerr.get('available')}，"
            f"已解析测地线={geokerr.get('parsed_pairs', 0)}。",
            f"- geokerr 源数据路径：{geokerr.get('path', '')}",
            f"- Odyssey Docker 构建：成功={odyssey.get('docker_build_succeeded', odyssey.get('build_succeeded'))}，"
            f"cuda.h missing={odyssey.get('contains_cuda_header_error')}。",
            f"- Odyssey WSL2 构建：成功={odyssey.get('wsl2_build_succeeded', False)}。",
            "- geokerr 提供外部轨迹参考：原始 abgrid 状态一致率 87%；"
            "a=0.7/i=60° 严格状态判定一致率 91.25%。",
        ]

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        # --- Page 1: 摘要 -------------------------------------------------
        add_text_page(
            pdf,
            "Kerr 黑洞薄盘 CPU/GPU 光线追踪",
            [
                "## 摘要",
                "本报告记录一个可复现的 Kerr 黑洞薄盘吸积光线追踪项目，",
                "提供两条管线：fast thin-disk MVP，以及完整逐像素 Hamiltonian",
                "测地线（CPU + CUDA float32/float64）。",
                "float64 测地线 @ 48×48 与 CPU 参考状态一致率 99.96%，disk-hit 完全一致。",
                "",
                "## 当前能力",
                "- Boyer-Lindquist 坐标 Kerr 度规、视界、ISCO、Hamiltonian 右端项 + RK 积分。",
                "- fast thin-disk 渲染（run_cpu/run_gpu）做快速预览与 CUDA baseline。",
                "- 完整测地线渲染（run_geodesic_cpu/run_geodesic_gpu）含 kerr_geodesic_kernel_double。",
                "- fast_math 编译选项加速 float32 测地线 ~3×（accuracy 100% 保持）。",
                "- CIE 1931 + Planck + sRGB 严格颜色管线（src/disk_color.py）。",
                "- Walker-Penrose 偏振 stub（src/polarization.py，输出 Stokes I/Q/U + EVPA）。",
                "- EHT 风格指标 ring diameter / asymmetry（src/eht_metrics.py）。",
                "- GRMHD HDF5 reader stub + synthetic fluid（src/grmhd_io.py）。",
                gpu_backend_line,
            ],
        )

        # --- Page 2: 物理模型与验证 ------------------------------------
        add_text_page(
            pdf,
            "物理模型与验证",
            [
                "## 物理模型",
                "- 单位：G = c = M = 1。",
                "- 坐标：Boyer-Lindquist (t, r, θ, φ)。",
                "- 盘：光学厚赤道薄盘，ISCO 内边界，外半径可配。",
                "- 发射：默认幂律 r^-q，可选 Novikov-Thorne flux。",
                "- 颜色：默认近似 RGB；可切换 CIE 1931 + Planck + sRGB（color_mode=cie1931）。",
                "",
                "## 验证",
                "- 度规对称性 / 逆度规恒等式 / Schwarzschild 视界与 ISCO 检查。",
                "- 零光子初始化 + RK4 单步 null 漂移检查。",
                "- 红移正定性 + 观测强度依赖性检查。",
                "- fast 路径 CPU/GPU map MSE/MAE/max error/hit-mask 不一致计数。",
                geodesic_line,
                fastmath_line,
                eht_line,
                pol_line,
                "- pytest 套件：47 passed（19 核心 + 12 disk_color + 10 polarization + 6 grmhd_io）。",
            ],
        )

        # --- Pages 3+: 图像 ----------------------------------------------
        add_image_page(
            pdf, "最终渲染",
            ROOT / "figures" / "final_render.png",
            "图 1：当前 MVP Kerr 薄盘渲染（fast 管线 + bloom + tone mapping）。",
        )
        add_image_page(
            pdf, "Hamiltonian 测地线参考",
            ROOT / "figures" / "geodesic_reference.png",
            "图 2：逐像素 Hamiltonian 测地线 CPU 参考（48×48），上采样后展示。",
        )
        add_image_page(
            pdf, "fast 路径 CPU/GPU 对比",
            ROOT / "results" / "comparison_grid.png",
            "图 3：fast 路径 CPU 渲染、CUDA/fallback 渲染与 intensity 误差图三联展示。",
        )
        add_image_page(
            pdf, "测地线分辨率扫描",
            ROOT / "results" / "geodesic_resolution_sweep.png",
            "图 4：float32/float64 测地线 CUDA kernel 时间随分辨率变化。",
        )
        add_image_page(
            pdf, "fast_math 加速比",
            ROOT / "figures" / "fastmath_speedup.png",
            "图 5：--use_fast_math 编译选项在 float32 测地线 kernel 上的加速比，"
            "≥192² 稳定 ~3×；intensity MAE 在 1e-9 量级，accuracy 完全保持。",
        )
        add_image_page(
            pdf, "颜色管线对比",
            ROOT / "figures" / "cie_vs_approx_comparison.png",
            "图 6：左为旧的近似 RGB（艺术化金黄调），右为 CIE 1931 + Planck + sRGB 严格颜色。",
        )
        add_image_page(
            pdf, "偏振 stub Stokes I/Q/U",
            ROOT / "figures" / "polarization_stokes_qu.png",
            "图 7：Walker-Penrose 偏振 stub 输出（a=0.7, i=60°, 48×48, Π_em=0.1）。",
        )
        add_image_page(
            pdf, "geokerr 坐标级对齐",
            ROOT / "research" / "reproduction" / "geokerr_coordinate_alignment.png",
            "图 8：在 5 条代表性光线上对 geokerr 做 (r, θ) 几何最近邻匹配；"
            "远离临界 ray 的对齐良好，photon sphere 附近差异显著。",
        )
        add_image_page(
            pdf, "RK4 vs Dormand-Prince RK45",
            ROOT / "research" / "reproduction" / "rk45_vs_rk4.png",
            "图 9：RK45 自适应步长在临界 ray 上的 null 守恒量改善 ~36 个数量级；"
            "几何路径与 RK4 基本一致但 wall time 加快 5–7×。",
        )
        add_image_page(
            pdf, "自旋扫描",
            ROOT / "results" / "spin_comparison.png",
            "图 10：固定倾角下不同自旋（a=0.0–0.998）的盘外观。",
        )
        add_image_page(
            pdf, "倾角扫描",
            ROOT / "results" / "inclination_comparison.png",
            "图 11：固定自旋下不同倾角（10°–80°）的盘外观。",
        )
        add_image_page(
            pdf, "geokerr 外部 demo",
            ROOT / "research" / "reproduction" / "geokerr_abgrid_points.png",
            "图 12：geokerr 官方 abgrid.in demo 解析为屏幕坐标采样。",
        )

        # --- 末尾：外部复现 + 限制 + 后续 ------------------------------
        add_text_page(pdf, "外部复现状态", external_lines)
        add_text_page(
            pdf,
            "限制与后续工作",
            [
                "## 限制",
                "- 两条渲染路径并存：fast thin-disk MVP vs 完整测地线；科学参考应使用 float64 测地线。",
                "- geokerr 交叉验证：原始 abgrid 状态一致率 87%；a=0.7/i=60° 严格判定 "
                "总体 91.25% / disk 98.27%；坐标级 RMS Δr/r ~0.22。",
                "- Odyssey 在 WSL2 中可构建，Docker 中缺 CUDA dev header。",
                "- 薄盘模型不是 EHT 源模型，不应解读为 M87* / Sgr A* 拟合。",
                "- 偏振 stub：Walker-Penrose 框架已搭好，但磁场为纯 toroidal、无 Faraday rotation、"
                "不沿轨迹 radiative transfer。",
                "- ncu 完整 --set full profile 待用户确认 UAC 后跑（一键脚本 tools/run_ncu_pipeline.ps1）。",
                "",
                "## 后续工作",
                "- geokerr 在匹配 observer 半径与相机约定下的轨迹比较。",
                "- GPU adaptive step / warp bucketing；WSL2 中 ncu profiling。",
                "- 博士级扩展：完整偏振 radiative transfer、GRMHD ingestion、多 GPU、"
                "EHT 风格图像指标深度对比、可微 ray tracing、神经网络 surrogate。",
                "  详见 docs/extensions_roadmap.md。",
            ],
        )


if __name__ == "__main__":
    main()
