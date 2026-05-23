# GPU Memory Static Analysis (Phase 8 — closed-loop without ncu)

更新时间：2026-05-23

## 摘要

`DELIVERY_STATUS.md` Phase 8 的两个未完成项是 **constant memory 优化** 与 **global memory 访问分析**。完整动态指标需要 `ncu --set full`（已就绪一键脚本 `tools/run_ncu_pipeline.ps1`，等用户一次 admin UAC 即可），但**静态可分析的部分**已足以给出闭环结论：

1. **Constant memory 优化已被 CuPy 自动完成**——`ptxas -v` 报告 `cmem[0] = 456 bytes`，意味着 RawKernel 把所有标量参数（spin、inclination、fov、disk 几何、积分常数、emission 模型 ID 等）放进 SM 的 constant cache。手写 `__constant__` struct 不会再带来增益。

2. **Global memory 写入是单次每像素 6 个标量**，按 `idx = iy * width + ix` 顺序写入 6 个独立数组（intensity / redshift / temperature / hit_mask / status_code / null_error）。访问 stride = 1，warp 内 32 个线程的写入完美 coalesce，单次 transaction = 一个 cache line。**不存在 coalescing 病理**。

3. **没有显式 shared memory** 使用，因为 kernel 是 per-pixel 独立的，邻居像素不共享中间数据。

下面是详细推导。

## 1. Constant memory 现状

### ptxas -v 输出（来自 `cuda/profile_geodesic.cu` standalone 编译）

```
ptxas info    : 24 bytes gmem
ptxas info    : Compiling entry function '_Z20kerr_geodesic_kernel...'
ptxas info    : Function properties for _Z20kerr_geodesic_kernel...
    32 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 82 registers, 456 bytes cmem[0], 8 bytes cmem[2]
```

### 解读

| 指标 | 值 | 含义 |
|---|---|---|
| `cmem[0]` | 456 bytes | 用户 kernel 参数槽。CUDA 把每个 kernel 调用的所有参数（指针 + 标量）打包放到 SM constant cache 的 bank 0。读取延迟 ~5 cycles，broadcast 到 warp 所有 lane。**已是 constant memory 行为，无需手写 `__constant__`**。 |
| `cmem[2]` | 8 bytes | 编译器隐式常量（PTX intrinsic 等） |
| `gmem` | 24 bytes | 静态全局变量（这里几乎没有，预期之内） |

### 数 456 字节怎么来的

我们的 kernel 签名（`kerr_geodesic_kernel`）：

```
const int width, const int height,                                 // 2 × 4 = 8
const float spin, const float inclination_rad, const float fov_m,  // 3 × 4 = 12
const float r_inner, const float r_outer, const float emissivity_q,// 3 × 4 = 12
const int emission_model, const float r_obs,                       // 4 + 4 = 8
const float step_size, const int max_steps,                        // 4 + 4 = 8
const float horizon_epsilon, const float escape_radius,            // 2 × 4 = 8
float* intensity, float* redshift, float* temperature,             // 3 × 8 = 24
unsigned char* hit_mask, unsigned char* status_code,               // 2 × 8 = 16
float* null_error                                                  // 8
```

= 8+12+12+8+8+8+24+16+8 = **104 bytes**。剩下 ~352 字节是 padding + ABI metadata（grid/block id、ABI prologue 等），属于 nvcc 的 ABI 行为。

### 结论

**没有再加 `__constant__ struct` 的必要**。手写 `__constant__` 仅在两种情况有意义：

1. 同一组常数被**反复跨 kernel 共享**（例如度规系数表）。我们的 kernel 是单次启动单 ray，参数生命周期 = 一次启动，CuPy 的 cmem[0] 已最优。
2. 数据量超过 64 KB 单参数槽限制。我们 104 bytes 远低于此。

**等价证据**：`ncu --set full` 跑出来后看 `smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_const.sum` 这个计数器，它会显示 constant cache 访问频次。预期不同 kernel 配置之间的差异可忽略。

## 2. Global memory 静态分析

### Kernel 输出布局

每个线程在 kernel 末尾执行：

```cuda
const int idx = iy * width + ix;
intensity[idx]   = out_intensity;       // float, 4 bytes
redshift[idx]    = out_redshift;        // float, 4 bytes
temperature[idx] = out_temperature;     // float, 4 bytes
hit_mask[idx]    = (final_status == 1) ? 1 : 0;  // u8, 1 byte
status_code[idx] = (unsigned char)final_status;  // u8, 1 byte
null_error[idx]  = out_null_error;      // float, 4 bytes
```

每个像素 6 次 store，总计 **18 bytes per pixel**。256×256 image → 65,536 pixels × 18 = ~**1.18 MB** total store traffic。

### Coalescing 分析

CUDA warp = 32 threads，threadIdx.x 连续 → 同一 warp 内 32 个线程的 `ix` 连续，`iy` 相同（取决于 block layout，但 16×16 block 时一个 warp 横跨两行的概率 = 16/32 = 50%）。

- **典型 case（16×16 block，warp 沿 x 方向）**：32 个线程的 `idx = iy*W + ix` 在同一行内步长 1，**完美 coalesce**，单次 32-byte（uchar）或 128-byte（float）transaction。
- **跨行 case（warp 横跨 row boundary）**：16 个线程一行 + 16 个下一行，仍然每个 16-thread group 是连续的，coalesce 仍然好（NVIDIA L1 拆 transaction，不会浪费带宽）。

**结论**：不存在 stride > 1 / random / scatter pattern。global memory store 已是带宽最优。

### 进一步证据

- ptxas 报告 `0 bytes spill stores` → register 没溢出到 local（local memory 写也是 global memory pattern，会污染 traffic）。
- kernel 内只有结尾 6 次 store，没有 read-modify-write 循环。
- 输入 (kernel 参数) 全在 cmem，不走 global。

### 待 ncu 量化的指标

如果跑 `ncu --set full` 跑通后想验证，关注：
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` → 应该 < 5%（compute-bound）
- `l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second` → store traffic
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio` → coalescing 效率（理想 1.0）

预期结果：global memory 远不是 bottleneck，kernel 是 **compute-bound**（warp divergence + transcendentals 主导，参见 `results/fastmath_optimization.md`）。

## 3. Shared memory

未使用。kernel 是 per-pixel 独立计算，邻居像素之间没有数据共享需求。

如果将来要做 **adaptive ray clustering**（按命运预测 group rays 减少 warp divergence），shared memory 可作为 group-level 工作队列；但这是 T8 的扩展工作，不是 T5。

## 4. 闭环说明

| 子项 | 状态 | 证据 |
|---|---|---|
| Constant memory 优化 | **已由 CuPy 自动完成** | `cmem[0] = 456 bytes` |
| Global memory 访问分析 | **静态分析完成；动态量化等 ncu** | 输出 18 bytes/pixel × 65k pixels @ 256² = 1.18 MB total，stride=1 完美 coalesce |
| Shared memory | 不使用 | per-pixel 独立 kernel |
| 寄存器压力 | 82 regs/thread，0 spill | `ptxas -v` 直接报告 |

`DELIVERY_STATUS.md` Phase 8 这两项可以视为**闭环**——更细的动态量化（cache hit / DRAM throughput / occupancy） 在 ncu 数据落盘后补到 `results/nsight_compute_report.md`。

## 5. 复现

```powershell
# Static analysis: nvcc -ptxas-options=-v
wsl -d Ubuntu -- bash '/mnt/d/Desktop/black hole/tools/wsl_profile_pipeline.sh'

# Dynamic ncu (要一次 UAC):
powershell -File 'D:\Desktop\black hole\tools\run_ncu_pipeline.ps1'
```
