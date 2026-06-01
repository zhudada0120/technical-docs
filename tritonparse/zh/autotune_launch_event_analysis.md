# Autotune Benchmark Launch 事件过多分析

## 问题现象

执行 `python tests/vector-add.py`，日志文件产生 **653 条事件**：

| 事件类型 | 数量 |
|---------|------|
| compilation | 8 |
| launch | **645** |

一个简单的 vector-add kernel 只应有 1 次 compilation + 1 次 launch，实际多出了 7 次 compilation 和 644 次 launch。

## 根因分析

### 原因一：Ascend AutoTilingTuner 自动生成配置

测试代码使用了 `@triton.autotune(configs=[], key=["n_elements"])`。

- **标准 Triton (GPU)**：`configs=[]` 生成 1 个默认配置 `{num_warps=4, num_stages=3}`，因为只有 1 个配置，**跳过 benchmark 直接执行**。
- **Ascend (NPU)**：`configs=[]` 触发 `AutoTilingTuner.auto_gen_config = True`，通过 `TileGenerator` 自动生成多个候选 tiling 配置，再扩展 warp/multibuffer 变体，最终产生 **8 个编译配置**。

8 个配置对应的 grid 分布：

| Grid | Launch 次数 |
|------|-----------|
| [8]  | 146 |
| [16] | 165 |
| [32] | 168 |
| [40] | 166 |

### 原因二：每个配置 benchmark 期间反复 launch

Autotune 对每个候选配置执行 benchmark 以选择最优配置：

- **GPU 标准 `do_bench`**：warmup 25ms + rep 100ms，对快 kernel 可能每个配置执行数百到数千次 launch
- **NPU `do_bench_npu`**：warmup=5 + active=30 = 35 次迭代/配置

实际数据：8 配置 × ~80 次 launch/配置 ≈ 645 次 launch。

### 原因三：structured_logging 未过滤 autotune benchmark launch（tritonparse 侧 bug）

这是核心问题。`LaunchHookImpl.__call__` 在 `structured_logging.py:1747` 对**每次 launch 都无条件记录**，不区分是否来自 autotune benchmark。

虽然 `_is_autotune_benchmark_launch()` 已存在于 `parse/sourcemap_utils.py:150`，但它**仅用于控制 tensor blob 保存**，没有用于过滤 launch 事件本身。

经验证：**645 次 launch 全部来自 autotune 调用栈**（stack 中包含 `autotuner` 路径）。

## GPU 环境验证要点

在 GPU 环境上复现时，需要确认以下内容：

### 1. 标准 autotune（configs=[]）

运行 `python tests/vector-add.py`，检查日志中 launch 事件数量。

**预期**：GPU 上 `configs=[]` 只生成 1 个默认配置且跳过 benchmark，应该只有 **1 个 compilation + 1 个 launch**。如果 GPU 上也出现大量 launch，说明 GPU 侧也有类似问题。

### 2. 显式多配置 autotune

修改测试代码，手动指定多个配置：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=["n_elements"]
)
```

检查日志中 launch 事件数量。预期每个配置被 benchmark 多次（`do_bench` 的 warmup + rep），这些 benchmark launch 也会被全部记录。

### 3. 检查 `_is_autotune_benchmark_launch` 在 GPU 上是否生效

`_is_autotune_benchmark_launch` 通过检查调用栈中是否有 `_bench` 函数（`triton/runtime/autotuner.py`）来判断。GPU 和 NPU 的 autotuner 代码路径不同：

- GPU：`triton/runtime/autotuner.py` 中的 `_bench` 方法
- NPU：`triton/backends/ascend/runtime/autotuner.py` 中的 `_batch_bench` / `_bench`

确认该函数在 GPU 路径下能否正确识别 autotune benchmark launch。

## 修复建议

在 `LaunchHookImpl.__call__`（`structured_logging.py`）中加入 autotune benchmark 过滤：

```python
def __call__(self, *args, **kwargs):
    # ... 现有代码 ...
    if _is_autotune_benchmark_launch():
        return  # 跳过 autotune benchmark 产生的 launch 事件
    # ... 继续记录 launch 事件 ...
```

这样只会保留 autotune 选出最优配置后的**最终 1 次 launch**。

## 文件位置参考

| 文件 | 位置 | 说明 |
|------|------|------|
| `tritonparse/structured_logging.py:1747` | `LaunchHookImpl.__call__` | launch 事件记录入口，无条件记录 |
| `tritonparse/structured_logging.py:1609` | `add_launch_metadata` | 已有 autotune 检测，但仅跳过 tensor blob |
| `tritonparse/parse/sourcemap_utils.py:150` | `_is_autotune_benchmark_launch` | autotune benchmark 检测函数 |
| `triton/runtime/autotuner.py:32-33` (GPU) | 标准 autotuner | `configs=[]` 生成 1 个默认配置，跳过 benchmark |
| `triton/backends/ascend/runtime/autotuner.py:108` (NPU) | `AutoTilingTuner` | `configs=[]` 触发自动配置生成 + benchmark |
