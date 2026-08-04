# GPU Triton Autotune 缓存行为验证报告

> 配套文档：`gpu验证指南.md`（实验设计）｜ 验证脚本：`gpu_autotune_exp.py`
> 验证日期：2026-08-06

## 1. 背景与目的

issue #1345 描述了 **NPU（triton-ascend）** 上 autotune 的三类行为。本验证的目标是在 **GPU triton** 上复现同样三个实验，作为对照基线，确认：

- 哪些行为是 triton 通用机制（GPU/NPU 一致）；
- 哪些行为是 GPU 独有、NPU 缺失（即 issue 的核心问题）。

需要验证的三个现象：

| # | 验证项 | 期望结论 |
|---|--------|----------|
| 实验 1 | 清缓存后，同一 kernel 每次 autotune 的 benchmark 执行次数是否不同？ | **不同**（`do_bench` 动态算次数，非确定） |
| 实验 2 | 同进程内不清理缓存，第二次调用是否命中内存缓存？ | **命中**（`/self.cache/` 生效） |
| 实验 3 | 开启磁盘缓存后，跨进程是否能命中缓存？ | **命中**（GPU triton v3.4.0+ 支持） |

## 2. 验证环境

| 项目 | 版本 / 型号 |
|------|------------|
| GPU | NVIDIA A100-SXM4-80GB |
| 驱动 | 580.126.16 |
| torch | 2.12.0+cu130 |
| triton | **3.7.1**（site-packages 安装，≥ 3.4.0，具备磁盘缓存能力） |
| conda env | `torch-gpu` |

## 3. 验证方法

### 3.1 脚本设计

`gpu_autotune_exp.py` 提供三个模式：

- `single` —— 单次运行，供独立进程实验调用；
- `double` —— 同进程连续两次调用，验证内存缓存；
- `diskcache` —— 打印磁盘缓存验证步骤说明（实际跨进程验证通过 shell 两步完成）。

### 3.2 Kernel 与 configs

> ⚠️ **与 NPU 脚本的关键差异**：NPU 上 `@triton.autotune(configs=[], ...)` 的**空 configs 会触发其 auto-tiling 机制**；GPU 没有该机制，空 configs 会导致没有任何 config 可 benchmark（恒为 0）。因此本脚本给 kernel 显式配置了一组 configs。
>
> 三个实验验证的是 autotune **机制本身**的行为，与 kernel 选型无关，故沿用最简单的 vector-add。

采用 vector-add kernel，配置 **8 个 autotune configs**（不同 `BLOCK_SIZE` / `num_warps` 组合）：

```python
configs = [
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
]
```

输入规模 `SIZE = 2**22`（4M 元素），使 `estimate_ms` 落在 0.01~0.1ms，执行轮数随抖动明显波动。

### 3.3 观测手段：monkey-patch `do_bench`

`triton.testing.do_bench` 会先跑 5 次估算单次耗时 `estimate_ms`，再动态推导预热/计时轮数：

```python
estimate_ms = elapsed / 5
n_warmup  = max(1, int(warmup / estimate_ms))   # warmup 默认 25ms
n_repeat  = max(1, int(rep   / estimate_ms))     # rep 默认 100ms
```

脚本 patch 了 `do_bench`，在估算阶段记录每个 config 的 `estimate_ms / n_warmup / n_repeat / total`（`total = 5 + n_warmup + n_repeat`）。patch 的估算逻辑与 triton 3.7.x `testing.py:147-167` 逐行对齐，仅作观察；真正 benchmark 仍交给原始 `do_bench` 完成。

---

## 4. 实验 1：`do_bench` 执行次数非确定性

**步骤**：每次先 `rm -rf ~/.triton/cache`，再独立进程运行 `single`，重复 3 次。

**结果**：

| 运行 | configs | total_calls | 耗时 |
|------|---------|-------------|------|
| Run 1 | 8 | **4864** | 3.47s |
| Run 2 | 8 | **4894** | 2.32s |
| Run 3 | 8 | **4994** | 2.28s |

**三次 `total_calls` 各不相同。** 逐 config 看 `n_repeat` 也整体抖动：

| Config (BLOCK_SIZE/warps) | Run 1 | Run 2 | Run 3 |
|---------------------------|-------|-------|-------|
| C0 (256/4)   | 29  | 65  | 67  |
| C1 (512/4)   | 539 | 541 | 559 |
| C2 (1024/4)  | 546 | 551 | 558 |
| C3 (1024/8)  | 551 | 558 | 554 |
| C4 (2048/8)  | 555 | 560 | 558 |
| C5 (4096/8)  | 539 | 550 | 555 |
| C6 (4096/16) | 549 | 506 | 557 |
| C7 (8192/16) | 555 | 555 | 559 |

> C0（BLOCK_SIZE=256，grid=16384 块）单次最慢、`estimate_ms` 抖动最大（3.4 / 1.5 / 1.5 ms），轮数波动最显著。

**结论 ✅**：清缓存后每次运行的 benchmark 执行次数确实不同，`do_bench` 因 `estimate_ms` 受 GPU 瞬时状态影响而非确定。**GPU 与 NPU 行为一致。**

---

## 5. 实验 2：同进程内存缓存

**步骤**：`rm -rf ~/.triton/cache` 后运行 `double`（同进程、相同 shape 连续调用两次）。

**结果**：

```
--- Run 1 (首次) ---
  configs=8 total_calls=4879 time=2.31s
--- Run 2 (相同 shape, 同进程) ---
  configs=0 total_calls=0 time=0.00s
>>> 同进程内第二次运行命中内存缓存 ✓ (无 benchmark)
```

**结论 ✅**：同进程内第二次调用 `configs=0`、耗时 ~0，`Autotuner` 的内存缓存（`self.cache`）生效，跳过整个 benchmark。**GPU 与 NPU 行为一致。**

---

## 6. 实验 3：跨进程磁盘缓存

**步骤**（`export TRITON_CACHE_AUTOTUNING=1` 启用磁盘缓存）：

1. `rm -rf ~/.triton/cache` → 运行 `single`（首次，写磁盘缓存）；
2. **不清理缓存** → 起新进程再运行 `single`（应命中磁盘缓存）。

**结果**：

| 步骤 | configs | total_calls | 耗时 | 说明 |
|------|---------|-------------|------|------|
| Step 1（首次） | 8 | 4944 | 2.32s | 正常 benchmark，写入磁盘 |
| Step 2（新进程） | **0** | **0** | 0.54s | **命中磁盘缓存，跳过 benchmark** |

**磁盘缓存文件确认存在**：

```
~/.triton/cache/EPSMW6M6BIMUXSUAO35PQIRD33KEQPMDSAOXRVAY2HGAPXA6GFQA/add_kernel.autotune.json
```

内容（节选）含完整 `configs_timings`：

```json
{"key": [4194304, "torch.float32", "torch.float32", "torch.float32"],
 "configs_timings": [
   [{"kwargs": {"BLOCK_SIZE": 256}, "num_warps": 4, ...}, [0.0414, 0.0407, 0.0420]],
   [{"kwargs": {"BLOCK_SIZE": 512}, "num_warps": 4, ...}, [0.0398, 0.0394, 0.0404]],
   ...
 ]}
```

**结论 ✅**：GPU triton 3.7.1 的 `check_disk_cache()` 跨进程生效，Step 2 完全跳过 benchmark（`configs=0`）。**此项 GPU 具备、NPU 缺失，正是 issue #1345 的核心。**

> 注：Step 2 耗时 0.54s 非 0，主要是进程启动 + kernel 二进制加载；autotune 本身已 0 调用。

---

## 7. 结论：GPU vs NPU 行为对照

| 行为 | GPU triton (3.7.1, A100) | NPU triton-ascend | 一致性 |
|------|--------------------------|-------------------|--------|
| `do_bench` 次数动态变化 | ✅ 三次 `total_calls=4864/4894/4994` 各异 | （同） | ✅ 一致 |
| 同进程内存缓存 | ✅ 第二次 `configs=0` | （同） | ✅ 一致 |
| 跨进程磁盘缓存 | ✅ 跨进程第二次 `configs=0` | ✗ | ❌ **NPU 缺失** |

**总结论**：issue #1345 描述的三个现象在 GPU triton 上全部可复现且符合预期。前两项（`do_bench` 非确定、同进程内存缓存）是 triton 通用行为，GPU/NPU 一致；第三项跨进程磁盘缓存为 **GPU 独有**，NPU 缺失。

## 8. 原理分析：为什么 GPU 有磁盘缓存而 NPU 没有

**GPU triton（v3.4.0 引入）**，`Autotuner.run()` 中会调用磁盘缓存检查（`python/triton/runtime/autotuner.py`）：

```python
# autotuner.py:39
self.cache_results = (cache_results or knobs.autotuning.cache) \
                     and not knobs.runtime.interpret

# autotuner.py:248-249（run() 内）
if self.cache_results:
    used_cached_result = self.check_disk_cache(key, pruned_configs, benchmark)
```

- `cache_results` 可由装饰器参数 `cache_results=True` 或环境变量 `TRITON_CACHE_AUTOTUNING=1`（即 `knobs.autotuning.cache`）开启；
- `check_disk_cache()`（autotuner.py:175）负责读/写 `~/.triton/cache/{hash}/<name>.autotune.json`，命中则跳过 benchmark。

**NPU triton-ascend**：基于 v3.6.0，父类 `Autotuner` 虽有 `check_disk_cache()`，但其 `AutoTilingTuner`（NPU 自动分块调优器）**完全覆写了 `run()`，未调用父类的 `check_disk_cache()`**，导致 autotune 结果无法持久化到磁盘，跨进程无法命中。这正是 issue #1345 需要修复的点。

---

## 附录：复现命令

```bash
conda activate torch-gpu
cd /root/project/triton

# 实验 1：非确定性（重复 3 次，对比 total_calls）
for i in 1 2 3; do rm -rf ~/.triton/cache; python gpu_autotune_exp.py single; done

# 实验 2：同进程内存缓存
rm -rf ~/.triton/cache
python gpu_autotune_exp.py double

# 实验 3：跨进程磁盘缓存
export TRITON_CACHE_AUTOTUNING=1
rm -rf ~/.triton/cache
python gpu_autotune_exp.py single   # Step 1: 写磁盘（configs=8）
python gpu_autotune_exp.py single   # Step 2: 命中磁盘（configs=0）
ls ~/.triton/cache/*/*.autotune.json
```
