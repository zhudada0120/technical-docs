# GPU Triton Autotune 行为验证指南

## 验证目的

确认 issue #1345 中描述的两个现象在 GPU triton 上是否同样存在：

| # | 验证项 | 期望结论 |
|---|--------|----------|
| **实验 1** | 清除缓存后，同一 kernel 每次 autotune 执行次数是否不同？ | 预期：**不同**（GPU 和 NPU 的 `do_bench` 逻辑一样） |
| **实验 2** | 同进程内不清理缓存，第二次调用是否命中内存缓存？ | 预期：**命中**（内存缓存 `/self.cache/` 生效） |
| **实验 3** | 开启磁盘缓存后，跨进程是否能命中缓存？ | 预期：**命中**（GPU triton v3.4.0+ 支持） |

## 环境准备

### 1. GPU 机器

需要一张 NVIDIA GPU + CUDA 环境。

### 2. 安装 triton

```bash
pip install triton>=3.4.0
# 或从源码安装:
# git clone https://github.com/triton-lang/triton.git
# cd triton && pip install -e python/
```

### 3. 确认环境

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); \
           import triton; print('triton:', triton.__version__)"
```

期望输出：
```
CUDA: True
triton: 3.x.x
```

### 4. 上传脚本

将 `gpu_autotune_exp.py` 上传到 GPU 机器。

---

## 实验 1: 验证执行次数是否每次不同

**目的**: 验证 `do_bench` 动态计算次数导致的非确定性。

### 步骤

```bash
# Run 1: 清除缓存后第一次运行
rm -rf ~/.triton/cache
python gpu_autotune_exp.py single

# Run 2: 再次清除缓存后第二次运行
rm -rf ~/.triton/cache
python gpu_autotune_exp.py single

# Run 3: 再跑一次（可选，增加置信度）
rm -rf ~/.triton/cache
python gpu_autotune_exp.py single
```

### 预期结果

每次运行输出类似：
```
RESULT: configs=7 total_calls=2738 time=7.91s
  CONFIG_0: est=32.2389ms warmup=1 repeat=3 total=9
  CONFIG_1: est=0.2788ms warmup=89 repeat=358 total=452
  CONFIG_2: est=0.2768ms warmup=90 repeat=361 total=456
  ...
```

### 关注点

| 关注指标 | 预期现象 |
|----------|----------|
| `total_calls` | 两次不同（如 2738 vs 2708） |
| 各 config 的 `n_warmup`/`n_repeat` | 有微小波动（89/358 vs 90/361 vs 87/350） |
| config 数量 | 两次相同（都是 7 个） |

### 结论判断

> **如果 `total_calls` 两次不同 → 执行次数确实不确定，GPU 和 NPU 行为一致。**

---

## 实验 2: 验证同进程内存缓存

**目的**: 验证同进程内第二次调用是否跳过 autotune。

### 步骤

```bash
rm -rf ~/.triton/cache
python gpu_autotune_exp.py double
```

### 预期结果

```
--- Run 1 (首次) ---
  configs=7 total_calls=2738 time=7.41s

--- Run 2 (相同 shape, 同进程) ---
  configs=0 total_calls=0 time=0.00s

>>> 同进程内第二次运行命中内存缓存 ✓ (无 benchmark)
```

### 关注点

Run 2 的 `configs` 必须为 **0**，耗时近乎 0。

### 结论判断

> **如果 Run 2 的 configs=0 → 同进程内存缓存生效，符合预期。**

---

## 实验 3: 验证 GPU triton 磁盘缓存（跨进程）

**目的**: 验证 GPU triton v3.4.0+ 的 `check_disk_cache()` 机制在跨进程场景下是否生效。

### 前置

需要修改 `gpu_autotune_exp.py`，在 `@triton.autotune(...)` 中添加 `cache_results=True`：

```python
# 修改前:
@triton.autotune(configs=[], key=["n_elements"])

# 修改后:
@triton.autotune(configs=[], key=["n_elements"], cache_results=True)
```

或者设置环境变量（效果相同）：

```bash
export TRITON_CACHE_AUTOTUNING=1
```

### 步骤

```bash
# 启用磁盘缓存
export TRITON_CACHE_AUTOTUNING=1

# Step 1: 清除缓存 → 第一次运行（写磁盘）
rm -rf ~/.triton/cache
python gpu_autotune_exp.py single
# 预期: configs=7, total_calls>0（正常 benchmark）

# Step 2: 不清理缓存 → 第二次运行（新进程，应命中磁盘缓存）
python gpu_autotune_exp.py single
# 预期: configs=0, total_calls=0（命中磁盘缓存!）

# 验证缓存文件存在
ls ~/.triton/cache/*/add_kernel.autotune.json
cat ~/.triton/cache/*/add_kernel.autotune.json | python -m json.tool | head -20
```

### 预期结果

Step 1:
```
RESULT: configs=7 total_calls=2738 time=7.91s
```

Step 2:
```
RESULT: configs=0 total_calls=0 time=0.01s
```

### 关注点

| 关注指标 | 预期现象 |
|----------|----------|
| Step 1 的 configs | > 0（首次，需要 benchmark） |
| Step 2 的 configs | **= 0**（命中磁盘缓存，跳过 benchmark） |
| `~/.triton/cache/.../add_kernel.autotune.json` | 存在且包含 `configs_timings` |

### 结论判断

> **如果 Step 2 的 configs=0 → GPU triton 磁盘缓存机制生效，跨进程可以命中。**

---

## 汇总：GPU vs NPU 行为对照

| 行为 | GPU triton (3.7.1, A100) | NPU triton-ascend | 一致性 |
|------|-----------|-------------------|--------|
| `do_bench` 次数动态变化 | ✅ 三次清缓存运行 `total_calls=4864/4894/4994` 各异 |   | ✅ 一致 |
| 同进程内存缓存 | ✅ `double` 第二次 `configs=0` |   | ✅ 一致 |
| 跨进程磁盘缓存 | ✅ 跨进程第二次 `configs=0`（命中 `add_kernel.autotune.json`） | ✗ | ❌ **NPU 缺失** |

> 验证环境：torch 2.12.0+cu130 / triton 3.7.1 / NVIDIA A100-SXM4-80GB / driver 580.126.16。
> 验证脚本：`gpu_autotune_exp.py`（vector-add + 8 个 autotune configs，monkey-patch `do_bench` 记录执行轮数）。
> 三个实验结果均符合预期，确认 GPU triton 的三项行为中前两项与 NPU 一致，第三项（跨进程磁盘缓存）GPU 具备而 NPU 缺失。

---

## 原理简述

### 为什么执行次数不确定？（GPU/NPU 一样）

`triton.testing.do_bench` 的核心逻辑：

```python
# 1. 先跑 5 次估算单次耗时
estimate_ms = elapsed / 5

# 2. 根据耗时动态计算执行轮数
n_warmup = max(1, int(25 / estimate_ms))    # ~25ms 预热
n_repeat = max(1, int(100 / estimate_ms))   # ~100ms 计时

# 3. 执行
for _ in range(n_warmup): fn()          # 预热
for _ in range(n_repeat): fn(); time()  # 计时
```

`estimate_ms` 每次运行时受 GPU 瞬时状态影响（频率、温度、首次 launch 开销等），导致 `n_warmup`/`n_repeat` 每次都略有不同。

### 为什么 GPU 有磁盘缓存而 NPU 没有？

- GPU triton v3.4.0（2025-03）引入了 `Autotuner.check_disk_cache()`，将 autotune 结果持久化到 `~/.triton/cache/{key}/xxx.autotune.json`
- NPU triton-ascend 基于 v3.6.0，父类有此方法，但 `AutoTilingTuner` 完全覆写了 `run()`，没有调用 `check_disk_cache()`
