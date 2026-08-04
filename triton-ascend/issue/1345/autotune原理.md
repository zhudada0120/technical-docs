# Triton Ascend Autotune 自动寻优原理

## 1. 整体流程概览

```
用户调用 kernel
      │
      ▼
AutoTilingTuner.run(*args, **kwargs)          ← 入口，被 @triton.autotune 装饰
      │
      ├─ generate_key_and_configs()            ← 第 1 步：生成 tuning key + 候选 configs
      │     │
      │     ├─ 构造 key: (n_elements, dtype)   ← 相同的 key → 复用缓存结果
      │     │
      │     ├─ 解析 kernel AST                ← 自动识别 split/tiling/reduction 轴
      │     │
      │     └─ TileGenerator 生成候选 configs  ← 二分递减生成多种分块大小
      │
      ├─ key in self.cache?                   ← 第 2 步：查内存缓存
      │     ├─ 命中 → 直接用最优 config       ← 同进程内有效
      │     └─ 未命中 ↓
      │
      ├─ _batch_bench(configs)                 ← 第 3 步：逐个 benchmark
      │     │
      │     ├─ 并行编译所有 config            ← ThreadPoolExecutor + AsyncCompileMode
      │     │
      │     └─ 对每个 config 调 do_bench()    ← 测量耗时
      │           └─ 5 次估算 + n_warmup 预热 + n_repeat 计时
      │
      ├─ 选最优 config → 存 self.cache[key]   ← 仅内存，不写磁盘
      │
      └─ 用最优 config 真正执行一次 kernel    ← 业务逻辑
```

---

## 2. 入口：`@triton.autotune` 装饰器

在 Ascend 平台上，`triton.autotune` 被 [third_party/ascend/backend/runtime/__init__.py]() 中的 `_patch_autotune()` 替换为 Ascend 版本：

```python
# 用户代码
@triton.autotune(configs=[], key=["n_elements"])   # configs=[] → 自动生成 config
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

实际创建的是 `AutoTilingTuner` 对象（继承自 `Autotuner`）。

### 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `configs` | 用户指定的候选配置列表。`[]` 表示自动生成 | `[Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4)]` |
| `key` | 哪些参数变化时需要重新 tuning | `["n_elements"]` |
| `hints` | 用户提示，辅助 AST 解析（可选） | `{"axes": {"x": "n_elements"}}` |

---

## 3. 第一步：生成候选 Config（`generate_key_and_configs`）

[third_party/ascend/backend/runtime/autotuner.py:1962](third_party/ascend/backend/runtime/autotuner.py#L1962)

### 3.1 构造 Tuning Key

```python
key = [_args[arg_name] for arg_name in self.keys if arg_name in _args]
# 例: key = [98432]   (n_elements 的值)

for _, arg in _args.items():
    if hasattr(arg, "dtype"):
        key.append(str(arg.dtype))
# 例: key = [98432, "torch.float32"]

key = tuple(key)   # 最终: (98432, "torch.float32")
```

**key 的作用**：相同 key → 相同的最优 config。如果 `n_elements` 变了（比如 98432 → 65536），就需要重新 tuning。

### 3.2 解析 Kernel AST，识别轴信息

当 `auto_gen_config=True`（即用户没提供 configs，或 hints 要求自动生成）时：

```python
cv_parse_result = self._autoparse_axis_params(all_args)
```

这一步会**解析 kernel 的 Python 源码 AST**，自动识别：

| 轴类型 | 含义 | 示例 |
|--------|------|------|
| **split axis** | 被 `program_id` 分块的轴 | `pid = tl.program_id(0); block_start = pid * BLOCK_SIZE` |
| **tiling axis** | 块内循环的轴 | `for loop1 in range(loops1): ... tl.arange(0, XBLOCK_SUB)` |
| **reduction axis** | 规约轴 | `tl.sum(x, axis=1)` |
| **low-dim axis** | 低维度轴 | 最内层循环对应的维度 |

解析结果决定 kernel 是 **vector / cube / mix** 三种模式之一。

### 3.3 生成候选分块大小（TileGenerator）

以 vector 模式为例，[tile_generator.py]() 的 `descend_split_tiling()`：

```
初始: BLOCK_SIZE = n_elements (例: 98432)

二分递减:
  BLOCK_SIZE=98432  →  生成 config
  BLOCK_SIZE=49216  →  生成 config
  BLOCK_SIZE=24608  →  生成 config
  BLOCK_SIZE=12304  →  生成 config
  ...
  直到 BLOCK_SIZE < min(1024//dtype_bytes, ...)  →  停止
```

每个 BLOCK_SIZE 生成一个 `Config`：
```python
Config(kwargs={"BLOCK_SIZE": 98432}, num_warps=1, num_stages=1)
```

这些 config 还会被**扩展**：
- SIMD 模式：每个 config 复制一份 toggle `multibuffer`（`num_stages` = 1 或 2）
- 如果有 `hints`：对 hint 参数做笛卡尔积展开

最终得到 7 个候选 config（例如实验中的情况）。

---

## 4. 第二步：查内存缓存

[autotuner.py:2025](third_party/ascend/backend/runtime/autotuner.py#L2025)

```python
if key not in self.cache:
    # cache miss → 需要 benchmark
    ...
else:
    config = self.cache[key]   # cache hit → 直接用历史最优
```

**关键事实**：
- `self.cache` 是 Python `dict`，只在当前进程内存中
- 进程退出 → 缓存消失
- **不写磁盘**，父类 `Autotuner` 有 `check_disk_cache()` 方法可以持久化到 `~/.triton/cache/`，但 `AutoTilingTuner` **从不调用它**

---

## 5. 第三步：Benchmark 所有候选（`_batch_bench`）

[autotuner.py:2094](third_party/ascend/backend/runtime/autotuner.py#L2094)

### 5.1 并行编译

```python
# 对每个 config 调用 fn.run(...) 触发 JIT 编译
kernels_call = {
    config: self._make_kernel_call(*args, config=config, **kwargs)
    for config in configs
}

# 使用线程池并行编译所有 config
with ThreadPoolExecutor(max_workers=N) as executor, triton.AsyncCompileMode(executor):
    for config, fn in kernels_call.items():
        future_kernels.append((config, fn(warmup=True)))   # 并行!
```

编译结果（kernel .so 文件）会被缓存到 `~/.triton/cache/`，但 **autotune 的最优 config 选择结果不会**。

### 5.2 逐 config Benchmark

```python
for config, fn in run_fns.items():
    timings = do_bench(fn, quantiles=(0.5, 0.2, 0.8))
```

`do_bench` 的工作方式（见前面对话）：
```
5 次估算 → 计算 n_warmup、n_repeat → n_warmup 次预热 → n_repeat 次计时测量
```

每个 config 得到一个 `(median, p20, p80)` 的耗时元组。

### 5.3 选最优

```python
self.cache[key] = builtins.min(timings, key=timings.get)  # 取 median 最小的 config
```

---

## 6. 最后：用最优 Config 执行一次真正的 Kernel

```python
config = self.cache[key]          # 最优 config
self.fn.run(*args, **kwargs, **config.all_kwargs())  # 真正执行
```

这一行就是业务逻辑真正需要的 kernel 执行。

---

## 7. 为什么会出现 issue 中的现象

### 7.1 两次独立进程 → 每次都重新 autotune

```
进程 A: self.cache = {}  →  cache miss → benchmark 7 个 config → self.cache = {key: best_config}
        进程退出，self.cache 消失

进程 B: self.cache = {}  →  cache miss → benchmark 7 个 config → self.cache = {key: best_config}
        进程退出，self.cache 消失
```

因为没有磁盘持久化，进程 B 不知道进程 A 已经 benchmark 过了。

### 7.2 两次 benchmark 次数不同

每次 `do_bench` 都独立估算 `estimate_ms`，受系统瞬时状态影响，算出的 `n_warmup`/`n_repeat` 不同。

---

## 8. 总结：三张缓存表

| 缓存层 | 位置 | 内容 | 生命周期 | 谁在用 |
|--------|------|------|----------|--------|
| **编译缓存** | `~/.triton/cache/{hash}/` | 编译好的 kernel .so 文件 | 持久化，除非手动清除 | `JITFunction._do_compile` |
| **Autotune 磁盘缓存** | `~/.triton/cache/{key}/xxx.autotune.json` | 最优 config + benchmark 计时 | 持久化（父类有此能力） | `Autotuner.check_disk_cache`（但 `AutoTilingTuner` 不调用） |
| **Autotune 内存缓存** | `AutoTilingTuner.cache: Dict` | 当前进程的最优 config | 进程退出即消失 | `AutoTilingTuner.run` |

**核心矛盾**：第三层（内存）是唯一实际生效的 autotune 缓存，但它不持久化；第二层（磁盘）有能力持久化，但从未被调用。
