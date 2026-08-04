#!/usr/bin/env python3
"""
GPU triton autotune 缓存行为验证脚本。

验证内容:
  experiment1: 清除缓存后多次独立进程运行，对比 benchmark 执行次数是否相同（do_bench 非确定性）
  experiment2: 同进程内两次调用，验证内存缓存是否生效
  experiment3: 开启 cache_results=True / TRITON_CACHE_AUTOTUNING，验证跨进程磁盘缓存是否生效

说明:
  - NPU(triton-ascend) 上空 configs 会触发其 auto-tiling 机制；GPU 没有，所以这里
    给 kernel 显式配了一组 configs（参考社区 vector-add autotune 写法）。
  - 三个实验验证的是 autotune 机制本身的行为，与 kernel 选型无关，故用最简单的 vector-add。

用法:
  python gpu_autotune_exp.py single     # 单次运行（供独立进程调用）
  python gpu_autotune_exp.py double     # 同进程连续两次（验证内存缓存）
  python gpu_autotune_exp.py diskcache  # 打印磁盘缓存验证步骤说明
"""
import os, sys, time, shutil

CACHE_DIR = os.path.expanduser("~/.triton/cache")


def setup():
    """设置 monkey-patch 和 kernel，返回 (triton, _BENCH_LOG, add_kernel, run_autotune)。"""
    import torch
    import triton
    import triton.language as tl

    # --- monkey-patch do_bench: 记录每个 config 的估算耗时与推导出的执行轮数 ---
    # 该估算逻辑与 triton/testing.py 的 do_bench 实现 (3.7.x) 逐行对齐，仅用于“观察”
    # estimate_ms / n_warmup / n_repeat 的取值；真正的 benchmark 仍交给原始 do_bench 完成。
    _original_do_bench = triton.testing.do_bench
    _BENCH_LOG = []

    def instrumented_do_bench(fn, warmup=25, rep=100, **kwargs):
        di = triton.runtime.driver.active.get_device_interface()
        fn()
        di.synchronize()
        cache_buf = triton.runtime.driver.active.get_empty_cache_for_benchmark()
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            triton.runtime.driver.active.clear_cache(cache_buf)
            fn()
        end_event.record()
        di.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
        _BENCH_LOG.append({
            "estimate_ms": round(estimate_ms, 4),
            "n_warmup": n_warmup,
            "n_repeat": n_repeat,
            "total": 5 + n_warmup + n_repeat,
        })
        return _original_do_bench(fn, warmup=warmup, rep=rep, **kwargs)

    triton.testing.do_bench = instrumented_do_bench

    # --- autotune configs: 8 个不同 BLOCK_SIZE / num_warps 组合 ---
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

    # --- kernel: vector-add ---
    @triton.autotune(configs=configs, key=["n_elements"])
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)

    def run_autotune(x, y):
        output = torch.empty_like(x)
        n_elements = output.numel()
        add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)](
            x, y, output, n_elements)
        return output

    return triton, _BENCH_LOG, add_kernel, run_autotune


SIZE = 2**22  # 4M elements（足够让 do_bench 的 estimate_ms 落在 0.01~0.1ms，执行轮数随抖动明显波动）


def _report(_BENCH_LOG, elapsed):
    total_calls = sum(e["total"] for e in _BENCH_LOG)
    print("RESULT: configs={} total_calls={} time={:.2f}s".format(
        len(_BENCH_LOG), total_calls, elapsed))
    for i, e in enumerate(_BENCH_LOG):
        print("  CONFIG_{}: est={}ms warmup={} repeat={} total={}".format(
            i, e["estimate_ms"], e["n_warmup"], e["n_repeat"], e["total"]))


def do_single():
    """单次运行 (用于独立进程实验)。"""
    import torch
    triton, _BENCH_LOG, add_kernel, run_autotune = setup()

    x = torch.rand(SIZE, device="cuda")
    y = torch.rand(SIZE, device="cuda")
    t0 = time.perf_counter()
    output = run_autotune(x, y)
    t1 = time.perf_counter()

    output_torch = x + y
    assert torch.allclose(output, output_torch)
    _report(_BENCH_LOG, t1 - t0)


def do_double():
    """同进程连续两次 (验证内存缓存)。"""
    import torch
    triton, _BENCH_LOG, add_kernel, run_autotune = setup()

    print("--- Run 1 (首次) ---")
    x = torch.rand(SIZE, device="cuda")
    y = torch.rand(SIZE, device="cuda")
    t0 = time.perf_counter()
    output1 = run_autotune(x, y)
    t1 = time.perf_counter()
    r1_configs = len(_BENCH_LOG)
    print("  configs={} total_calls={} time={:.2f}s".format(
        r1_configs, sum(e["total"] for e in _BENCH_LOG), t1 - t0))
    for i, e in enumerate(_BENCH_LOG):
        print("  CONFIG_{}: est={}ms warmup={} repeat={} total={}".format(
            i, e["estimate_ms"], e["n_warmup"], e["n_repeat"], e["total"]))

    _BENCH_LOG.clear()

    print("--- Run 2 (相同 shape, 同进程) ---")
    x2 = torch.rand(SIZE, device="cuda")
    y2 = torch.rand(SIZE, device="cuda")
    t0 = time.perf_counter()
    output2 = run_autotune(x2, y2)
    t1 = time.perf_counter()
    r2_configs = len(_BENCH_LOG)
    print("  configs={} total_calls={} time={:.2f}s".format(
        r2_configs, sum(e["total"] for e in _BENCH_LOG), t1 - t0))

    output_torch = x2 + y2
    assert torch.allclose(output2, output_torch)

    if r2_configs == 0:
        print("\n>>> 同进程内第二次运行命中内存缓存 ✓ (无 benchmark)")
    else:
        print("\n>>> 同进程内内存缓存未生效! (仍有 {} config)".format(r2_configs))


def do_diskcache():
    """打印磁盘缓存验证步骤说明（实际验证通过 shell 两步完成）。"""
    print("""
磁盘缓存验证方法（跨进程）：

  # 开启磁盘缓存（二者等价，任选其一）：
  export TRITON_CACHE_AUTOTUNING=1
  # 或在 @triton.autotune(...) 里加 cache_results=True

  # Step 1: 清除所有缓存，第一次运行（写磁盘缓存）
  rm -rf ~/.triton/cache
  python gpu_autotune_exp.py single
  # 预期: configs=8, total_calls>0（正常 benchmark）

  # Step 2: 不清理磁盘缓存，第二次运行（新进程，应命中磁盘缓存）
  python gpu_autotune_exp.py single
  # 预期: configs=0, total_calls=0（命中磁盘缓存!）

  # 验证缓存文件存在
  ls ~/.triton/cache/*/add_kernel.autotune.json
""")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "single":
        do_single()
    elif mode == "double":
        do_double()
    elif mode == "diskcache":
        do_diskcache()
    else:
        print("Usage: python gpu_autotune_exp.py [single|double|diskcache]")
