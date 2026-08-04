#!/usr/bin/env python3
"""
验证 issue #1345 的独立进程实验脚本。
用法: python autotune_exp.py single   # 单次运行
      python autotune_exp.py double   # 同进程连续两次
"""
import os, sys, time

import torch
import torch_npu
import triton
import triton.language as tl
import triton.backends.ascend.runtime  # noqa: F401

# --- monkey-patch do_bench ---
_original_do_bench = triton.testing.do_bench
_BENCH_LOG = []


def instrumented_do_bench(fn, warmup=25, rep=100, **kwargs):
    di = triton.runtime.driver.active.get_device_interface()
    fn()
    di.synchronize()
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        triton.runtime.driver.active.clear_cache(cache)
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


# --- kernel (same as 01-vector-add.py) ---
@triton.autotune(configs=[], key=["n_elements"])
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


SIZE = 98432


def do_single():
    """单次运行 (用于独立进程实验)"""
    x = torch.rand(SIZE, device="npu")
    y = torch.rand(SIZE, device="npu")
    t0 = time.perf_counter()
    output = run_autotune(x, y)
    t1 = time.perf_counter()
    output_torch = x + y
    assert torch.allclose(output, output_torch)
    total_calls = sum(e["total"] for e in _BENCH_LOG)
    print("RESULT: configs={} total_calls={} time={:.2f}s".format(
        len(_BENCH_LOG), total_calls, t1 - t0))
    for i, e in enumerate(_BENCH_LOG):
        print("  CONFIG_{}: est={}ms warmup={} repeat={} total={}".format(
            i, e["estimate_ms"], e["n_warmup"], e["n_repeat"], e["total"]))


def do_double():
    """同进程连续两次 (验证内存缓存)"""
    print("--- Run 1 (首次) ---")
    x = torch.rand(SIZE, device="npu")
    y = torch.rand(SIZE, device="npu")
    t0 = time.perf_counter()
    output1 = run_autotune(x, y)
    t1 = time.perf_counter()
    r1_configs = len(_BENCH_LOG)
    r1_total = sum(e["total"] for e in _BENCH_LOG)
    print("  configs={} total_calls={} time={:.2f}s".format(r1_configs, r1_total, t1 - t0))
    for i, e in enumerate(_BENCH_LOG):
        print("  CONFIG_{}: est={}ms warmup={} repeat={} total={}".format(
            i, e["estimate_ms"], e["n_warmup"], e["n_repeat"], e["total"]))

    _BENCH_LOG.clear()

    print("--- Run 2 (相同 shape, 同进程) ---")
    x2 = torch.rand(SIZE, device="npu")
    y2 = torch.rand(SIZE, device="npu")
    t0 = time.perf_counter()
    output2 = run_autotune(x2, y2)
    t1 = time.perf_counter()
    r2_configs = len(_BENCH_LOG)
    r2_total = sum(e["total"] for e in _BENCH_LOG)
    print("  configs={} total_calls={} time={:.2f}s".format(r2_configs, r2_total, t1 - t0))

    output_torch = x2 + y2
    assert torch.allclose(output2, output_torch)

    if r2_configs == 0:
        print("\n>>> 结论: 同进程内第二次运行命中内存缓存 (无 benchmark)")
    else:
        print("\n>>> 结论: 同进程内内存缓存未生效! (仍有 {} config)".format(r2_configs))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "single":
        do_single()
    elif mode == "double":
        do_double()
