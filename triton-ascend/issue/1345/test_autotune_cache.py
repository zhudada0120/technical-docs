#!/usr/bin/env python3
"""
验证 issue #1345 的两个核心问题（修正版）。

关键修正: clear_cache() 只清理磁盘缓存，不清内存中 AutoTilingTuner 的 self.cache。
因此验证「清除缓存后是否次数不同」需要在**独立进程**中运行。

用法:
  python test_autotune_cache.py experiment1   # 两次独立进程，验证执行次数是否不同
  python test_autotune_cache.py experiment2   # 同进程连续两次调用，验证内存缓存
"""

import os
import sys
import shutil
import subprocess
import time

CACHE_DIR = os.path.expanduser("~/.triton/cache")

SCRIPT_CONTENT = r'''
import os
import sys
import time

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

# --- kernel ---
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

mode = sys.argv[1]
size = 98432

if mode == "single":
    # 单次运行 (用于独立进程实验)
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")
    t0 = time.perf_counter()
    output = run_autotune(x, y)
    t1 = time.perf_counter()
    output_torch = x + y
    assert torch.allclose(output, output_torch)
    # 输出结果便于父进程解析
    print(f"RESULT: configs={len(_BENCH_LOG)} total_calls={sum(e['total'] for e in _BENCH_LOG)} time={t1-t0:.2f}s")
    for i, e in enumerate(_BENCH_LOG):
        print(f"  CONFIG_{i}: est={e['estimate_ms']}ms warmup={e['n_warmup']} repeat={e['n_repeat']} total={e['total']}")

elif mode == "double":
    # 同进程连续两次 (用于验证内存缓存)
    print("--- Run 1 (首次) ---")
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")
    t0 = time.perf_counter()
    output1 = run_autotune(x, y)
    t1 = time.perf_counter()
    r1_configs = len(_BENCH_LOG)
    r1_total = sum(e['total'] for e in _BENCH_LOG)
    print(f"  configs={r1_configs} total_calls={r1_total} time={t1-t0:.2f}s")
    for i, e in enumerate(_BENCH_LOG):
        print(f"  CONFIG_{i}: est={e['estimate_ms']}ms warmup={e['n_warmup']} repeat={e['n_repeat']} total={e['total']}")

    _BENCH_LOG.clear()

    print("--- Run 2 (相同 shape, 同进程) ---")
    x2 = torch.rand(size, device="npu")
    y2 = torch.rand(size, device="npu")
    t0 = time.perf_counter()
    output2 = run_autotune(x2, y2)
    t1 = time.perf_counter()
    r2_configs = len(_BENCH_LOG)
    r2_total = sum(e['total'] for e in _BENCH_LOG)
    print(f"  configs={r2_configs} total_calls={r2_total} time={t1-t0:.2f}s")

    output_torch = x2 + y2
    assert torch.allclose(output2, output_torch)

    if r2_configs == 0:
        print("\n>>> 结论: 同进程内第二次运行命中内存缓存 ✓ (无 benchmark)")
    else:
        print(f"\n>>> 结论: 同进程内内存缓存未生效! (仍有 {r2_configs} 个 config benchmark)")
'''

def clear_disk_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"[INFO] 已清除磁盘缓存: {CACHE_DIR}")
    else:
        print(f"[INFO] 磁盘缓存目录不存在")

def run_subprocess(env_cmd, mode, run_label):
    """在独立 Python 进程中运行单次 autotune 测试"""
    full_cmd = (
        f"source /home/zhudada/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate triton-ascend-3.6 && "
        f"source /home/zhudada/miniconda3/envs/cann9.0/Ascend/cann/set_env.sh && "
        f"export PATH=/home/zhudada/project/npuir/tools/bishengir/bin:$PATH && "
        f"python -c \"{SCRIPT_CONTENT}\" single"
    )
    print(f"\n>>> {run_label}: 启动独立进程...")
    result = subprocess.run(
        ["bash", "-c", full_cmd],
        capture_output=True, text=True, timeout=600
    )
    print(result.stdout)
    if result.stderr:
        # Filter out common non-error warnings
        relevant_stderr = [l for l in result.stderr.split('\n')
                          if 'Warning' not in l and 'warn' not in l.lower()
                          and l.strip()]
        if relevant_stderr:
            print("[STDERR]:", '\n'.join(relevant_stderr[:20]))
    return result.returncode == 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    # =========================================================================
    # 实验 1: 两次独立进程，验证清除缓存后执行次数是否不同
    # =========================================================================
    if mode in ("experiment1", "all"):
        print("=" * 70)
        print("实验 1: 清除磁盘缓存 → 独立进程运行两次 → 对比次数")
        print("=" * 70)

        clear_disk_cache()
        ok1 = run_subprocess(SCRIPT_CONTENT, "single", "Run 1")

        clear_disk_cache()
        ok2 = run_subprocess(SCRIPT_CONTENT, "single", "Run 2")

        if ok1 and ok2:
            print("\n>>> 对比: 看两次的 total_calls 值是否相同。如果不同 → 次数确实不确定。")

    # =========================================================================
    # 实验 2: 同进程内两次调用，验证内存缓存
    # =========================================================================
    if mode in ("experiment2", "all"):
        print("\n" + "=" * 70)
        print("实验 2: 同进程连续调用 → 验证内存缓存")
        print("=" * 70)

        clear_disk_cache()
        full_cmd = (
            f"source /home/zhudada/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate triton-ascend-3.6 && "
            f"source /home/zhudada/miniconda3/envs/cann9.0/Ascend/cann/set_env.sh && "
            f"export PATH=/home/zhudada/project/npuir/tools/bishengir/bin:$PATH && "
            f"python -c \"{SCRIPT_CONTENT}\" double"
        )
        result = subprocess.run(
            ["bash", "-c", full_cmd],
            capture_output=True, text=True, timeout=600
        )
        print(result.stdout)
        if result.stderr:
            relevant = [l for l in result.stderr.split('\n')
                       if 'Warning' not in l and 'warn' not in l.lower() and l.strip()]
            if relevant:
                print("[STDERR]:", '\n'.join(relevant[:20]))


if __name__ == "__main__":
    main()
