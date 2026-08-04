# Issue #1345 Response

## Confirmation

Thank you for the feedback. After thorough verification on both NPU and GPU, we confirm that the three phenomena you described do exist on NPU:

1. **Autotune execution count varies after clearing cache** — Confirmed
2. **Cache appears to be unused** — Confirmed (cross-process disk cache is missing)
3. **Unfriendly for performance data collection** — Confirmed, as a result of the above two issues combined

---

## NPU vs. GPU Cross-Validation

To determine whether these behaviors are NPU-specific or inherent to the triton mechanism, we conducted controlled experiments on **GPU triton 3.7.1 (NVIDIA A100)** using the same methodology (vector-add kernel + monkey-patched `do_bench` to record execution rounds). See the attachments for the detailed verification report.

### Results

| Behavior | GPU triton (3.7.1, A100) | NPU triton-ascend | Consistent? |
|----------|--------------------------|-------------------|-------------|
| Dynamic `do_bench` execution count | ✅ Three cache-cleared runs: `total_calls=4864/4894/4994` (varying) |   Same | ✅ **Yes** |
| In-process memory cache | ✅ Second call within same process: `configs=0` (benchmark skipped) |   Same | ✅ **Yes** |
| Cross-process disk cache | ✅ Second call across processes: `configs=0` (hit `add_kernel.autotune.json`) | ✗ Not working | ❌ **NPU missing** |

### Conclusion

- **The first two** (dynamic `do_bench` execution count, in-process memory cache) are general triton mechanisms — GPU and NPU behave consistently.
- **The third** (cross-process disk cache) was introduced in upstream triton v3.4.0 (PR [#6261](https://github.com/triton-lang/triton/pull/6261), `Autotuner.check_disk_cache()`). It is available on GPU but **currently missing on NPU**. This is the root cause of the "cache appears unused" issue.

---

## Why NPU Currently Lacks Autotune Disk Cache

Upstream triton introduced `check_disk_cache()` in `Autotuner` at v3.4.0 (2025-03). When enabled via `cache_results=True` or the environment variable `TRITON_CACHE_AUTOTUNING=1`, autotune results are persisted to `~/.triton/cache/`, allowing cross-process cache hits.

triton-ascend's `AutoTilingTuner` was originally developed based on upstream triton 3.2, before the disk cache mechanism existed (it appeared in v3.4.0). To implement NPU-specific auto-tiling capabilities (automatic kernel AST parsing, candidate tiling configuration generation, etc.), `AutoTilingTuner` completely overrides the parent `Autotuner.run()` method, so disk caching was not involved at that time.

triton-ascend has recently completed the upgrade to upstream v3.6. Although the parent class now has `check_disk_cache()`, this capability does not affect core functionality and has not yet been integrated into `AutoTilingTuner`. This feature requires internal review to establish a plan for alignment with the upstream capability. The review conclusions will be synced in this issue.

---

## Mitigation

### Use `TRITON_BENCH_METHOD=npu` for Deterministic Execution Counts

By default, autotune benchmarking uses `triton.testing.do_bench`, which dynamically computes the number of execution rounds based on an estimated per-invocation latency (`n_warmup = max(1, 25ms/estimate_ms)`), causing variation between runs.

Setting `export TRITON_BENCH_METHOD=npu` switches to the NPU profiling path (`do_bench_npu`), where each config runs a fixed `warmup=5` warmup rounds + `active=30` timed rounds, making the **total execution count deterministic**. This at least ensures that profiling data is reproducible and comparable across runs, facilitating A/B performance analysis.

---

## Note

You are currently using triton-ascend 3.2.2, which is built on upstream triton 3.2. The autotune disk cache mechanism was introduced upstream in v3.4.0. Even when triton-ascend adds this capability in the future, it will be supported on the 3.6 release and will not be backported to 3.2. We recommend keeping an eye on new triton-ascend releases for subsequent cache capability support.
