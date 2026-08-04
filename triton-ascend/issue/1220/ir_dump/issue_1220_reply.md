# Issue #1220 Reply: Triton Bitwise Operations (>>) Compiled as Scalar Instructions

## Conclusion

Hi @CYa9-a11y, after our analysis, this is **not a triton-ascend IR conversion issue**. The `>>` operator has been correctly lowered to the vector instruction `hivm.hir.vshr` in the triton-ascend compilation pipeline, but the downstream **AscendNPU-IR (bishengir) compiler downgrades it to scalar execution**. We recommend filing an issue for the NPU IR team to continue the investigation: https://gitcode.com/Ascend/AscendNPU-IR/issues

---

## Evidence

### Reproduction

An AWQ dequantization kernel exactly ported from the upstream vllm source, with the core bitwise logic:

```python
@triton.jit
def awq_dequantize_kernel(...):
    # ...
    iweights = (iweights >> shifts) & 0xF   # ← right shift + bitwise AND
    zeros = (zeros >> shifts) & 0xF         # ← right shift + bitwise AND
```

The test script is included in the appendix, tile size `BLOCK_SIZE_X=8, BLOCK_SIZE_Y=16`, shape `rows=64, full_cols=256`.

### Evidence 1: ttadapter IR — bitwise ops correctly lowered to arith dialect (as expected)

In `kernel.ttadapter.mlir`:

```mlir
%iweights_68 = arith.shrsi %iweights_54, %shifts_67 : tensor<16x64xi32>   // >> right shift
%iweights_69 = arith.andi  %iweights_68, %zeros_14 : tensor<16x64xi32>    // &  bitwise AND
%zeros_101  = arith.shrsi %zeros_100,  %shifts_67 : tensor<16x64xi32>    // >> right shift
%zeros_102  = arith.andi  %zeros_101,  %zeros_14 : tensor<16x64xi32>     // &  bitwise AND
```

`arith.shrsi` / `arith.andi` are standard MLIR arithmetic dialect ops. This is the normal behavior of triton-ascend converting Triton IR to upstream MLIR dialect. **No issue at this stage.**

### Evidence 2: NPU IR — `>>` converted to `hivm.hir.vshr`, then downgraded back to `arith.shrsi` by bishengir (root cause)

Warnings at the top of `kernel.npuir.mlir`:

```
loc("iweights"("test_awq_dequant.py":72:28)):
  warning: Op 'hivm.hir.vshr' will execute by scalar instruction with low effiency

loc("zeros"("test_awq_dequant.py":90:22)):
  warning: Op 'hivm.hir.vshr' will execute by scalar instruction with low effiency
```

The actual IR following the warnings:

```mlir
// >> downgraded to a scalar loop + arith.shrsi
hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
scf.for %arg14 = %c0 to %c512 step %c1 {
  scf.for %arg15 = %c0 to %c2 step %c1 {
    %101 = memref.load %collapse_shape_22[%arg14, %arg15] : memref<512x2xi32, ...>
    %102 = memref.load %collapse_shape_23[%arg14, %arg15] : memref<512x2xi32, ...>
    %103 = arith.shrsi %101, %102 : i32          // ← scalar right shift!
    memref.store %103, %collapse_shape_24[%arg14, %arg15] : memref<512x2xi32, ...>
  }
}
hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]

// & remains a proper vector instruction, no downgrade warning
hivm.hir.vand ins(%collapse_shape_25, %collapse_shape_14 : ...) outs(%collapse_shape_26 : ...)
```

Key facts:

- `arith.shrsi` has been converted to `hivm.hir.vshr` via `ArithToHFusion` → `HFusionToHIVM` — the triton-ascend conversion chain is complete
- `arith.andi` has been converted to `hivm.hir.vand` and **remains a vector instruction** (`hivm.hir.vand ins(...) outs(...)`) with no warning
- **`hivm.hir.vshr` is downgraded by the bishengir backend**: it emits a `scf.for` scalar loop + `arith.shrsi` element-by-element scalar operations, accompanied by the `will execute by scalar instruction with low effiency` warning

In summary: triton-ascend correctly lowers both bitwise ops to the HIVM vector dialect. The downgrade to scalar execution happens in bishengir, and only affects `vshr`, not `vand`.

## Full Compilation Pipeline

```
Triton Python:     iweights >> shifts
       ↓  triton-ascend frontend
TTIR:              arith.shrsi              ← ✅ normal
       ↓  ArithToHFusion → HFusionToHIVM
NPU IR (Bisheng):  hivm.hir.vshr            ← ✅ converted to vector op
       ↓  bishengir backend               ← ❌ issue occurs here
Hardware:          Scalar Pipe (PIPE_S)      ← ❌ downgraded to scalar
```

**Issue ownership**: the final lowering of `hivm.hir.vshr` → hardware instructions is performed by **AscendNPU-IR (bishengir compiler)**, and is outside the scope of triton-ascend.

---

## Appendix: Reproduction Test Script

```python
"""
Reproduce issue #1220: verify the compilation output of >> and & in the AWQ dequant kernel.
Core logic: (iweights >> shifts) & 0xF

Exact port from vllm upstream:
  vllm/model_executor/layers/quantization/awq_triton.py
"""
import torch
import torch_npu
import triton
import triton.language as tl

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# ======================== Triton Kernel (exact vllm upstream version) ========================
@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,   # packed int32 weights  [num_rows, num_packed_cols]
    scales_ptr,    # float16 scales         [num_rows // group_size, num_packed_cols * 8]
    zeros_ptr,     # packed int32 zeros     [num_rows // group_size, num_packed_cols]
    group_size,    # per-group quantization size
    result_ptr,    # float16 output         [num_rows, num_packed_cols * 8]
    num_cols,      # packed columns (= K // 8)
    num_rows,      # output rows
    BLOCK_SIZE_X: tl.constexpr,  # tile size in packed cols
    BLOCK_SIZE_Y: tl.constexpr,  # tile size in rows
):
    # ---- pids ----
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # ---- qweight offsets & masks ----
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols
    masks = masks_y[:, None] & masks_x[None, :]

    # ---- result offsets & masks ----
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # ---- load & expand qweight ----
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # ---- AWQ reverse-order shifts ----
    # Build AWQ reverse order: [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Shift amounts: [0, 16, 4, 20, 8, 24, 12, 28]
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # ═══════════════════════════════════════════════════════════════
    # Core bitwise ops (focus of issue #1220)
    # ═══════════════════════════════════════════════════════════════
    iweights = (iweights >> shifts) & 0xF

    # ---- load & expand zeros (per-group) ----
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # ═══════════════════════════════════════════════════════════════
    zeros = (zeros >> shifts) & 0xF

    # ---- load & broadcast scales (per-group) ----
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # ---- dequantize: (w - z) * scale ----
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # ---- store ----
    tl.store(result_ptr + result_offsets, iweights, result_masks)


# ======================== Host Launcher (exact vllm upstream version) ========================
def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 16,
    block_size_y: int = 32,
) -> torch.Tensor:
    """
    Args:
        qweight: [num_rows, num_packed_cols] int32 packed weights
        scales:  [num_rows // group_size, num_packed_cols * 8] float16
        zeros:   [num_rows // group_size, num_packed_cols] int32 packed zeros
    Returns:
        result:  [num_rows, num_packed_cols * 8] float16 dequantized
    """
    Y = qweight.shape[0]  # num_rows
    X = qweight.shape[1]  # num_packed_cols
    group_size = Y // scales.shape[0]

    assert Y > 0 and scales.shape[1] > 0
    assert scales.shape[0] == Y // group_size
    assert zeros.shape[0] == Y // group_size and zeros.shape[1] == X
    assert group_size <= Y

    result = torch.empty(
        Y, X * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )
    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result


# ======================== Test helpers ========================
def make_test_data(rows: int, full_cols: int, group_size: int):
    """Generate test data: AWQ-packed weights, zeros, and scales."""
    assert full_cols % 8 == 0
    packed_cols = full_cols // 8
    groups = rows // group_size

    # Random int4 weights (0-15)
    w_int4 = torch.randint(0, 16, (rows, full_cols), dtype=torch.int32)
    z_int4 = torch.randint(0, 16, (rows, full_cols), dtype=torch.int32)

    # Pack in AWQ order
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    qweight = torch.zeros((rows, packed_cols), dtype=torch.int32)
    zeros_packed = torch.zeros((groups, packed_cols), dtype=torch.int32)
    zeros_int4_grouped = z_int4.view(groups, group_size, full_cols)

    for i, order in enumerate(awq_order):
        qweight |= (w_int4[:, i::8] & 0xF) << (order * 4)
        zeros_packed |= (zeros_int4_grouped[:, 0, i::8] & 0xF) << (order * 4)

    scales = torch.randn((groups, full_cols), dtype=torch.float16) * 0.1 + 1.0

    return qweight, scales, zeros_packed, w_int4, z_int4


def make_ref_result(qweight, scales, zeros_packed, group_size):
    """PyTorch reference implementation."""
    rows = qweight.shape[0]
    packed_cols = qweight.shape[1]
    full_cols = packed_cols * 8

    awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
    shifts = awq_order * 4  # [8]

    iw_list = []
    for s in shifts:
        iw_list.append(((qweight >> s) & 0xF).unsqueeze(-1))
    iw = torch.cat(iw_list, dim=-1).reshape(rows, full_cols)

    z_list = []
    for s in shifts:
        z_list.append(((zeros_packed >> s) & 0xF).unsqueeze(-1))
    z = torch.cat(z_list, dim=-1).reshape(scales.shape[0], full_cols)
    z = z.repeat_interleave(group_size, dim=0)

    return ((iw - z).to(scales.dtype) * scales)


if __name__ == "__main__":
    import os

    print("=== AWQ Dequantize Kernel (exact vllm upstream version) ===")
    print()

    rows, full_cols, group_size = 64, 256, 64
    print(f"rows={rows}, full_cols={full_cols}, group_size={group_size}")

    qweight, scales, zeros_packed, _, _ = make_test_data(rows, full_cols, group_size)

    qweight_npu = qweight.npu()
    scales_npu = scales.npu()
    zeros_npu = zeros_packed.npu()

    ref = make_ref_result(qweight, scales, zeros_packed, group_size)

    # warmup
    for _ in range(5):
        awq_dequantize_triton(
            qweight_npu, scales_npu, zeros_npu,
            block_size_x=8, block_size_y=16,
        )
    torch.npu.synchronize()

    iters = int(os.environ.get("MSPROF_ITERS", 100))
    print(f"Running {iters} iterations for profiling...")
    for _ in range(iters):
        result = awq_dequantize_triton(
            qweight_npu, scales_npu, zeros_npu,
            block_size_x=8, block_size_y=16,
        )
    torch.npu.synchronize()

    result_cpu = result.cpu()
    max_diff = (result_cpu.float() - ref.float()).abs().max().item()
    mean_diff = (result_cpu.float() - ref.float()).abs().mean().item()

    print(f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if max_diff < 0.01:
        print("PASS")
    else:
        print("FAIL - accuracy mismatch")

    print()
    print("Done.")
```