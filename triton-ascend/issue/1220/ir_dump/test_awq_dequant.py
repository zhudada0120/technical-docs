"""
复现 issue #1220: 验证 AWQ 反量化 kernel 中 >> 和 & 的编译产物
核心逻辑: (iweights >> shifts) & 0xF

本文件严格按 vllm 上游源码移植:
  vllm/model_executor/layers/quantization/awq_triton.py
"""
import torch
import torch_npu
import triton
import triton.language as tl

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# ======================== Triton Kernel (vllm 上游精确版) ========================
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
    # 构造 AWQ 反序: [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # 位移量: [0, 16, 4, 20, 8, 24, 12, 28]
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # ═══════════════════════════════════════════════════════════════
    # 核心位运算 (issue #1220 关注点)
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


# ======================== Host 端 Launcher (vllm 上游精确版) ========================
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


# ======================== 测试 ========================
def make_test_data(rows: int, full_cols: int, group_size: int):
    """生成测试数据: 按 AWQ 格式打包的权重/零点/scale"""
    assert full_cols % 8 == 0
    packed_cols = full_cols // 8
    groups = rows // group_size

    # 随机 int4 权重 (0-15)
    w_int4 = torch.randint(0, 16, (rows, full_cols), dtype=torch.int32)
    z_int4 = torch.randint(0, 16, (rows, full_cols), dtype=torch.int32)

    # AWQ 顺序打包
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    qweight = torch.zeros((rows, packed_cols), dtype=torch.int32)
    zeros_packed = torch.zeros((groups, packed_cols), dtype=torch.int32)
    zeros_int4_grouped = z_int4.view(groups, group_size, full_cols)

    for i, order in enumerate(awq_order):
        qweight |= (w_int4[:, i::8] & 0xF) << (order * 4)
        # zeros: 同一 group 内每 row 取相同 packed zeros
        zeros_packed |= (zeros_int4_grouped[:, 0, i::8] & 0xF) << (order * 4)

    scales = torch.randn((groups, full_cols), dtype=torch.float16) * 0.1 + 1.0

    return qweight, scales, zeros_packed, w_int4, z_int4


def make_ref_result(qweight: torch.Tensor, scales: torch.Tensor,
                    zeros_packed: torch.Tensor, group_size: int) -> torch.Tensor:
    """PyTorch 参考实现"""
    rows = qweight.shape[0]
    packed_cols = qweight.shape[1]
    full_cols = packed_cols * 8

    awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
    shifts = awq_order * 4  # [8]

    # 每行独立解包
    iw_list = []
    for s in shifts:
        iw_list.append(((qweight >> s) & 0xF).unsqueeze(-1))
    iw = torch.cat(iw_list, dim=-1).reshape(rows, full_cols)  # [rows, full_cols]

    # zeros 按 group 解包后 broadcast
    z_list = []
    for s in shifts:
        z_list.append(((zeros_packed >> s) & 0xF).unsqueeze(-1))
    z = torch.cat(z_list, dim=-1).reshape(scales.shape[0], full_cols)  # [groups, full_cols]

    # broadcast zeros to all rows in each group
    z = z.repeat_interleave(group_size, dim=0)  # [rows, full_cols]

    return ((iw - z).to(scales.dtype) * scales)


if __name__ == "__main__":
    import os

    print("=== AWQ Dequantize Kernel (vllm 上游精确版) ===")
    print()

    # 测试配置
    rows, full_cols, group_size = 64, 256, 64
    print(f"rows={rows}, full_cols={full_cols}, group_size={group_size}")

    qweight, scales, zeros_packed, _, _ = make_test_data(rows, full_cols, group_size)

    qweight_npu = qweight.npu()
    scales_npu = scales.npu()
    zeros_npu = zeros_packed.npu()

    # PyTorch 参考
    ref = make_ref_result(qweight, scales, zeros_packed, group_size)

    # warmup
    for _ in range(5):
        awq_dequantize_triton(
            qweight_npu, scales_npu, zeros_npu,
            block_size_x=8, block_size_y=16,
        )
    torch.npu.synchronize()

    # msprof profiling 模式: 多跑几次
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
        print("✅ PASS")
    else:
        print("❌ FAIL — 精度差异过大")

    print()
    print("Done.")
