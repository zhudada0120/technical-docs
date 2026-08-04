loc("iweights"("/home/zhudada/project/triton-ascend/ir_dump/test_awq_dequant.py":72:28)): warning: Op 'hivm.hir.vshr' will execute by scalar instruction with low effiency
loc("zeros"("/home/zhudada/project/triton-ascend/ir_dump/test_awq_dequant.py":90:22)): warning: Op 'hivm.hir.vshr' will execute by scalar instruction with low effiency
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @awq_dequantize_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32 = arith.constant 32 : index
  %c512 = arith.constant 512 : index
  %c77152_i64 = arith.constant 77152 : i64
  %c72384_i64 = arith.constant 72384 : i64
  %c68288_i64 = arith.constant 68288 : i64
  %c67008_i64 = arith.constant 67008 : i64
  %c66752_i64 = arith.constant 66752 : i64
  %c66112_i64 = arith.constant 66112 : i64
  %c65984_i64 = arith.constant 65984 : i64
  %c65664_i64 = arith.constant 65664 : i64
  %c65600_i64 = arith.constant 65600 : i64
  %c77120_i64 = arith.constant 77120 : i64
  %c65568_i64 = arith.constant 65568 : i64
  %c45088_i64 = arith.constant 45088 : i64
  %c73024_i64 = arith.constant 73024 : i64
  %c40992_i64 = arith.constant 40992 : i64
  %c30752_i64 = arith.constant 30752 : i64
  %c28704_i64 = arith.constant 28704 : i64
  %c23584_i64 = arith.constant 23584 : i64
  %c22560_i64 = arith.constant 22560 : i64
  %c72512_i64 = arith.constant 72512 : i64
  %c22048_i64 = arith.constant 22048 : i64
  %c17952_i64 = arith.constant 17952 : i64
  %c544_i64 = arith.constant 544 : i64
  %c1568_i64 = arith.constant 1568 : i64
  %c288_i64 = arith.constant 288 : i64
  %c96_i64 = arith.constant 96 : i64
  %c160_i64 = arith.constant 160 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c64_i32 = arith.constant 64 : i32
  %c15_i32 = arith.constant 15 : i32
  %c4_i32 = arith.constant 4 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c2 = arith.constant 2 : index
  %c1_i32 = arith.constant 1 : i32
  %c40_i32 = arith.constant 40 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = arith.ceildivsi %1, %c40_i32 : i32
  %3 = hivm.hir.get_block_idx -> i64
  %4 = arith.trunci %3 : i64 to i32
  %5 = arith.muli %arg12, %arg11 : i32
  %6 = arith.muli %arg8, %c8_i32 : i32
  %7 = arith.index_cast %arg8 : i32 to index
  %8 = arith.index_cast %arg9 : i32 to index
  %9 = hivm.hir.pointer_cast(%c0_i64) : memref<2xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%9 : memref<2xi32, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c0_i64) : memref<2xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%9, %c4_i32 : memref<2xi32, #hivm.address_space<ub>>, i32) outs(%10 : memref<2xi32, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c32_i64) : memref<2x8x1xi32, #hivm.address_space<ub>>
  %subview = memref.subview %11[0, 0, 0] [2, 2, 1] [1, 1, 1] : memref<2x8x1xi32, #hivm.address_space<ub>> to memref<2x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c2, %c1] outs(%subview : memref<2x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c160_i64) : memref<2x2x8x1xi32, #hivm.address_space<ub>>
  %subview_0 = memref.subview %12[0, 0, 0, 0] [2, 2, 2, 1] [1, 1, 1, 1] : memref<2x2x8x1xi32, #hivm.address_space<ub>> to memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>>
  %13 = hivm.hir.pointer_cast(%c96_i64) : memref<1x2x8x1xi32, #hivm.address_space<ub>>
  %subview_1 = memref.subview %13[0, 0, 0, 0] [1, 2, 2, 1] [1, 1, 1, 1] : memref<1x2x8x1xi32, #hivm.address_space<ub>> to memref<1x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>>
  %expand_shape = memref.expand_shape %10 [[0, 1]] output_shape [1, 2] : memref<2xi32, #hivm.address_space<ub>> into memref<1x2xi32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %subview_1 [[0, 1], [2]] : memref<1x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>> into memref<2x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<1x2xi32, #hivm.address_space<ub>>) outs(%collapse_shape : memref<2x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%subview_1 : memref<1x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>>) outs(%subview_0 : memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
  %expand_shape_2 = memref.expand_shape %subview [[0], [1, 2]] output_shape [2, 2, 1] : memref<2x2xi32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<2x2x1xi32, strided<[8, 1, 1]>, #hivm.address_space<ub>>
  %14 = hivm.hir.pointer_cast(%c160_i64) : memref<2x2x8x1xi32, #hivm.address_space<ub>>
  %subview_3 = memref.subview %14[0, 0, 0, 0] [2, 2, 2, 1] [1, 1, 1, 1] : memref<2x2x8x1xi32, #hivm.address_space<ub>> to memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>>
  %15 = hivm.hir.pointer_cast(%c288_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%subview_0, %expand_shape_2 : memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>>, memref<2x2x1xi32, strided<[8, 1, 1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>>) temp_buffer(%15 : memref<64xi32, #hivm.address_space<ub>>) broadcast = [2]
  %16 = hivm.hir.pointer_cast(%c160_i64) : memref<2x2x8x1xi32, #hivm.address_space<ub>>
  %subview_4 = memref.subview %16[0, 0, 0, 0] [2, 2, 2, 1] [1, 1, 1, 1] : memref<2x2x8x1xi32, #hivm.address_space<ub>> to memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>> into memref<4x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %subview_4 [[0, 1], [2]] : memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>> into memref<4x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%collapse_shape_5, %c4_i32 : memref<4x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>, i32) outs(%collapse_shape_6 : memref<4x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %expand_shape_7 = memref.expand_shape %subview_4 [[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 1, 2, 2, 2] : memref<2x2x2xi32, strided<[16, 8, 1]>, #hivm.address_space<ub>> into memref<1x1x1x2x2x2xi32, strided<[32, 32, 32, 16, 8, 1]>, #hivm.address_space<ub>>
  %17 = hivm.hir.pointer_cast(%c1568_i64) : memref<16x1x8x2x2x8x1xi32, #hivm.address_space<ub>>
  %subview_8 = memref.subview %17[0, 0, 0, 0, 0, 0, 0] [16, 1, 8, 2, 2, 2, 1] [1, 1, 1, 1, 1, 1, 1] : memref<16x1x8x2x2x8x1xi32, #hivm.address_space<ub>> to memref<16x1x8x2x2x2xi32, strided<[256, 256, 32, 16, 8, 1]>, #hivm.address_space<ub>>
  %18 = hivm.hir.pointer_cast(%c544_i64) : memref<1x1x8x2x2x8x1xi32, #hivm.address_space<ub>>
  %subview_9 = memref.subview %18[0, 0, 0, 0, 0, 0, 0] [1, 1, 8, 2, 2, 2, 1] [1, 1, 1, 1, 1, 1, 1] : memref<1x1x8x2x2x8x1xi32, #hivm.address_space<ub>> to memref<1x1x8x2x2x2xi32, strided<[256, 256, 32, 16, 8, 1]>, #hivm.address_space<ub>>
  %collapse_shape_10 = memref.collapse_shape %expand_shape_7 [[0, 1, 2], [3, 4], [5]] : memref<1x1x1x2x2x2xi32, strided<[32, 32, 32, 16, 8, 1]>, #hivm.address_space<ub>> into memref<1x4x2xi32, strided<[32, 8, 1]>, #hivm.address_space<ub>>
  %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1, 2], [3, 4], [5]] : memref<1x1x8x2x2x2xi32, strided<[256, 256, 32, 16, 8, 1]>, #hivm.address_space<ub>> into memref<8x4x2xi32, strided<[32, 8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%collapse_shape_10 : memref<1x4x2xi32, strided<[32, 8, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<8x4x2xi32, strided<[32, 8, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
  %collapse_shape_12 = memref.collapse_shape %subview_9 [[0, 1], [2, 3, 4], [5]] : memref<1x1x8x2x2x2xi32, strided<[256, 256, 32, 16, 8, 1]>, #hivm.address_space<ub>> into memref<1x32x2xi32, strided<[256, 8, 1]>, #hivm.address_space<ub>>
  %collapse_shape_13 = memref.collapse_shape %subview_8 [[0, 1], [2, 3, 4], [5]] : memref<16x1x8x2x2x2xi32, strided<[256, 256, 32, 16, 8, 1]>, #hivm.address_space<ub>> into memref<16x32x2xi32, strided<[256, 8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%collapse_shape_12 : memref<1x32x2xi32, strided<[256, 8, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<16x32x2xi32, strided<[256, 8, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
  %19 = hivm.hir.pointer_cast(%c17952_i64) : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>>
  %collapse_shape_14 = memref.collapse_shape %19 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c15_i32 : i32) outs(%collapse_shape_14 : memref<1024xi32, #hivm.address_space<ub>>)
  %20 = arith.index_cast %6 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg13 = %c0_i32 to %2 step %c1_i32  : i32 {
    %21 = arith.index_cast %arg13 : i32 to index
    %22 = arith.index_cast %c0_i32 : i32 to index
    %23 = arith.index_cast %2 : i32 to index
    %24 = arith.index_cast %c1_i32 : i32 to index
    %25 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%21, %22, %24]
    %26 = arith.index_cast %25 : index to i1
    %c0_i64_15 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %27 = arith.select %26, %c0_i64_15, %c1_i64 : i64
    %28 = hivm.hir.pointer_cast(%c22048_i64, %c72512_i64) : memref<16x8xi32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<16x8xi32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c65568_i64, %c77120_i64) : memref<1x8xi32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<1x8xi32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c40992_i64, %c73024_i64) : memref<16x1x8x2x2x2xf16, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<16x1x8x2x2x2xf16, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c72384_i64, %c77152_i64) : memref<1x8x2x2x2xf16, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<1x8x2x2x2xf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %32 = arith.muli %arg13, %c40_i32 : i32
    %33 = arith.addi %32, %4 : i32
    %34 = arith.minsi %33, %1 : i32
    %35 = arith.divsi %34, %arg12 : i32
    %36 = arith.remsi %35, %arg11 : i32
    %37 = arith.divsi %34, %5 : i32
    %38 = arith.remsi %37, %arg10 : i32
    %39 = arith.muli %36, %c16_i32 : i32
    %40 = arith.muli %38, %c8_i32 : i32
    %41 = arith.muli %38, %c64_i32 : i32
    %42 = arith.index_cast %39 : i32 to index
    %43 = arith.index_cast %40 : i32 to index
    %44 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%43, %42, %7]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [16, 8], strides: [%7, 1] : memref<?xi32, #hivm.address_space<gm>> to memref<16x8xi32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %45 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%42, %8]
    %46 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%45, %42]
    %47 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%43, %7]
    %48 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%47, %43]
    %49 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%46, %42]
    %50 = affine.max affine_map<()[s0] -> (0, s0)>()[%49]
    %51 = affine.min affine_map<()[s0, s1] -> (8, s0 - s1)>()[%48, %43]
    %52 = affine.max affine_map<()[s0] -> (0, s0)>()[%51]
    %53 = arith.cmpi slt, %50, %c16 : index
    %54 = arith.cmpi slt, %52, %c8 : index
    %55 = arith.ori %53, %54 : i1
    %subview_16 = memref.subview %reinterpret_cast[0, 0] [%50, %52] [1, 1] : memref<16x8xi32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xi32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_17 = memref.subview %28[0, 0] [%50, %52] [1, 1] : memref<16x8xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[8, 1]>, #hivm.address_space<ub>>
    scf.if %55 {
      %collapse_shape_51 = memref.collapse_shape %28 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%c0_i32 : i32) outs(%collapse_shape_51 : memref<128xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_16 : memref<?x?xi32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_17 : memref<?x?xi32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %56 = hivm.hir.pointer_cast(%c22560_i64) : memref<16x8x2xi32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %28 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %56 [[0, 1, 2]] : memref<16x8x2xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
    %57 = hivm.hir.pointer_cast(%c23584_i64) : memref<1280xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vinterleave ins(%collapse_shape_18, %collapse_shape_18 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<256xi32, #hivm.address_space<ub>>) interleave_channel_nums = 2 temp_buffer(%57 : memref<1280xi32, #hivm.address_space<ub>>)
    %58 = hivm.hir.pointer_cast(%c28704_i64) : memref<16x8x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %58 [[0, 1, 2, 3]] : memref<16x8x2x2xi32, #hivm.address_space<ub>> into memref<512xi32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c30752_i64) : memref<2560xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vinterleave ins(%collapse_shape_19, %collapse_shape_19 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<512xi32, #hivm.address_space<ub>>) interleave_channel_nums = 2 temp_buffer(%59 : memref<2560xi32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c40992_i64, %c73024_i64) : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %60 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c45088_i64) : memref<5120xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %27]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vinterleave ins(%collapse_shape_20, %collapse_shape_20 : memref<512xi32, #hivm.address_space<ub>>, memref<512xi32, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<1024xi32, #hivm.address_space<ub>>) interleave_channel_nums = 2 temp_buffer(%61 : memref<5120xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %62 = hivm.hir.pointer_cast(%c40992_i64, %c73024_i64) : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %60 [[0, 1, 2, 3, 4], [5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<512x2xi32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %subview_8 [[0, 1, 2, 3, 4], [5]] : memref<16x1x8x2x2x2xi32, strided<[256, 256, 32, 16, 8, 1]>, #hivm.address_space<ub>> into memref<512x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %62 [[0, 1, 2, 3, 4], [5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<512x2xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg14 = %c0 to %c512 step %c1 {
      scf.for %arg15 = %c0 to %c2 step %c1 {
        %101 = memref.load %collapse_shape_22[%arg14, %arg15] : memref<512x2xi32, #hivm.address_space<ub>>
        %102 = memref.load %collapse_shape_23[%arg14, %arg15] : memref<512x2xi32, strided<[8, 1]>, #hivm.address_space<ub>>
        %103 = arith.shrsi %101, %102 : i32
        memref.store %103, %collapse_shape_24[%arg14, %arg15] : memref<512x2xi32, #hivm.address_space<ub>>
      }
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %63 = hivm.hir.pointer_cast(%c40992_i64, %c73024_i64) : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %62 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %63 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vand ins(%collapse_shape_25, %collapse_shape_14 : memref<1024xi32, #hivm.address_space<ub>>, memref<1024xi32, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<1024xi32, #hivm.address_space<ub>>)
    %64 = arith.divsi %39, %arg6 : i32
    %65 = arith.muli %arg8, %64 : i32
    %66 = arith.divsi %arg9, %arg6 : i32
    %67 = arith.cmpi slt, %64, %66 : i32
    %68 = arith.index_cast %65 : i32 to index
    %69 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%68, %43]
    %reinterpret_cast_27 = memref.reinterpret_cast %arg5 to offset: [%69], sizes: [1, 8], strides: [8, 1] : memref<?xi32, #hivm.address_space<gm>> to memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>
    %70 = arith.index_castui %67 : i1 to index
    %71 = affine.min affine_map<()[s0] -> (1, s0)>()[%70]
    %72 = affine.max affine_map<()[s0] -> (0, s0)>()[%71]
    %73 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%70, %48, %43]
    %74 = affine.max affine_map<()[s0] -> (0, s0)>()[%73]
    %75 = arith.cmpi slt, %72, %c1 : index
    %76 = arith.cmpi slt, %74, %c8 : index
    %77 = arith.ori %75, %76 : i1
    %subview_28 = memref.subview %reinterpret_cast_27[0, 0] [%72, %74] [1, 1] : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xi32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_29 = memref.subview %29[0, 0] [%72, %74] [1, 1] : memref<1x8xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[8, 1]>, #hivm.address_space<ub>>
    scf.if %77 {
      %collapse_shape_51 = memref.collapse_shape %29 [[0, 1]] : memref<1x8xi32, #hivm.address_space<ub>> into memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%c0_i32 : i32) outs(%collapse_shape_51 : memref<8xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_28 : memref<?x?xi32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_29 : memref<?x?xi32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %78 = hivm.hir.pointer_cast(%c65600_i64) : memref<1x8x2xi32, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %29 [[0, 1]] : memref<1x8xi32, #hivm.address_space<ub>> into memref<8xi32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %78 [[0, 1, 2]] : memref<1x8x2xi32, #hivm.address_space<ub>> into memref<16xi32, #hivm.address_space<ub>>
    %79 = hivm.hir.pointer_cast(%c65664_i64) : memref<80xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vinterleave ins(%collapse_shape_30, %collapse_shape_30 : memref<8xi32, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<16xi32, #hivm.address_space<ub>>) interleave_channel_nums = 2 temp_buffer(%79 : memref<80xi32, #hivm.address_space<ub>>)
    %80 = hivm.hir.pointer_cast(%c65984_i64) : memref<1x8x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %80 [[0, 1, 2, 3]] : memref<1x8x2x2xi32, #hivm.address_space<ub>> into memref<32xi32, #hivm.address_space<ub>>
    %81 = hivm.hir.pointer_cast(%c66112_i64) : memref<160xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vinterleave ins(%collapse_shape_31, %collapse_shape_31 : memref<16xi32, #hivm.address_space<ub>>, memref<16xi32, #hivm.address_space<ub>>) outs(%collapse_shape_32 : memref<32xi32, #hivm.address_space<ub>>) interleave_channel_nums = 2 temp_buffer(%81 : memref<160xi32, #hivm.address_space<ub>>)
    %82 = hivm.hir.pointer_cast(%c66752_i64) : memref<1x8x2x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_33 = memref.collapse_shape %82 [[0, 1, 2, 3, 4]] : memref<1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<64xi32, #hivm.address_space<ub>>
    %83 = hivm.hir.pointer_cast(%c67008_i64) : memref<320xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vinterleave ins(%collapse_shape_32, %collapse_shape_32 : memref<32xi32, #hivm.address_space<ub>>, memref<32xi32, #hivm.address_space<ub>>) outs(%collapse_shape_33 : memref<64xi32, #hivm.address_space<ub>>) interleave_channel_nums = 2 temp_buffer(%83 : memref<320xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %84 = hivm.hir.pointer_cast(%c68288_i64) : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_34 = memref.collapse_shape %82 [[0], [1, 2, 3], [4]] : memref<1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<1x32x2xi32, #hivm.address_space<ub>>
    %collapse_shape_35 = memref.collapse_shape %84 [[0, 1], [2, 3, 4], [5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<16x32x2xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg14 = %c0 to %c16 step %c1 {
      scf.for %arg15 = %c0 to %c32 step %c1 {
        scf.for %arg16 = %c0 to %c2 step %c1 {
          %101 = memref.load %collapse_shape_34[%c0, %arg15, %arg16] : memref<1x32x2xi32, #hivm.address_space<ub>>
          %102 = memref.load %collapse_shape_13[%arg14, %arg15, %arg16] : memref<16x32x2xi32, strided<[256, 8, 1]>, #hivm.address_space<ub>>
          %103 = arith.shrsi %101, %102 : i32
          memref.store %103, %collapse_shape_35[%arg14, %arg15, %arg16] : memref<16x32x2xi32, #hivm.address_space<ub>>
        }
      }
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %85 = hivm.hir.pointer_cast(%c68288_i64) : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_36 = memref.collapse_shape %84 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %collapse_shape_37 = memref.collapse_shape %85 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vand ins(%collapse_shape_36, %collapse_shape_14 : memref<1024xi32, #hivm.address_space<ub>>, memref<1024xi32, #hivm.address_space<ub>>) outs(%collapse_shape_37 : memref<1024xi32, #hivm.address_space<ub>>)
    %86 = arith.muli %6, %64 : i32
    %87 = arith.index_cast %86 : i32 to index
    %88 = arith.index_cast %41 : i32 to index
    %89 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%87, %88]
    %reinterpret_cast_38 = memref.reinterpret_cast %arg4 to offset: [%89], sizes: [1, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_39 = memref.collapse_shape %31 [[0], [1, 2, 3, 4]] : memref<1x8x2x2x2xf16, #hivm.address_space<ub>> into memref<1x64xf16, #hivm.address_space<ub>>
    %90 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%88, %20]
    %91 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%90, %88]
    %92 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 64)>()[%70, %91, %88]
    %93 = affine.max affine_map<()[s0] -> (0, s0)>()[%92]
    %94 = arith.cmpi slt, %93, %c64 : index
    %95 = arith.ori %75, %94 : i1
    %subview_40 = memref.subview %reinterpret_cast_38[0, 0] [%72, %93] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_41 = memref.subview %collapse_shape_39[0, 0] [%72, %93] [1, 1] : memref<1x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    scf.if %95 {
      %collapse_shape_51 = memref.collapse_shape %31 [[0, 1, 2, 3, 4]] : memref<1x8x2x2x2xf16, #hivm.address_space<ub>> into memref<64xf16, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape_51 : memref<64xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_40 : memref<?x?xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_41 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %96 = hivm.hir.pointer_cast(%c40992_i64, %c73024_i64) : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>>
    %collapse_shape_42 = memref.collapse_shape %96 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%collapse_shape_26, %collapse_shape_37 : memref<1024xi32, #hivm.address_space<ub>>, memref<1024xi32, #hivm.address_space<ub>>) outs(%collapse_shape_42 : memref<1024xi32, #hivm.address_space<ub>>)
    %97 = hivm.hir.pointer_cast(%c40992_i64, %c73024_i64) : memref<16x1x8x2x2x2xf32, #hivm.address_space<ub>>
    %collapse_shape_43 = memref.collapse_shape %97 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_42 : memref<1024xi32, #hivm.address_space<ub>>) outs(%collapse_shape_43 : memref<1024xf32, #hivm.address_space<ub>>)
    %collapse_shape_44 = memref.collapse_shape %30 [[0, 1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xf16, #hivm.address_space<ub>> into memref<1024xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_43 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_44 : memref<1024xf16, #hivm.address_space<ub>>)
    %collapse_shape_45 = memref.collapse_shape %30 [[0, 1], [2, 3, 4, 5]] : memref<16x1x8x2x2x2xf16, #hivm.address_space<ub>> into memref<16x64xf16, #hivm.address_space<ub>>
    %collapse_shape_46 = memref.collapse_shape %30 [[0, 1], [2, 3, 4, 5]] : memref<16x1x8x2x2x2xf16, #hivm.address_space<ub>> into memref<16x64xf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_45, %collapse_shape_39 : memref<16x64xf16, #hivm.address_space<ub>>, memref<1x64xf16, #hivm.address_space<ub>>) outs(%collapse_shape_46 : memref<16x64xf16, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %collapse_shape_47 = memref.collapse_shape %30 [[0], [1, 2, 3, 4, 5]] : memref<16x1x8x2x2x2xf16, #hivm.address_space<ub>> into memref<16x64xf16, #hivm.address_space<ub>>
    %98 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%88, %42, %20]
    %reinterpret_cast_48 = memref.reinterpret_cast %arg7 to offset: [%98], sizes: [16, 64], strides: [%20, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<16x64xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %99 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%91, %88]
    %100 = affine.max affine_map<()[s0] -> (0, s0)>()[%99]
    %subview_49 = memref.subview %collapse_shape_47[0, 0] [%50, %100] [1, 1] : memref<16x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_50 = memref.subview %reinterpret_cast_48[0, 0] [%50, %100] [1, 1] : memref<16x64xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_49 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_50 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %27]
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
