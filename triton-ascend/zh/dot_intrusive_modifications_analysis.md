# dot 函数侵入式修改分析

> 文件：`python/triton/language/semantic.py`，方法：`TritonSemantic.dot`（第 1515–1603 行）
>
> 目标：将 Ascend 特有的修改从上游文件中完全移除，所有修改仅在 `third_party/ascend/` 下实现。

---

## 修改点总览

| # | 行号 | 描述 | 来源 commit |
|---|------|------|-------------|
| 1 | 1523–1526 | dtype 断言中放行了 `tl.int1` | `af7e69d9b` |
| 2 | 1594–1597 | HF32 精度守卫（非 fp32 时静默回退） | `af7e69d9b` → `f6324548d` |
| 3 | 1599–1601 | `max_num_imprecise_acc` 无条件覆盖为 0 | `af7e69d9b` |

---

## 修改点 1：dtype 断言中放行 `tl.int1`

### 修改背景

**Commit**: `af7e69d9b7dcb646a6b29381cbf903da1b08825d`
- 作者：luobaiqing <luobaiqing1@huawei.com>
- 日期：2026-01-22
- 标题：`fix(dot op): support input precision hf32`
- 描述："转测需要，侵入式修改dot，支持hf32"

```diff
- assert lhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32)
+ assert lhs.dtype in (tl.int1, tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32)
```

### 为什么加 `int1`？

这个修改不是误操作，而是与 Ascend 整体 bool 处理策略相关。回顾 Ascend 对 bool 类型的完整链路：

1. **Load 层面**（修改点 8）：`_load_legacy` 加载 `int1` 时内部提升为 `int8`，**不做 cast 回 `int1`**，而是标记 `was_bool_to_int8 = True`。因此从 load 出来的 bool 数据对外表现为 `int8`，走 dot 时自然通过。

2. **二元运算层面**（`logical_and/or/xor` 等，第 438–527 行）：检查 `was_bool_to_int8`，将其 cast 回 `int1` 做逻辑运算。因此逻辑运算产生的结果是**真正的 `int1`**。

3. **mod 运算层面**（第 350–352 行）：对 `was_bool_to_int8` 的输入，直接返回 `int1`（false）。

4. **cast 层面**（第 913–914 行）：`dst_sca_ty.is_bool()` 分支支持转换为 `int1`。

也就是说，Ascend 的计算图中**确实会存在 `int1`（真 bool）类型的 tensor**——来自比较运算、逻辑运算、mod 运算等。如果用户把这些 tensor 传给 `tl.dot`（例如 `mask = a > 0; c = tl.dot(mask, w)`），没有 `int1` 在断言中就会直接报 `"Unsupported lhs dtype int1"`。

加上 `int1` 的意图是**允许这些 `int1` tensor 进入 dot，然后在后续处理中（隐式）提升为 `int8`**。

### 修改是否有必要

**意图合理，但实现不完整。** `int1` 被加入了 dtype 断言（第 1523 行），但后续的整型处理分支（[第 1566–1569 行](python/triton/language/semantic.py#L1566-L1569)）并未同步更新：

```python
if lhs.type.scalar.is_int():
    assert lhs.type.scalar == tl.int8, "only int8 supported!"
```

`int1.is_int()` 返回 `True`（`int1` 在 `UINT_TYPES` 中），但 `== tl.int8` 不成立 → 断言失败。所以 `int1` 虽然通过了 dtype 守卫，却在整型分支被拦截。

**两种修正方向：**

| 方向 | 做法 | 适用场景 |
|------|------|----------|
| A. 删除 `int1` | 从断言中移除 `tl.int1` | 不需要 bool dot 支持，希望在入口处给出明确错误 |
| B. 完善实现 | 在 int 分支中加入对 `int1` 的处理（cast 为 int8） | 需要支持 `int1` 输入到 dot（如 mask 矩阵乘） |

**推荐方向 A**，理由：

1. 上游社区不支持 `int1` dot。
2. Ascend 的 `was_bool_to_int8` 机制已经确保 load 出来的 bool 数据是 int8，最常见的 bool→dot 路径已覆盖。
3. 比较/逻辑运算产生的 `int1` 进入 dot 是罕见场景，用户在入口处收到 `"Unsupported dtype int1"` 的错误比 `"only int8 supported!"` 更清晰，会自然想到用 `.to(tl.int8)` 显式转换。
4. 如果未来确实需要，可以通过 monkey-patch 或完善 int 分支来支持。

### 修改方案

```diff
- assert lhs.dtype in (tl.int1, tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
+ assert lhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                       tl.float64), f"Unsupported lhs dtype {lhs.dtype}"
- assert rhs.dtype in (tl.int1, tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
+ assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                       tl.float64), f"Unsupported rhs dtype {rhs.dtype}"
```

**风险**：如果有用户代码将比较/逻辑运算的 `int1` 结果直接传入 `dot`，删除后会在入口处收到清晰的类型错误（`"Unsupported lhs dtype int1"`），而非之前绕到 `"only int8 supported!"` 的混乱错误。此行为变化是向好的方向。

---

## 修改点 2：HF32 精度守卫

### 修改背景

**两阶段演进：**

- **Phase 1** — commit `af7e69d9b`（2026-01-22）：添加 HF32 守卫，非 fp32 时**抛出 ValueError**：

  ```python
  if (input_precision == getattr(ir.INPUT_PRECISION, "HF32")):
      if (not lhs.dtype.is_fp32() or not rhs.dtype.is_fp32() or not ret_scalar_ty.is_fp32()):
          raise ValueError("input_precision = 'hf32' must be used with f32 * f32 = f32 on Ascend")
  ```

- **Phase 2** — commit `f6324548db0bcc1a37567edf716cc7946d1b1470`（2026-02-25），标题 `fix(dot): support allow_tf32 and set precision to hf32`：

  描述："按照评审要求，开放支持allow_tf32选项，当其为真时设为hf32。当输入tf32时，将其设为hf32。当设置hf32/tf32，且输入不满足fp32要求时，与社区保持一致，忽略input precision，设置默认值"

  将 ValueError 改为**静默回退**：

  ```diff
  - raise ValueError("input_precision = 'hf32' must be used with f32 * f32 = f32 on Ascend")
  + # when input and result is not fp32, ignore input_precision (default is ieee)
  + input_precision = self._str_to_dot_input_precision(self.builder.options.default_dot_input_precision)
  ```

### 修改是否有必要

**有必要，但不能以目前的侵入式方式存在。**

HF32（High-precision Float32）是 Ascend NPU Cube 单元做 FP32 矩阵乘时的内部截断精度格式，类似 NVIDIA 的 TF32。问题是：

- HF32 精度**只对 fp32 × fp32 → fp32** 有意义，如果用户传了 `hf32` 但输入不是纯 fp32（如 fp16 输入、int32 累加器），HF32 无法生效。
- 如果不处理这个情况，用户会在不知情的情况下得到非预期的计算精度（实际走 ieee，而非 hf32）。
- 上游 Triton 没有类似 HF32 的概念，因此上游代码不会做此校验。

**结论：守卫逻辑需要保留，但应该通过 monkey-patch 方式在 `third_party/ascend/` 下实现，保持上游代码零改动。**

### 修改方案

**Monkey-patch `TritonSemantic.dot`（与现有 `_apply_ascend_patch` 模式一致）**

在 `third_party/ascend/backend/__init__.py` 的 `_apply_ascend_patch` 中扩展 patch 逻辑：

```python
def _apply_ascend_patch():
    from triton.compiler.code_generator import CodeGenerator
    from triton.language.semantic import TritonSemantic

    # ---- 现有 patch: CodeGenerator.__init__ ----
    if not getattr(CodeGenerator, "_ascend_patch_applied", False):
        _original_cg_init = CodeGenerator.__init__

        def _patched_cg_init(self, *args, **kwargs):
            _original_cg_init(self, *args, **kwargs)
            options = self.builder.options
            context = self.context
            if hasattr(options, "arch") and options.arch:
                try:
                    builder = ascend_ir.ascendnpu_ir_builder(context, options.arch)
                    target_attr_str = f'#hacc.target<"{options.arch}">'
                    self.module.set_attr("hacc.target", builder.parse_attr(target_attr_str))
                except Exception as e:
                    logging.warning(f"[Ascend Patch] Failed to set hacc.target: {e}")

        CodeGenerator.__init__ = _patched_cg_init
        CodeGenerator._ascend_patch_applied = True

    # ---- 新增 patch: TritonSemantic.dot (HF32 精度守卫) ----
    if not getattr(TritonSemantic, "_ascend_dot_patch_applied", False):
        _original_dot = TritonSemantic.dot

        def _patched_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype):
            # HF32 精度仅对 fp32 x fp32 -> fp32 有效
            # 当输入或输出不是纯 fp32 时，静默回退到默认精度（ieee）
            from triton._C.libtriton import ir
            if (input_precision is not None
                    and input_precision == getattr(ir.INPUT_PRECISION, "HF32", None)):
                if (not lhs.dtype.is_fp32() or not rhs.dtype.is_fp32()
                        or not out_dtype.is_fp32()):
                    input_precision = self._str_to_dot_input_precision(
                        self.builder.options.default_dot_input_precision)
            return _original_dot(self, lhs, rhs, acc, input_precision,
                                 max_num_imprecise_acc, out_dtype)

        TritonSemantic.dot = _patched_dot
        TritonSemantic._ascend_dot_patch_applied = True
```

**方案对比：**

| 方式 | 上游改动 | Ascend 逻辑位置 |
|------|:---:|------|
| Monkey-patch `TritonSemantic.dot`（采纳） | **0 行** | 全部在 `third_party/` |
| `codegen_fns` hook | 3 行 | 全部在 `third_party/` |
| 还原上游无守卫 | 0 行 | 无，但行为不正确 |

**风险说明：**

Monkey-patch 的固有风险是上游 `TritonSemantic.dot` 方法签名变化会导致 patch 失效。缓解措施：

- 通过 `_ascend_dot_patch_applied` 标志位保证只 patch 一次
- 如果上游新增参数，`_patched_dot` 会因参数不匹配而**显式报错**，不会静默异常，便于及时发现和适配
- 参考：现有 `_apply_ascend_patch` 已对 `CodeGenerator.__init__` 采用相同策略，运行稳定

---

## 修改点 3：`max_num_imprecise_acc` 无条件覆盖

### 修改背景

**Commit**: `af7e69d9b`（同修改点 1）

```diff
- # max_num_imprecise_acc only applies to fp8 -> fp32 dot on sm_90
- if max_num_imprecise_acc is None:
-     if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
-         max_num_imprecise_acc = self.builder.options.max_num_imprecise_acc_default
-     else:
-         max_num_imprecise_acc = 0
- else:
-     if lhs.dtype.is_fp8() and rhs.dtype.is_fp8() and max_num_imprecise_acc > K:
-         raise ValueError(...)
+ if max_num_imprecise_acc is not None:
+     print("max_num_imprecise_acc in tl.dot is not supported on Ascend yet. Thus it is ignored.")
+ max_num_imprecise_acc = 0
```

这段修改将上游基于 NVIDIA Hopper（SM90）的 WGMMA 精度控制逻辑完全替换为 Ascend 的无条件覆盖。

### 修改是否有必要

**不必要。可以安全还原上游代码。**

理由：

1. **`max_num_imprecise_acc` 在整个 Ascend 编译管线中被完全忽略。** 全仓 C++ 扫描结果：
   - `third_party/ascend/lib/` 和 `third_party/ascend/*/lib/`：**0 处引用**
   - 所有读取该属性的 C++ 代码均在 NVIDIA 专属路径下：
     - `TritonNvidiaGPU/IR/Ops.cpp`
     - `TritonGPU/Transforms/AccelerateMatmul.cpp`
     - `TritonGPU/Transforms/F32DotTC.cpp`
     - `TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp`
   - Ascend 的 `MatmulConverter`（[TritonOpConverter.cpp:2126]）只读取 `inputPrecision`，**不读取** `maxNumImpreciseAcc`。

2. **`NPUOptions.max_num_imprecise_acc_default = 0`** 已经存在于 [compiler.py:809](third_party/ascend/backend/compiler.py#L809)，覆盖了用户不传值的情况（上游逻辑：`None` → 取默认值 `0`）。

3. 用户显式传值的情况：上游逻辑保留用户值，但 Ascend 后端忽略它，实际结果等同于传 `0`。

### 修改方案

直接还原为上游代码：

```python
# max_num_imprecise_acc only applies to fp8 -> fp32 dot on sm_90
if max_num_imprecise_acc is None:
    if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
        max_num_imprecise_acc = self.builder.options.max_num_imprecise_acc_default
    else:
        max_num_imprecise_acc = 0
else:
    if lhs.dtype.is_fp8() and rhs.dtype.is_fp8() and max_num_imprecise_acc > K:
        raise ValueError(
            f"max_num_imprecise_acc ({max_num_imprecise_acc}) must be <= K ({K})")
```

依赖已存在的 `NPUOptions.max_num_imprecise_acc_default = 0` 即可。

### 风险

还原后，当用户显式传入 `max_num_imprecise_acc` 为非零值时，**不再打印 warning 提示**。由于 Ascend 后端忽略该参数，实际计算行为不受影响（始终为全精度累加），唯一的差异是用户在查看日志时无法看到 "该参数不被 Ascend 支持" 的提示信息。

---

## 总结

| 修改点 | 行动 | 上游改动 | 依赖 |
|--------|------|:---:|------|
| 1. `tl.int1` 放行 | 直接删除 | 0 行 | 无 |
| 2. HF32 精度守卫 | Monkey-patch `TritonSemantic.dot` | 0 行 | 扩展 `_apply_ascend_patch` |
| 3. `max_num_imprecise_acc` 覆盖 | 还原上游代码 | 0 行 | `NPUOptions.max_num_imprecise_acc_default = 0`（已存在） |

最终效果：

- **上游代码改动：0 行**
- `third_party/ascend/backend/__init__.py`：在 `_apply_ascend_patch` 中新增 `_patched_dot`（~15 行）
- `python/triton/language/semantic.py`：删除 3 处侵入式修改，完全还原为上游代码
