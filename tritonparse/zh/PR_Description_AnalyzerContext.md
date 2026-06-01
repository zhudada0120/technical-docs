# PR: 引入 AnalyzerContext 统一 Analyzer 签名 & 补充 Registry 隔离测试

## 背景信息

- **背景PR**：https://github.com/meta-pytorch/tritonparse/pull/409 （Registry 架构重构 — 消除循环依赖 & 引入实例级隔离）

在前置 PR 的 Review 中，针对 `procedure_checks` 是否应保留在统一 analyzer 签名中这一问题，最终采用了可扩展性更好的方案：将 `procedure_checks` 封装为 `AnalyzerContext` 对象，使得未来新增 per-call 上下文字段时无需逐个修改所有 analyzer 签名。本 PR 即为该方案的落地实现。

## 摘要

本 PR 解决 Analysis 分发系统中的两个问题：

1. **Analyzer 签名耦合**：当前三个 analyzer wrapper 统一签名 `(entry, procedure_checks) -> dict | None`，但 `procedure_checks` 只被 `_analyze_procedures_generic` 使用，另外两个 analyzer（`loop_schedules`、`amd_buffer_ops`）完全忽略它。未来新增上下文字段（如 `device_info`）时，每新增一个字段都需要修改所有 analyzer 签名。本 PR 引入 `AnalyzerContext` dataclass 封装 per-call 上下文，签名简化为 `(entry, ctx) -> dict | None`，后续扩展只需在 `AnalyzerContext` 中加字段。

2. **`register_backend_analyzer` 参数传递 bug 及测试覆盖缺失**：`register_backend_analyzer`、`register_backend_derived_artifact` 与 `register_backend_parser` 同属一类注册便利方法，设计初衷是为测试提供外部注册入口，避免直接访问 adapter 内部 `_xxx_registry` 属性。生产代码中子类均在 `__init__` 中直接操作内部 registry，因此这些方法无生产调用点。`register_backend_analyzer` 存在参数传递 bug（向 3 参数方法传了 4 个参数），因从未被调用而一直未触发。进一步排查发现 `register_backend_analyzer` 和 `register_backend_derived_artifact` 均缺少与 `register_backend_parser` 对等的隔离性测试。本 PR 修复该 bug，并补齐两项注册便利方法的隔离性测试。

---

## 核心改动

### 1. 新增 `AnalyzerContext` dataclass (`tritonparse/backend.py`)

在 `AnalyzerInfo` 之前新增 `AnalyzerContext`，与 `AnalyzerInfo`、`AnalysisRegistry` 并列，属于 adapter/registry 基础设施的一部分：

```python
@dataclass
class AnalyzerContext:
    """Per-call context passed to analyzers. Extensible without changing signatures."""

    procedure_checks: list[dict[str, Any]] | None = None
    # 未来可扩展: device_info, compile_options, etc.
```

**设计要点**：
- 所有字段带默认值，支持 `AnalyzerContext()` 构造空上下文
- `AnalyzerInfo.func` 类型注解收紧为 `Callable[[dict[str, Any], AnalyzerContext], dict[str, Any] | None]`，类型检查器可校验注册函数签名是否正确

### 2. Analyzer 签名统一变更

从 `(entry, procedure_checks=None)` 统一变为 `(entry, ctx: AnalyzerContext)`：

```python
# Before
def _analyze_procedures_generic(entry, procedure_checks=None) -> dict | None:
    if not procedure_checks:
        return None
    procedure_results = find_procedures_with_patterns(procedure_checks, ...)

# After
def _analyze_procedures_generic(entry, ctx: AnalyzerContext) -> dict | None:
    if not ctx.procedure_checks:
        return None
    procedure_results = find_procedures_with_patterns(ctx.procedure_checks, ...)
```

三个 analyzer wrapper（`_analyze_loop_schedules_generic`、`_analyze_procedures_generic`、`_analyze_amd_buffer_ops`）均同步更新。

### 3. Dispatch 层改造

`AnalyzerContext` 在分析子系统的入口处（`trace_processor.py`）构建：

```python
# trace_processor.py (调用方)
ctx = AnalyzerContext(procedure_checks=procedure_checks)
ir_analysis = _generate_ir_analysis(compilation_event, ctx)
```

`_generate_ir_analysis`、`_generate_ir_analysis_adapter_driven`、`_generate_ir_analysis_legacy`、`run_analysis_pass` 四层函数的 `procedure_checks` 参数全部替换为 `ctx: AnalyzerContext`，且为必选参数（无默认值），从调用链头到尾保证类型安全。

### 4. 修复 `register_backend_analyzer` 参数 bug

```python
# Before (bug: 向 3 参数方法传了 4 个参数)
self._analysis_registry.register(
    analyzer_id, analyzer_func, required_stages, self.adapter_name
)

# After
self._analysis_registry.register(
    analyzer_id, analyzer_func, required_stages
)
```

该函数是为测试用例预留的外部可调用的注册函数，在生产代码中无调用点，入参不匹配是实例化重构PR移除 `adapter_affinity` 字段后的遗留问题，本次PR修复该问题并补充了相关的测试用例。

### 5. 补充 Registry 隔离测试

新增两个测试用例，与 `register_backend_parser` 的测试模式对齐：

- **`test_register_backend_analyzer_isolation`**：在 NVIDIA adapter 注册自定义 analyzer，验证仅 NVIDIA adapter 可见，AMD adapter 不可见
- **`test_register_backend_derived_artifact_isolation`**：在 NVIDIA adapter 注册自定义 derived artifact，验证仅 NVIDIA adapter 可见

---

## 文件变更

| 文件 | 变更 |
|------|------|
| `tritonparse/backend.py` | 新增 `AnalyzerContext`；`AnalyzerInfo.func` 类型注解更新；`run_analysis_pass` 签名改为接收 `ctx`；修复 `register_backend_analyzer` 参数 bug |
| `tritonparse/parse/ir_analysis.py` | 导入 `AnalyzerContext`；三个 wrapper 签名改为 `(entry, ctx)`；dispatch 层传递 `ctx` 代替 `procedure_checks` |
| `tritonparse/parse/trace_processor.py` | 导入 `AnalyzerContext`；在调用 `_generate_ir_analysis` 处构建 `ctx` |
| `tests/cpu/test_multi_backend_stage.py` | 现有测试适配新签名；新增 `test_register_backend_analyzer_isolation` 和 `test_register_backend_derived_artifact_isolation` |

---

## 变更影响分析

### 外部 API 不变

以下调用链不受影响：

```
CLI --procedure-checks → unified_parse → parse_logs → parse_single_rank → _generate_ir_analysis
```

`parse_single_file(procedure_checks=...)` 接口保持不变，`AnalyzerContext` 的构建发生在 `trace_processor.py` 内部。


---

## 测试验证

- make format-check

- make test

- make test-cuda

---

## 总结

本 PR 引入 `AnalyzerContext` dataclass 统一 analyzer 的 per-call 上下文传递，将签名从 `(entry, procedure_checks)` 简化为 `(entry, ctx)`。未来扩展上下文字段只需修改 `AnalyzerContext` 和需要该字段的 analyzer，不波及其他 analyzer。同时修复了 `register_backend_analyzer` 的参数传递 bug，并补齐了 analyzer 和 derived artifact 注册功能的隔离性测试。
