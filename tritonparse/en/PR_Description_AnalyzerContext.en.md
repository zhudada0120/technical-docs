# PR: Introduce AnalyzerContext to Unify Analyzer Signatures & Add Registry Isolation Tests

## Context

- **Preceding PR**: https://github.com/meta-pytorch/tritonparse/pull/409 (Registry Architecture Refactor — Eliminate Circular Dependencies & Introduce Instance-Level Isolation)

During the review of the preceding PR, the question arose whether `procedure_checks` should remain in the unified analyzer signature. The conclusion was to adopt a more extensible approach: encapsulate `procedure_checks` into an `AnalyzerContext` object, so that future per-call context fields can be added without modifying every analyzer signature. This PR implements that decision.

## Summary

This PR addresses two issues in the analysis dispatch system:

1. **Analyzer signature coupling**: The three analyzer wrappers share the signature `(entry, procedure_checks) -> dict | None`, yet `procedure_checks` is only used by `_analyze_procedures_generic`. The other two analyzers (`loop_schedules`, `amd_buffer_ops`) ignore it entirely. Adding future context fields (e.g., `device_info`) would require changing every analyzer signature for each new field. This PR introduces the `AnalyzerContext` dataclass to encapsulate per-call context, simplifying the signature to `(entry, ctx) -> dict | None`. Future extensions only need to add fields to `AnalyzerContext`.

2. **`register_backend_analyzer` argument bug and missing test coverage**: `register_backend_analyzer`, `register_backend_derived_artifact`, and `register_backend_parser` are all registration convenience methods designed to provide an external registration entry point for tests, avoiding direct access to adapter-internal `_xxx_registry` attributes. In production code, subclasses operate on internal registries directly in `__init__`, so these methods have no production callers. `register_backend_analyzer` had an argument bug (passing 4 arguments to a 3-parameter method) that was never triggered because the function was never called. Further investigation revealed that both `register_backend_analyzer` and `register_backend_derived_artifact` lacked isolation tests on par with `register_backend_parser`. This PR fixes the bug and adds isolation tests for both registration convenience methods.

---

## Key Changes

### 1. New `AnalyzerContext` dataclass (`tritonparse/backend.py`)

Added `AnalyzerContext` before `AnalyzerInfo`, alongside `AnalyzerInfo` and `AnalysisRegistry`, as part of the adapter/registry infrastructure:

```python
@dataclass
class AnalyzerContext:
    """Per-call context passed to analyzers. Extensible without changing signatures."""

    procedure_checks: list[dict[str, Any]] | None = None
    # Future: device_info, compile_options, etc.
```

**Design points**:
- All fields have default values, supporting `AnalyzerContext()` to construct an empty context
- `AnalyzerInfo.func` type annotation tightened to `Callable[[dict[str, Any], AnalyzerContext], dict[str, Any] | None]`, enabling type checkers to validate registered function signatures

### 2. Unified Analyzer Signature Change

From `(entry, procedure_checks=None)` to `(entry, ctx: AnalyzerContext)`:

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

All three analyzer wrappers (`_analyze_loop_schedules_generic`, `_analyze_procedures_generic`, `_analyze_amd_buffer_ops`) updated accordingly.

### 3. Dispatch Layer Refactor

`AnalyzerContext` is constructed at the entry point of the analysis subsystem (`trace_processor.py`):

```python
# trace_processor.py (caller)
ctx = AnalyzerContext(procedure_checks=procedure_checks)
ir_analysis = _generate_ir_analysis(compilation_event, ctx)
```

The `procedure_checks` parameter across all four dispatch functions — `_generate_ir_analysis`, `_generate_ir_analysis_adapter_driven`, `_generate_ir_analysis_legacy`, and `run_analysis_pass` — is replaced with `ctx: AnalyzerContext` as a required parameter (no default value), ensuring type safety from end to end of the call chain.

### 4. Fix `register_backend_analyzer` Argument Bug

```python
# Before (bug: passing 4 arguments to a 3-parameter method)
self._analysis_registry.register(
    analyzer_id, analyzer_func, required_stages, self.adapter_name
)

# After
self._analysis_registry.register(
    analyzer_id, analyzer_func, required_stages
)
```

This function is a registration convenience method reserved for external test use. It has no callers in production code. The argument mismatch is a leftover from the instance-level isolation refactor PR, which removed the `adapter_affinity` field. This PR fixes the issue and adds corresponding test coverage.

### 5. Registry Isolation Tests

Added two test cases, aligned with the `register_backend_parser` test pattern:

- **`test_register_backend_analyzer_isolation`**: Registers a custom analyzer on the NVIDIA adapter, verifies it is visible only on the NVIDIA adapter and not on the AMD adapter
- **`test_register_backend_derived_artifact_isolation`**: Registers a custom derived artifact on the NVIDIA adapter, verifies it is visible only on the NVIDIA adapter and not on the AMD adapter

---

## Files Changed

| File | Change |
|------|--------|
| `tritonparse/backend.py` | Added `AnalyzerContext`; updated `AnalyzerInfo.func` type annotation; changed `run_analysis_pass` signature to accept `ctx`; fixed `register_backend_analyzer` argument bug |
| `tritonparse/parse/ir_analysis.py` | Imported `AnalyzerContext`; changed three wrapper signatures to `(entry, ctx)`; dispatch layer passes `ctx` instead of `procedure_checks` |
| `tritonparse/parse/trace_processor.py` | Imported `AnalyzerContext`; constructs `ctx` at the `_generate_ir_analysis` call site |
| `tests/cpu/test_multi_backend_stage.py` | Adapted existing tests to new signature; added `test_register_backend_analyzer_isolation` and `test_register_backend_derived_artifact_isolation` |

---

## Impact Analysis

### External API Unchanged

The following call chain is unaffected:

```
CLI --procedure-checks → unified_parse → parse_logs → parse_single_rank → _generate_ir_analysis
```

The `parse_single_file(procedure_checks=...)` interface remains unchanged. `AnalyzerContext` construction happens internally in `trace_processor.py`.

---

## Testing

- make format-check

- make test

- make test-cuda

---

## Conclusion

This PR introduces the `AnalyzerContext` dataclass to unify per-call context passing for analyzers, simplifying the signature from `(entry, procedure_checks)` to `(entry, ctx)`. Future context field extensions only require modifying `AnalyzerContext` and the specific analyzers that need the new field, without affecting other analyzers. Additionally, it fixes the `register_backend_analyzer` argument bug and adds isolation tests for both analyzer and derived artifact registration convenience methods.
