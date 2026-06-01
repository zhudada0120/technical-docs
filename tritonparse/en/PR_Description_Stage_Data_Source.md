# PR: Unify Stage Discovery to Adapter-Driven — Remove backend dependency on `stage_descriptors`

## Context

- **RFC**: https://github.com/meta-pytorch/tritonparse/issues/367
- **Depends on**: https://github.com/meta-pytorch/tritonparse/pull/387 (Reader-side infrastructure & general parse logic)

## Summary

`_resolve_source_mappable_stage_keys()` previously read `metadata.stage_descriptors` from the trace to discover stages. This PR changes it to resolve stages directly from the adapter (via `backend_name`), so the backend no longer depends on the `stage_descriptors` metadata field. Three unused functions and their associated tests are removed.

**Why**: The adapter already defines all stage information via `get_ir_stages()`. Reading the same data from trace metadata is redundant for the backend. After this change, `stage_descriptors` is only consumed by the frontend (Phase 2).

---

## Key Changes

### 1. Stage discovery: adapter-driven instead of metadata-driven (`trace_processor.py`)

**Before**: Read `metadata.stage_descriptors` → deserialize → match extensions against `file_content`.

**After**: Read `metadata.backend_name` → resolve adapter → `adapter.get_ir_stages()` → match extensions against `file_content`.

This aligns with how parser dispatch (PR 2) and analysis dispatch (PR 3) already work — all three now use `backend_name` as the sole dispatch condition.

### 2. Remove redundant `backend_name` pre-checks in parser and stage resolution (`trace_processor.py`)

This follow-up change removes an extra layer of conditional logic that checked whether `backend_name` was a string before attempting adapter-driven resolution.

**Before**:

- `generate_source_mappings()` only entered the adapter-driven parser path when `metadata is not None and isinstance(metadata.get("backend_name"), str)`
- `_resolve_source_mappable_stage_keys()` only entered adapter-driven stage resolution when `isinstance(backend_name, str)`

**After**:

- both functions now directly attempt `resolve_from_trace(metadata)` whenever metadata is available
- if adapter resolution fails, they fall back to the original hardcoded path via the existing `ValueError` handling
- the related docstrings and inline comments were updated to describe the actual control flow more accurately: try adapter-driven resolution first, then fall back when resolution fails

**Why this matters**:

Once `backend_name` was confirmed to be consistently present in normal traces, these pre-checks no longer carried meaningful dispatch information. The real branch condition is whether adapter resolution succeeds, not whether a separate local `isinstance(..., str)` gate passes before calling the resolver.

### 3. Removed functions (`backend.py`)

- `deserialize_stage_descriptors_from_event()` — deserialized `stage_descriptors` from trace metadata
- `_deserialize_stage_descriptor()` — constructed `IRStageDescriptor` from dict
- `_validate_stage_dict()` — validated required fields in raw stage dict

No backend code calls these functions anymore.

### 4. Removed tests (`test_multi_backend_stage.py`)

- `test_deserialize_and_ordering` — tested deserialization and `display_order` sorting
- `test_missing_required_field_raises` — tested validation of missing fields

---

## Impact

- **Backend**: All three dispatch sites (stage discovery, parser, analysis) now use `backend_name` uniformly. No backend code reads `stage_descriptors` from trace metadata.
- **Frontend**: `stage_descriptors` in trace metadata remains available for frontend consumption (Phase 2).
- **Compatibility**: Hardcoded extension fallback is used in two scenarios, with no behavior change:
  - Trace has no `backend_name` (very early traces)
  - `backend_name` maps to an unregistered adapter (e.g. a backend not yet implemented)


## Testing

Format checking
The result of make format-check is shown below:
image


Functional testing
The result of make test-cuda is shown below:
image


The result of make test is shown below:
image
