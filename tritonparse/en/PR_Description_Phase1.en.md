# PR: Reader-Side Foundation - Backend Adapter and Generic Parse Flow Refactor

## Background

- **RFC document**: [Flexible_backend_support.md](./Flexible_backend_support.md)

## Summary

This PR is the **first PR in Flexible Backend Support RFC Phase 1**. Phase 1 is split into 3 PRs. See the final section for the next steps.

**What this PR does**: this PR focuses on the reader-side foundation and the generic parse flow refactor. It adds the backend adapter infrastructure and refactors the generic parse scheduling logic in `trace_processor.py`, including stage discovery, processing flow, and mapping construction.

---

## Main Changes

### 1. Backend Adapter Infrastructure (`tritonparse/backend.py` - new file)

**New core data structure**:

```python
@dataclass(frozen=True)
class IRStageDescriptor:
    name: str                      # Stage name, for example "ttir"
    extension: str                 # File extension, for example ".ttir"
    display_name: str              # Display name, for example "TTIR"
    display_order: int             # Display order (smaller number means earlier)
    is_text: bool                  # Whether the file is text
    supports_source_mapping: bool  # Whether source mapping is supported
    parser_id: str                 # Parser ID, for example "generic_loc"
    syntax_id: str                 # Syntax highlight ID, for example "mlir"
```

**New adapter abstraction**:

```python
class CompilationPipelineAdapter(ABC):
    @abstractmethod
    def adapter_name(self) -> str: ...

    @abstractmethod
    def runtime_backend(self) -> str: ...

    @abstractmethod
    def pytorch_module(self) -> str: ...

    @abstractmethod
    def get_ir_stages(self) -> list[IRStageDescriptor]: ...
```

**New concrete adapters**:

- `NvidiaTritonAdapter`: defines 7 stages for the CUDA backend (TTIR, TTGIR, LLIR, PTX, CUBIN, SASS, JSON)
- `AmdTritonAdapter`: defines 5 stages for the HIP backend (TTIR, TTGIR, LLIR, AMDGCN, JSON)

**New registry mechanism**:

```python
class PipelineAdapterRegistry:
    def register(self, adapter_cls: type[CompilationPipelineAdapter]) -> None: ...

    def resolve(self, adapter_name: str) -> CompilationPipelineAdapter: ...

    def resolve_from_trace(self, metadata: dict[str, Any]) -> CompilationPipelineAdapter: ...

    def create_all(self) -> list[CompilationPipelineAdapter]: ...
```

**New helper functions**:

- `get_present_stage_descriptors_from_event()`: gets stage descriptors from a trace event, with metadata-first and fallback behavior
- `get_backend_registry()`: gets the global registry instance

### 2. Generic Parse Scheduling Refactor (`tritonparse/parse/trace_processor.py`)

**New core function**:

#### 2.1 Dynamic Stage Discovery

```python
def _resolve_source_mappable_stage_keys(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Resolve stage keys that support source mapping from the trace.

    Two-level resolution:
    1. First: metadata.stage_descriptors (new trace format)
    2. Fallback: hardcoded extensions (old trace format)
    """
```

#### 2.2 Dynamic Stage Processing Flow

**Before this change (hardcoded 5 stages)**:

```python
# Hardcoded stage lookup
ttir_key = next((k for k in file_content if k.endswith(".ttir")), None)
ttgir_key = next((k for k in file_content if k.endswith(".ttgir")), None)
ptx_key = next((k for k in file_content if k.endswith(".ptx")), None)
amdgcn_key = next((k for k in file_content if k.endswith(".amdgcn")), None)
sass_key = next((k for k in file_content if k.endswith(".sass")), None)

# Hardcoded processing for each stage
ttir_map = process_ir(ttir_key, file_content, file_path)
ttgir_map = process_ir(ttgir_key, file_content, file_path)
ptx_map = process_ir(ptx_key, file_content, file_path, [ttir_map, ttgir_map])
amdgcn_map = process_ir(amdgcn_key, file_content, file_path, [ttir_map, ttgir_map])
sass_map = process_ir(sass_key, file_content, file_path, [ttir_map, ttgir_map])

# Hardcoded mapping construction
ir_maps = {
    "ttir": ttir_map,
    "ttgir": ttgir_map,
    "ptx": ptx_map,
    "amdgcn": amdgcn_map,
    "sass": sass_map,
}
```

**After this change (dynamic handling for any number of stages)**:

```python
# Discover stages dynamically
stage_keys = _resolve_source_mappable_stage_keys(entry)

# Process each stage in a dynamic loop
stage_maps = {}
for i, stage_name in enumerate(stage_names):
    artifact_key = stage_keys[stage_name]
    other_mappings = [stage_maps[prev] for prev in stage_names[:i]]
    stage_map = process_ir(artifact_key, file_content, file_path, other_mappings)
    stage_maps[stage_name] = stage_map

# Build bidirectional mappings dynamically (supports any number of stages)
for src_stage in stage_names:
    for tgt_stage in stage_names[i + 1:]:
        create_bidirectional_mapping(
            stage_maps[src_stage], stage_maps[tgt_stage],
            src_stage, tgt_stage
        )
```

**Key improvements**:
1. **Removed the hardcoded stage list**: no longer limited to 5 fixed stages
2. **Dynamic stage processing loop**: automatically handles any number of stages
3. **Dynamic mapping construction**: bidirectional mapping now adapts to the number of stages
4. **Dynamic Python mapping**: automatically includes all available stages

**Key improvements**:
1. **Supports any number of stages**: no longer limited to 5 hardcoded stages
2. **Automatically discovers new stages**: for example AMDGCN and future Ascend stages
3. **Keeps backward compatibility**: the fallback path still works for old traces
4. **Metadata-first**: new traces use the full metadata from the writer side

---

## Architecture Update

### Data flow before this change

```
Trace Event
    ↓
trace_processor.py (hardcoded 5 stages)
    ↓
process_ir() (called manually 5 times)
    ↓
source_mappings (fixed 5 keys)
```

### Data flow after this change

```
Trace Event
    ↓
get_backend_registry()
    ↓
resolve_from_trace(metadata)
    ↓
Adapter (NvidiaTritonAdapter / AmdTritonAdapter)
    ↓
_resolve_source_mappable_stage_keys() (with fallback logic)
    ↓
Dynamic stage loop processing
    ↓
source_mappings (dynamic keys)
```

---

## Validation

### Compatibility testing

To verify that the current parse path works for both new and old traces, I built the first version of the trace adaptation logic on my fork branch `logs_creator`. I used it to generate a new trace file for the new-trace validation case.
The validation results are good for both the new trace and the old trace. Both can be parsed correctly.

### Functional testing

The result of `make test-cuda` is shown below:


### Multi-backend testing


---

## Conclusion

This PR completes **PR 1 of the Flexible Backend Support RFC**. It builds the **reader-side foundation and generic parse flow refactor**, and it sets up the base for multi-backend support.

### Main contributions

1. ✅ **Adapter infrastructure**: complete contract, registry, and concrete implementations (`NvidiaTritonAdapter`, `AmdTritonAdapter`)
2. ✅ **Dynamic stage discovery**: metadata-first with hardcoded fallback
3. ✅ **Generic parse scheduling refactor**:
   - Dynamic stage discovery (`_resolve_source_mappable_stage_keys`)
   - Dynamic stage processing loop (supports any number of stages)
   - Dynamic bidirectional mapping construction (adapts automatically to the number of stages)
   - Dynamic Python mapping (automatically includes all available stages)
4. ✅ **Removed hardcoded logic**: `trace_processor.py` no longer hardcodes stage lookup and stage processing
5. ✅ **Backward compatibility**: 100% compatible with existing traces

---

## RFC Phase 1 Status and Next Steps

### ✅ PR 1 (this PR): Reader-side foundation and generic parse flow refactor
- Adapter infrastructure + generic parse scheduling refactor in `trace_processor.py`
- Add `tritonparse/backend.py`: `IRStageDescriptor`, `CompilationPipelineAdapter`, `NvidiaTritonAdapter`, `AmdTritonAdapter`, `PipelineAdapterRegistry`
- Refactor `trace_processor.py`: dynamic stage discovery (fallback included), dynamic stage processing loop, and dynamic mapping construction

### 🔜 PR 2: Parse and source mapping
- Backend-specific parser refactor in `ir_parser.py` (parser dispatch)

### 🔜 PR 3: Analysis and reproducer
- Migration of `ir_analysis.py` and the `reproducer/` modules