# Trace Enhancement — Data Foundation for Frontend Generalization

## Background

The current frontend contains extensive hardcoded logic tied to backend IR stages (stage names, display names, syntax highlighting, default panels, etc.). To achieve frontend generalization — where the frontend does not depend on any specific backend and all backend information is dynamically read from the trace — the trace must first carry sufficient data.

**Note:** This document does not cover the generalization of the IR Analysis feature. That will be addressed in a follow-up PR.

---

## 1. What Data the Frontend Needs

The frontend hardcodes stage-related information in multiple locations:

| Frontend File | Hardcoded Content |
|---|---|
| `irLanguage.ts` | Stage → display name mapping: `"ttgir"` → `"TTGIR (TritonGPU MLIR)"` |
| `languageUtils.ts` | Stage → syntax highlighting mapping: `"ttgir"` → `"mlir"` |
| `CodeComparisonView.tsx` | Source-mappable stage list: `[{type:"ttgir"}, {type:"ptx"}, ...]` |
| `CodeView.tsx` | Default panel selection: left `"ttgir"`, right `"ptx"` |
| `CodeComparisonView.tsx` | Default panel titles: `"TTGIR"` / `"PTX"` |
| `FileDiffSession.tsx` | Default irType: `"ttgir"` |
| `FileDiffView.tsx` | `"ttgir"` fallback value |
| `SingleCodeViewer.tsx` | Intra-file line grouping anchor: `ttgir_line` |

Frontend generalization (excluding IR Analysis) requires only one additional piece of data in the trace: **a list of stage definitions per compilation event.** See Section 2 for the field specification.

---

## 2. New Trace Field: `compilation.payload.ir_stages`

### Location

In the `payload` dictionary of each `compilation` event, alongside existing fields such as `file_content` and `source_mappings`.

### Why per-compilation

Different compilations within the same trace may use different backends (e.g., CUDA and AMD kernels in the same process). Each compilation event carries its own stage descriptor.

### Format

NVIDIA backend example:

```jsonc
{
  "event_type": "compilation",
  "payload": {
    "metadata": { "...existing fields..." },
    "file_content": { "...existing fields..." },
    "source_mappings": { "...existing fields..." },

    // ========== NEW ==========
    "ir_stages": [
      {
        "name": "ttir",
        "extension": ".ttir",
        "display_name": "TTIR",
        "display_order": 10,
        "is_text": true,
        "supports_source_mapping": true,
        "syntax_id": "mlir"
      },
      {
        "name": "ttgir",
        "extension": ".ttgir",
        "display_name": "TTGIR",
        "display_order": 20,
        "is_text": true,
        "supports_source_mapping": true,
        "syntax_id": "mlir"
      },
      {
        "name": "llir",
        "extension": ".llir",
        "display_name": "LLIR",
        "display_order": 30,
        "is_text": true,
        "supports_source_mapping": true,
        "syntax_id": "llvm"
      },
      {
        "name": "ptx",
        "extension": ".ptx",
        "display_name": "PTX",
        "display_order": 40,
        "is_text": true,
        "supports_source_mapping": true,
        "syntax_id": "ptx"
      },
      {
        "name": "cubin",
        "extension": ".cubin",
        "display_name": "CUBIN",
        "display_order": 50,
        "is_text": false,
        "supports_source_mapping": false,
        "syntax_id": "plaintext"
      },
      {
        "name": "sass",
        "extension": ".sass",
        "display_name": "SASS",
        "display_order": 60,
        "is_text": true,
        "supports_source_mapping": true,
        "syntax_id": "asm"
      },
      {
        "name": "json",
        "extension": ".json",
        "display_name": "JSON",
        "display_order": 100,
        "is_text": true,
        "supports_source_mapping": false,
        "syntax_id": "json"
      }
    ]
  }
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `name` | string | Stage identifier. Matches keys in `source_mappings` and is used to construct cross-reference field names (`{name}_lines`, `{name}_line`) |
| `extension` | string | File extension used in `file_content` keys (e.g., `"add_kernel.ttir"` contains this extension) |
| `display_name` | string | Human-readable name for UI headers, panel titles, and tabs |
| `display_order` | int | UI display order. Lower values appear first. Convention: 10, 20, 30... with gaps for future insertion |
| `is_text` | bool | Whether the stage content is text (viewable) or binary (not viewable) |
| `supports_source_mapping` | bool | Whether this stage supports source-to-source line mapping |
| `syntax_id` | string | Syntax highlighting language identifier for the web code editor. Values: `"mlir"`, `"llvm"`, `"ptx"`, `"asm"`, `"amdgcn"`, `"json"`, `"plaintext"` |

Note: `parser_id` is intentionally excluded — it is a backend-internal concern (used to select the IR parser) and irrelevant to the frontend.

---

## 3. Backend Implementation

In `trace_processor.py`, inside `parse_single_trace_content()`, after adapter resolution, serialize `adapter._stages` into the compilation event payload:

```python
# After adapter resolution (existing code)
payload["ir_stages"] = [
    {
        "name": stage.name,
        "extension": stage.extension,
        "display_name": stage.display_name,
        "display_order": stage.display_order,
        "is_text": stage.is_text,
        "supports_source_mapping": stage.supports_source_mapping,
        "syntax_id": stage.syntax_id,
    }
    for stage in adapter._stages
]
```

The data source is the `IRStageDescriptor` dataclass. Each adapter defines its own `_stages` list in `__init__`.
