# PR: Frontend Generalization — Eliminate Hardcoded Stage Names via `ir_stages`

## Context

- **Depends on**: https://github.com/meta-pytorch/tritonparse/pull/406 (backend trace enrichment)

## Summary

This PR generalizes the frontend to consume the `ir_stages` field carried by each compilation event in the trace, removing all hardcoded GPU stage names (`ttgir`, `ptx`, `sass`, etc.). Every piece of logic that depends on stage names — display names, syntax highlighting, default panel selection, source mapping cross-references, and line grouping anchors — now derives its values dynamically from `ir_stages`.

For older traces that do not contain `ir_stages`, all change points retain the original hardcoded logic as a fallback path, preserving identical behavior to before this PR.

---

## Key Changes

### 1. Data Layer: `IRStageDescriptor` Interface and `ir_stages` Parsing (`dataLoader.ts`)

New `IRStageDescriptor` interface, matching the backend definition:

```typescript
export interface IRStageDescriptor {
    name: string;
    extension: string;
    display_name: string;
    display_order: number;
    is_text: boolean;
    supports_source_mapping: boolean;
    syntax_id: string;
}
```

- `ProcessedKernel` gains `ir_stages?: IRStageDescriptor[]`
- `LogEntry.payload` type gains the `ir_stages` field
- `processKernelData()` reads `entry.payload.ir_stages` from compilation events
- `SourceMapping` interface generalized: 12 hardcoded cross-reference fields (`ttgir_lines`, `ptx_lines`, etc.) replaced with `[key: string]: unknown` index signature, supporting dynamic fields for any stage name

### 2. Utility Functions: `getDefaultPanels()` and `getGroupingAnchor()` (`dataLoader.ts`)

Both functions filter stages by `is_text && supports_source_mapping` and sort by `display_order`:

- **`getDefaultPanels(kernel)`**: returns the first 2 stages as default left/right panels. Falls back to `{ left: "ttgir", right: "ptx" }`
- **`getGroupingAnchor(kernel)`**: returns the middle stage for line grouping. Falls back to `"ttgir"`

Examples:

| Backend | Viewable Stages | Default Panels | Grouping Anchor |
|---------|----------------|----------------|-----------------|
| NVIDIA  | ttir, ttgir, llir, ptx, sass | ttir / ttgir | llir |
| AMD     | ttir, ttgir, llir, amdgcn | ttir / ttgir | llir |

### 3. Dynamic Display Names (`irLanguage.ts`)

`getDisplayLanguage(irType, irStages?)` gains an optional `irStages` parameter. It first looks up `display_name` from `ir_stages`; if not found, falls back to the original hardcoded if-else chain.

All call sites (`CodeView.tsx`, `CodeComparisonView.tsx`, `SingleCodeViewer.tsx`, `App.tsx`) now pass `kernel?.ir_stages`.

### 4. Dynamic Syntax Highlighting (`languageUtils.ts`)

`mapLanguageToHighlighter(language, irStages?)` gains an optional `irStages` parameter. It first looks up `syntax_id` from `ir_stages`; if not found, falls back to the original hardcoded logic.

### 5. Dynamic Source Mapping Cross-References (`CodeComparisonView.tsx`)

`irTypesToCheck` in `calculateMappedLines()` changed from 6 hardcoded GPU stages to dynamic generation:

```typescript
const irTypesToCheck = irStages && irStages.length > 0
    ? irStages
        .filter(s => s.supports_source_mapping)
        .map(s => ({ type: s.name, property: `${s.name}_lines` }))
    : [/* original GPU hardcoded list */];
```

For NPU traces, cross-reference fields like `ttadapter_lines` and `bcmlir_lines` are automatically discovered with no frontend changes.

### 6. Dynamic Default Panel Selection (`CodeView.tsx`)

`findDefaultIRFiles()` no longer hardcodes `ttgir`/`ptx`. Instead, it calls `getDefaultPanels(kernel)` to get stage names and matches them against available files.

### 7. Dynamic Fallback in File Diff (`FileDiffView.tsx`)

- `irType` initial value changed from hardcoded `"ttgir"` to empty string
- `effectiveIrType` final fallback changed from `"ttgir"` to `getDefaultPanels(leftKernel).left`
- Moved `leftKernel` resolution earlier to eliminate duplicate computation

### 8. Dynamic Line Grouping Anchor (`SingleCodeViewer.tsx`)

The line grouping anchor in `handleLineClick` changed from hardcoded `ttgir_line` to `getGroupingAnchor()`, accessing the corresponding field via `${anchorStage}_line`.

---

## Fallback Strategy

All change points follow the same pattern:

```
if (kernel.ir_stages && kernel.ir_stages.length > 0) {
    // New path: derive from ir_stages dynamically
} else {
    // Fallback: keep original hardcoded logic unchanged
}
```

Old traces without `ir_stages` take the fallback path and behave exactly as before.

---

## Files Changed

| File | Change |
|------|--------|
| `website/src/utils/dataLoader.ts` | Add `IRStageDescriptor` interface; add `ir_stages` to `ProcessedKernel`; generalize `SourceMapping` with index signature; add `getDefaultPanels()` and `getGroupingAnchor()` |
| `website/src/utils/irLanguage.ts` | Add optional `irStages` param to `getDisplayLanguage()`; dynamic lookup first |
| `website/src/utils/languageUtils.ts` | Add optional `irStages` param to `mapLanguageToHighlighter()`; dynamic lookup first |
| `website/src/components/CodeComparisonView.tsx` | Dynamic `irTypesToCheck`; add `irStages` prop; update `useMemo` deps |
| `website/src/pages/CodeView.tsx` | `findDefaultIRFiles()` uses `getDefaultPanels()`; pass `irStages` to all `getDisplayLanguage`/`mapLanguageToHighlighter` calls |
| `website/src/pages/FileDiffView.tsx` | `effectiveIrType` fallback uses `getDefaultPanels()`; move kernel resolution earlier |
| `website/src/components/SingleCodeViewer.tsx` | Grouping anchor from `ttgir_line` to `getGroupingAnchor()`; add `irStages` prop |
| `website/src/App.tsx` | Pass `irStages` to `mapLanguageToHighlighter` and `SingleCodeViewer` |

---

## Testing

Verified on legacy traces, NVIDIA GPU traces, and NPU multi-backend traces. All frontend features work as expected.

---

## Conclusion

This PR completes frontend consumption of `ir_stages`. New backends only need to implement an adapter with stage descriptors (`name`, `display_name`, `syntax_id`, `display_order`, etc.) and the frontend adapts automatically:

- Display names, syntax highlighting, panel selection, cross-references, and line grouping — all driven by stage descriptors
- New backends (e.g., NPU with `ttadapter`, `bcmlir`) are supported with zero frontend changes
- Old traces remain fully compatible via the fallback path
