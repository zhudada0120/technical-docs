# 前端通用化改造方案（不含 IR Analysis）

## 背景

Trace 已增强：每个 compilation 事件的 payload 中新增了 `ir_stages` 字段，包含该后端所有 stage 的定义（名称、显示名、语法高亮、排序等）。本方案描述前端如何利用 `ir_stages` 消除硬编码，并在旧 trace（不含 `ir_stages`）上降级到原有逻辑。

---

## 1. 硬编码现状

### 1.1 Stage 名称 → 显示名称（`irLanguage.ts`）

整个 `getDisplayLanguage()` 函数是一张硬编码映射表，共 9 个 if-else 分支：

| 行号 | 硬编码 | 返回值 |
|---|---|---|
| 11 | `"ttgir"` | `"TTGIR (TritonGPU MLIR)"` |
| 13 | `"ttir"` | `"TTIR (Triton MLIR)"` |
| 15 | `"llir"` | `"LLIR (LLVM IR)"` |
| 17 | `"ptx"` | `"PTX (NVIDIA Parallel Thread Execution)"` |
| 19 | `"cubin"` | `"CUBIN (NVIDIA CUDA Binary)"` |
| 21 | `"python"` | `"Python"` |
| 23 | `"json"` | `"JSON"` |
| 25 | `"amdgcn"` | `"AMDGCN (AMD GCN Assembly)"` |
| 27 | `"sass"` | `"SASS (NVIDIA Shader Assembly)"` |

### 1.2 Stage 名称 → 语法高亮（`languageUtils.ts`）

整个 `mapLanguageToHighlighter()` 函数，6 个分支：

| 行号 | 硬编码 | 返回值 |
|---|---|---|
| 10 | `"ttgir"`, `"ttir"` | `'mlir'` |
| 12 | `"llir"` | `'llvm'` |
| 14 | `"ptx"` | `'ptx'` |
| 16 | `"amdgcn"` | `'amdgcn'` |
| 18-19 | `"sass"` | `'asm'` |
| 20 | `"python"` | `'python'` |

### 1.3 Source Mapping 交叉引用表（`CodeComparisonView.tsx`）

`calculateMappedLines()` 中的 `irTypesToCheck`（行 248-255），6 个硬编码条目：

```typescript
const irTypesToCheck = [
    { type: "ttgir", property: "ttgir_lines" },
    { type: "ttir", property: "ttir_lines" },
    { type: "ptx", property: "ptx_lines" },
    { type: "llir", property: "llir_lines" },
    { type: "amdgcn", property: "amdgcn_lines" },
    { type: "sass", property: "sass_lines" },
];
```

### 1.4 默认面板选择

| 文件 | 行号 | 硬编码 |
|---|---|---|
| `CodeView.tsx` | 23 | 左面板：`key.includes("ttgir")` |
| `CodeView.tsx` | 30 | 右面板：`key.includes("ptx")` |
| `CodeComparisonView.tsx` | 195, 206 | 默认标题 `"TTGIR"` / `"PTX"` |
| `FileDiffSession.tsx` | 62 | 默认 `irType: 'ttgir'` |
| `FileDiffView.tsx` | 95, 234 | 兜底值 `"ttgir"` |

### 1.5 文件内行分组锚点（`SingleCodeViewer.tsx`）

行 54、59 硬编码使用 `ttgir_line` 做同文件行分组。

### 1.6 SourceMapping TypeScript 接口（`dataLoader.ts`）

行 12-23，12 个硬编码字段（`ttgir_line`、`ptx_lines` 等）。接口本身需要泛化，但这不需要 trace 改动。

---

## 2. 改造方案

### 2.0 数据层：在 `dataLoader.ts` 中引入 `ir_stages`

**新增接口：**

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

**`ProcessedKernel` 接口新增字段：**

```typescript
ir_stages?: IRStageDescriptor[];
```

**在 `processKernelData()` 解析 compilation 事件时读取 `ir_stages`：**

```typescript
// 处理 compilation 事件时
kernel.ir_stages = entry.payload.ir_stages;
```

**提供降级工具函数：**

```typescript
export function getIRStages(kernel: ProcessedKernel): IRStageDescriptor[] | null {
    return kernel.ir_stages || null;
}

/**
 * 获取按优先级排序的可展示、可映射的 stage 列表。
 * 规则：is_text=true AND supports_source_mapping=true，按 display_order 升序。
 * NVIDIA: [ttir(10), ttgir(20), llir(30), ptx(40), sass(60)]
 * AMD:    [ttir(10), ttgir(20), llir(30), amdgcn(40)]
 */
function getViewableStages(kernel: ProcessedKernel): IRStageDescriptor[] {
    const stages = kernel.ir_stages;
    if (!stages || stages.length === 0) return [];
    return stages
        .filter(s => s.is_text && s.supports_source_mapping)
        .sort((a, b) => a.display_order - b.display_order);
}

/**
 * 默认面板选择：取优先级最高的前 2 个 viewable stage。
 * NVIDIA: left=ttir, right=ttgir
 * AMD:    left=ttir, right=ttgir
 */
export function getDefaultPanels(kernel: ProcessedKernel): { left: string; right: string } {
    const viewable = getViewableStages(kernel);
    if (viewable.length >= 2) {
        return { left: viewable[0].name, right: viewable[1].name };
    }
    if (viewable.length === 1) {
        return { left: viewable[0].name, right: viewable[0].name };
    }
    // 降级：旧逻辑
    return { left: "ttgir", right: "ptx" };
}

/**
 * 分组锚点：取 viewable stages 中优先级处于中间的 stage。
 * 规则：index = Math.floor(length / 2)
 * NVIDIA: [ttir, ttgir, llir, ptx, sass] → 5个 → index 2 → llir
 * AMD:    [ttir, ttgir, llir, amdgcn]    → 4个 → index 2 → llir
 */
export function getGroupingAnchor(kernel: ProcessedKernel): string {
    const viewable = getViewableStages(kernel);
    if (viewable.length > 0) {
        return viewable[Math.floor(viewable.length / 2)].name;
    }
    return "ttgir";
}
```

### 2.1 改造 `irLanguage.ts`（显示名称）

**改造后：**

```typescript
export function getDisplayLanguage(irTypeOrFilename: string, irStages?: IRStageDescriptor[]): string {
    // 优先从 ir_stages 查找
    if (irStages && irStages.length > 0) {
        const type = irTypeOrFilename.split('.').pop()?.toLowerCase() || irTypeOrFilename;
        const stage = irStages.find(s => s.name === type);
        if (stage) return stage.display_name;
    }
    // 降级：保留原有硬编码逻辑
    const lower = irTypeOrFilename.toLowerCase();
    if (lower.endsWith("ttgir")) return "TTGIR (TritonGPU MLIR)";
    // ... 其余原有分支不变
}
```

**改动要点：** 函数签名新增可选参数 `irStages`，所有调用点传入 `kernel.ir_stages`。降级时走原有 if-else 链。

### 2.2 改造 `languageUtils.ts`（语法高亮）

**改造后：**

```typescript
export function mapLanguageToHighlighter(lang: string, irStages?: IRStageDescriptor[]): string {
    // 优先从 ir_stages 查找
    if (irStages && irStages.length > 0) {
        const stage = irStages.find(s => s.name === lang);
        if (stage) return stage.syntax_id;
    }
    // 降级：保留原有硬编码逻辑
    if (lang === "ttgir" || lang === "ttir") return 'mlir';
    // ... 其余原有分支不变
}
```

**改动要点：** 同上，新增可选参数，降级时走原有逻辑。`"python"` 不是 IR stage，仍走硬编码分支。

### 2.3 改造 `CodeComparisonView.tsx`（交叉引用表）

**`irTypesToCheck` 改为动态生成：**

```typescript
// 组件内，接收 kernel 作为 prop 或从 context 获取
const irTypesToCheck = useMemo(() => {
    const stages = kernel?.ir_stages;
    if (stages && stages.length > 0) {
        return stages
            .filter(s => s.supports_source_mapping)
            .map(s => ({ type: s.name, property: `${s.name}_lines` }));
    }
    // 降级：原有硬编码表
    return [
        { type: "ttgir", property: "ttgir_lines" },
        { type: "ttir", property: "ttir_lines" },
        { type: "ptx", property: "ptx_lines" },
        { type: "llir", property: "llir_lines" },
        { type: "amdgcn", property: "amdgcn_lines" },
        { type: "sass", property: "sass_lines" },
    ];
}, [kernel?.ir_stages]);
```

**默认面板标题改造：**

```typescript
const defaultPanels = getDefaultPanels(kernel);

// 行 195
title: leftPanel.title || defaultPanels.left,
// 行 206
title: rightPanel.title || defaultPanels.right,
```

### 2.4 改造 `CodeView.tsx`（默认面板选择）

**`findDefaultIRFiles()` 改造：**

```typescript
function findDefaultIRFiles(irFiles: string[], kernel?: ProcessedKernel): { left: string; right: string } {
    let left = irFiles[0] || "";
    let right = irFiles[1] || "";

    const defaultPanels = kernel ? getDefaultPanels(kernel) : null;
    if (defaultPanels) {
        // 新路径：用 getDefaultPanels() 返回的 stage name 匹配文件
        const leftFile = irFiles.find(key =>
            key.toLowerCase().includes(defaultPanels.left.toLowerCase())
        );
        const rightFile = irFiles.find(key =>
            key.toLowerCase().includes(defaultPanels.right.toLowerCase())
        );
        if (leftFile) left = leftFile;
        if (rightFile) right = rightFile;
    } else {
        // 降级：原有逻辑
        const ttgirFile = irFiles.find(key => key.toLowerCase().includes("ttgir"));
        if (ttgirFile) left = ttgirFile;
        const ptxFile = irFiles.find(key => key.toLowerCase().includes("ptx"));
        if (ptxFile) right = ptxFile;
    }

    return { left, right };
}
```

### 2.5 改造 `FileDiffSession.tsx` 和 `FileDiffView.tsx`

**`FileDiffSession.tsx` 行 62：**

```typescript
const defaultPanels = kernel ? getDefaultPanels(kernel) : { left: "ttgir", right: "ptx" };
const defaultOptions: DiffOptionsState = {
    irType: defaultPanels.left,  // 原来是 'ttgir'
    ...
};
```

**`FileDiffView.tsx` 行 95：**

```typescript
initialParams.get(PARAM_IR) || getDefaultPanels(kernel).left  // 原来是 "ttgir"
```

**`FileDiffView.tsx` 行 234：**

```typescript
irType || (unionIrTypes.length > 0 ? unionIrTypes[0] : getDefaultPanels(kernel).left)  // 原来是 "ttgir"
```

### 2.6 改造 `SingleCodeViewer.tsx`（行分组锚点）

```typescript
const anchorStage = getGroupingAnchor(kernel);

const handleLineClick = (lineNumber: number) => {
    setHighlightedLines([lineNumber]);
    if (sourceMapping) {
        const lineKey = lineNumber.toString();
        const clickedMapping = sourceMapping[lineKey];
        const anchorProperty = `${anchorStage}_line`;

        if (clickedMapping && clickedMapping[anchorProperty]) {
            const relatedLines = Object.entries(sourceMapping)
                .filter(
                    ([key, mapping]) =>
                        mapping[anchorProperty] === clickedMapping[anchorProperty] &&
                        parseInt(lineKey, 10) !== parseInt(key, 10)
                )
                .map(([line]) => parseInt(line, 10));
            if (relatedLines.length > 0) {
                setHighlightedLines([lineNumber, ...relatedLines]);
            }
        }
    }
};
```

### 2.7 泛化 `SourceMapping` 接口（`dataLoader.ts`）

将 12 个硬编码字段改为动态索引签名：

```typescript
export interface SourceMapping {
    line: number;
    file?: string;
    column?: number;
    type?: string;
    loc_id?: string;
    alias_name?: string;
    alias_of?: string;
    // 动态字段：{stage_name}_line（number，自引用）和 {stage_name}_lines（number[]，交叉引用）
    [key: string]: unknown;
}
```

所有通过 `.` 访问硬编码字段的地方（如 `mapping.ttgir_lines`）改为动态访问（`mapping[`${stageName}_lines`]`）。降级时原有的 `mapping.ttgir_lines` 仍然可用（因为 `[key: string]: unknown` 兼容固定字段）。

---

## 3. 降级策略总结

所有改造点遵循同一模式：

```
if (kernel.ir_stages && kernel.ir_stages.length > 0) {
    // 新路径：从 ir_stages 动态获取
} else {
    // 降级：保留原有硬编码逻辑，原样不动
}
```

降级场景：旧 trace 文件不包含 `ir_stages` 字段，前端走原有逻辑，行为与改造前完全一致。

---

## 4. 改动文件清单

| 文件 | 改动内容 |
|---|---|
| `dataLoader.ts` | 新增 `IRStageDescriptor` 接口，`ProcessedKernel` 增加 `ir_stages` 字段，解析 compilation 事件时读取，新增 `getDefaultPanels()` 和 `getGroupingAnchor()` 工具函数，`SourceMapping` 接口泛化 |
| `irLanguage.ts` | `getDisplayLanguage()` 新增可选 `irStages` 参数 |
| `languageUtils.ts` | `mapLanguageToHighlighter()` 新增可选 `irStages` 参数 |
| `CodeComparisonView.tsx` | `irTypesToCheck` 动态生成，默认面板标题动态获取 |
| `CodeView.tsx` | `findDefaultIRFiles()` 从 `ir_stages` 推导 |
| `FileDiffSession.tsx` | 默认 `irType` 从 `getDefaultPanels()` 获取 |
| `FileDiffView.tsx` | 两处兜底 `"ttgir"` 改为 `getDefaultPanels()` |
| `SingleCodeViewer.tsx` | `ttgir_line` 改为 `getGroupingAnchor()` 动态获取 |

**注意：** 所有调用 `getDisplayLanguage()` 和 `mapLanguageToHighlighter()` 的地方都需要传入 `irStages` 参数。需要逐一排查调用点。
