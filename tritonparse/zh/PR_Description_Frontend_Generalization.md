# PR: 前端通用化改造 — 利用 `ir_stages` 消除 Stage 硬编码

## 背景信息

- **前置 PR**：https://github.com/meta-pytorch/tritonparse/pull/406 （后端trace改造）

## 摘要

本 PR 完成前端的通用化改造，利用 trace 中每个 compilation 事件携带的 `ir_stages` 字段，消除前端对 GPU stage 名称（`ttgir`、`ptx`、`sass` 等）的全部硬编码。所有涉及 stage 名称的逻辑（显示名称、语法高亮、默认面板选择、source mapping 交叉引用、行分组锚点）均改为从 `ir_stages` 动态获取。

对于不含 `ir_stages` 的旧 trace 文件，所有改动点保留原有硬编码逻辑作为降级路径，行为与改造前完全一致。

---

## 核心改动

### 1. 数据层：`IRStageDescriptor` 接口与 `ir_stages` 解析（`dataLoader.ts`）

新增 `IRStageDescriptor` 接口，与后端定义一一对应：

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

- `ProcessedKernel` 新增 `ir_stages?: IRStageDescriptor[]` 字段
- `LogEntry.payload` 类型新增 `ir_stages` 字段
- `processKernelData()` 从 compilation 事件中读取 `entry.payload.ir_stages`
- `SourceMapping` 接口泛化：12 个硬编码交叉引用字段（`ttgir_lines`、`ptx_lines` 等）替换为 `[key: string]: unknown` 索引签名，支持任意 stage 名的动态字段

### 2. 工具函数：`getDefaultPanels()` 和 `getGroupingAnchor()`（`dataLoader.ts`）

从 `ir_stages` 中筛选 `is_text && supports_source_mapping` 的 stage，按 `display_order` 排序后：

- **`getDefaultPanels(kernel)`**：取前 2 个作为默认左右面板。降级返回 `{ left: "ttgir", right: "ptx" }`
- **`getGroupingAnchor(kernel)`**：取中间位置 stage 作为行分组锚点。降级返回 `"ttgir"`

示例：

| 后端 | Viewable Stages | 默认面板 | 分组锚点 |
|------|----------------|---------|---------|
| NVIDIA | ttir, ttgir, llir, ptx, sass | ttir / ttgir | llir |
| AMD | ttir, ttgir, llir, amdgcn | ttir / ttgir | llir |

### 3. 显示名称动态化（`irLanguage.ts`）

`getDisplayLanguage(irType, irStages?)` 新增可选 `irStages` 参数。优先从 `ir_stages` 查找 `display_name`，未命中时走原有 if-else 硬编码降级链。

所有调用点（`CodeView.tsx`、`CodeComparisonView.tsx`、`SingleCodeViewer.tsx`、`App.tsx`）均已传入 `kernel?.ir_stages`。

### 4. 语法高亮动态化（`languageUtils.ts`）

`mapLanguageToHighlighter(language, irStages?)` 新增可选 `irStages` 参数。优先从 `ir_stages` 查找 `syntax_id`，未命中时走原有硬编码降级链。

### 5. Source Mapping 交叉引用动态化（`CodeComparisonView.tsx`）

`calculateMappedLines()` 中的 `irTypesToCheck` 从硬编码 6 条 GPU stage 改为动态生成：

```typescript
const irTypesToCheck = irStages && irStages.length > 0
    ? irStages
        .filter(s => s.supports_source_mapping)
        .map(s => ({ type: s.name, property: `${s.name}_lines` }))
    : [/* 原有 GPU 硬编码列表 */];
```

NPU 场景下自动识别 `ttadapter_lines`、`bcmlir_lines` 等交叉引用字段，无需前端改动。

### 6. 默认面板选择动态化（`CodeView.tsx`）

`findDefaultIRFiles()` 从硬编码 `ttgir`/`ptx` 改为调用 `getDefaultPanels(kernel)` 获取 stage 名，再在文件列表中匹配。

### 7. File Diff 降级动态化（`FileDiffView.tsx`）

- `irType` 初始值从硬编码 `"ttgir"` 改为空字符串
- `effectiveIrType` 最终降级从 `"ttgir"` 改为 `getDefaultPanels(leftKernel).left`
- 提前 `leftKernel` 解析，消除重复计算

### 8. 行分组锚点动态化（`SingleCodeViewer.tsx`）

`handleLineClick` 中行分组锚点从硬编码 `ttgir_line` 改为 `getGroupingAnchor()` 动态获取，通过 `${anchorStage}_line` 访问对应字段。

---

## 降级策略

所有改动点遵循统一模式：

```
if (kernel.ir_stages && kernel.ir_stages.length > 0) {
    // 新路径：从 ir_stages 动态获取
} else {
    // 降级：保留原有硬编码逻辑，原样不动
}
```

旧 trace 文件不含 `ir_stages` 字段，前端走降级路径，行为与改造前完全一致。

---

## 改动文件总览

| 文件 | 改动 |
|------|------|
| `website/src/utils/dataLoader.ts` | 新增 `IRStageDescriptor` 接口；`ProcessedKernel` 增加 `ir_stages` 字段；`SourceMapping` 泛化为动态索引签名；新增 `getDefaultPanels()`、`getGroupingAnchor()` 工具函数 |
| `website/src/utils/irLanguage.ts` | `getDisplayLanguage()` 新增可选 `irStages` 参数，优先动态查找 |
| `website/src/utils/languageUtils.ts` | `mapLanguageToHighlighter()` 新增可选 `irStages` 参数，优先动态查找 |
| `website/src/components/CodeComparisonView.tsx` | `irTypesToCheck` 动态生成；新增 `irStages` prop 透传；`useMemo` 依赖更新 |
| `website/src/pages/CodeView.tsx` | `findDefaultIRFiles()` 使用 `getDefaultPanels()`；所有 `getDisplayLanguage`/`mapLanguageToHighlighter` 调用传入 `irStages` |
| `website/src/pages/FileDiffView.tsx` | `effectiveIrType` 降级使用 `getDefaultPanels()`；提前 kernel 解析消除重复计算 |
| `website/src/components/SingleCodeViewer.tsx` | 行分组锚点从 `ttgir_line` 改为 `getGroupingAnchor()`；新增 `irStages` prop |
| `website/src/App.tsx` | `mapLanguageToHighlighter` 和 `SingleCodeViewer` 传入 `irStages` |

---

## 测试

测试了旧trace  Nvidia GPU和NPU多后端场景下的效果，前端功能均正常

---


## 总结

本 PR 完成前端对 `ir_stages` 的完整消费。新增后端只需实现 adapter 并定义 stage 描述（`name`、`display_name`、`syntax_id`、`display_order` 等），前端即可自动适配：

- 显示名称、语法高亮、面板选择、交叉引用、行分组——全部从 stage 描述动态驱动
- 零前端改动即可支持新后端（如 NPU 的 `ttadapter`、`bcmlir`）
- 旧 trace 完全兼容，降级行为与改造前一致
