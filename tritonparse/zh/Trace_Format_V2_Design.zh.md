# Trace增强 — 为前端通用化改造提供数据基础

## 背景

当前前端存在大量与后端 IR stage 相关的硬编码（stage 名称、显示名称、语法高亮、默认面板等）。为了实现前端通用化（即前端不依赖任何特定后端，所有后端信息从 trace 中动态获取），需要先让 trace 携带足够的信息。

**注意：** 本文前端通用化改造分析不含 IR Analysis 功能，该功能的通用化改造会在后续PR中完成。

---

## 1. 前端通用化需要哪些数据


前端在 多个位置硬编码了 stage 相关信息：

| 前端文件 | 硬编码内容 |
|---|---|
| `irLanguage.ts` | stage → 显示名称映射：`"ttgir"` → `"TTGIR (TritonGPU MLIR)"` |
| `languageUtils.ts` | stage → 语法高亮映射：`"ttgir"` → `"mlir"` |
| `CodeComparisonView.tsx` | source mapping 可用的 stage 列表：`[{type:"ttgir"}, {type:"ptx"}, ...]` |
| `CodeView.tsx` | 默认面板选择：左 `"ttgir"`，右 `"ptx"` |
| `CodeComparisonView.tsx` | 默认面板标题：`"TTGIR"` / `"PTX"` |
| `FileDiffSession.tsx` | 默认 irType：`"ttgir"` |
| `FileDiffView.tsx` | `"ttgir"` 兜底值 |
| `SingleCodeViewer.tsx` | 文件内行分组锚点：`ttgir_line` |

前端通用化（不含 IR Analysis）只需要 trace 提供一项额外数据：**每个 compilation 事件的 stage 定义列表。** 具体包含哪些字段见第 2 节的字段说明。

---

## 2. Trace 新增字段：`compilation.payload.ir_stages`

### 添加位置

在 `compilation` 事件的 `payload` 字典中，与现有的 `file_content`、`source_mappings` 等并列。

### 为什么放在每个 compilation 事件中

同一个 trace 中不同的 compilation 可能使用不同的后端（例如同一进程中同时存在 CUDA 和 AMD 内核）。每个 compilation 事件携带自己的 stage 描述符。

### 格式

NVIDIA 后端示例：

```jsonc
{
  "event_type": "compilation",
  "payload": {
    "metadata": { "...现有字段..." },
    "file_content": { "...现有字段..." },
    "source_mappings": { "...现有字段..." },

    // ========== 新增 ==========
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



### 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | Stage 标识符。与 `source_mappings` 的 key 匹配，用于构造交叉引用字段名（`{name}_lines`、`{name}_line`） |
| `extension` | string | `file_content` key 中使用的文件扩展名（例如 `"add_kernel.ttir"` 包含此扩展名） |
| `display_name` | string | 面向用户的显示名称，用于 UI 标题、面板标题、标签页 |
| `display_order` | int | UI 显示排序。数值越小越靠前。约定：10, 20, 30... 留有间隔以便未来插入 |
| `is_text` | bool | 该 stage 的内容是文本（可查看）还是二进制（不可查看） |
| `supports_source_mapping` | bool | 该 stage 是否支持源码到源码的行映射 |
| `syntax_id` | string | Web 代码编辑器的语法高亮语言标识符。 |

注意：`parser_id` 被有意排除 — 它是后端内部概念（用于选择 IR 解析器），与前端无关。

---

## 3. 后端实现

在 `trace_processor.py` 的 `parse_single_trace_content()` 中，解析 adapter 之后，将 `adapter._stages` 序列化到 compilation 事件的 payload 中：

```python
# 在 adapter 解析之后（现有代码）
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

数据来源是 `IRStageDescriptor` 数据类，每个 adapter 在 `__init__` 中定义自己的 `_stages` 列表。