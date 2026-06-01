# PR: Reader-side 基础设施层 - Backend Adapter 与通用 Parse 逻辑改造

## 背景信息

- **RFC 文档**：[Flexible_backend_support.md](./Flexible_backend_support.md)

## 摘要

本 PR 是 **Flexible Backend Support RFC Phase 1 的第一个 PR**（Phase 1 总共拆分为 3 个 PR，下一步计划见文末）。

**本 PR 内容**：Reader-side 基础设施层与通用 Parse 逻辑改造。本 PR 建立了后端适配器（backend adapter）基础设施，并改造了 `trace_processor.py` 中的通用 parse 调度逻辑（stage 发现、处理流程、映射构建）。

---

## 核心改动

### 1. 后端适配器基础设施 (`tritonparse/backend.py` - 新文件)

**新增核心数据结构**：

```python
@dataclass(frozen=True)
class IRStageDescriptor:
    name: str                      # stage 名称，例如 "ttir"
    extension: str                 # 文件扩展名，例如 ".ttir"
    display_name: str              # 显示名称，例如 "TTIR"
    display_order: int             # 显示顺序（数字越小越靠前）
    is_text: bool                  # 是否为文本文件
    supports_source_mapping: bool  # 是否支持源映射
    parser_id: str                 # 解析器 ID，例如 "generic_loc"
    syntax_id: str                 # 语法高亮 ID，例如 "mlir"
```

**新增适配器抽象**：

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

**新增具体适配器**：

- `NvidiaTritonAdapter`: 定义 CUDA 后端的 7 个 stages（TTIR, TTGIR, LLIR, PTX, CUBIN, SASS, JSON）
- `AmdTritonAdapter`: 定义 HIP 后端的 5 个 stages（TTIR, TTGIR, LLIR, AMDGCN, JSON）

**新增 Registry 机制**：

```python
class PipelineAdapterRegistry:
    def register(self, adapter_cls: type[CompilationPipelineAdapter]) -> None: ...
    
    def resolve(self, adapter_name: str) -> CompilationPipelineAdapter: ...
    
    def resolve_from_trace(self, metadata: dict[str, Any]) -> CompilationPipelineAdapter: ...
    
    def create_all(self) -> list[CompilationPipelineAdapter]: ...
```

**新增辅助函数**：

- `get_present_stage_descriptors_from_event()`: 从 trace event 中提取 stage descriptors（支持元数据和降级）
- `get_backend_registry()`: 获取全局 registry 实例

### 2. 通用 Parse 调度逻辑改造 (`tritonparse/parse/trace_processor.py`)

**新增核心函数**：

#### 2.1 动态 Stage 发现

```python
def _resolve_source_mappable_stage_keys(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    解析 trace 中支持 source mapping 的 stage keys。

    两级解析机制：
    1. 优先：metadata.stage_descriptors（新 trace 格式）
    2. 降级：硬编码扩展名（旧 trace 格式）
    """
```

#### 2.2 动态 Stage 处理流程

**改造前（硬编码 5 个 stages）**：

```python
# 硬编码查找 stages
ttir_key = next((k for k in file_content if k.endswith(".ttir")), None)
ttgir_key = next((k for k in file_content if k.endswith(".ttgir")), None)
ptx_key = next((k for k in file_content if k.endswith(".ptx")), None)
amdgcn_key = next((k for k in file_content if k.endswith(".amdgcn")), None)
sass_key = next((k for k in file_content if k.endswith(".sass")), None)

# 硬编码处理每个 stage
ttir_map = process_ir(ttir_key, file_content, file_path)
ttgir_map = process_ir(ttgir_key, file_content, file_path)
ptx_map = process_ir(ptx_key, file_content, file_path, [ttir_map, ttgir_map])
amdgcn_map = process_ir(amdgcn_key, file_content, file_path, [ttir_map, ttgir_map])
sass_map = process_ir(sass_key, file_content, file_path, [ttir_map, ttgir_map])

# 硬编码构建映射
ir_maps = {
    "ttir": ttir_map,
    "ttgir": ttgir_map,
    "ptx": ptx_map,
    "amdgcn": amdgcn_map,
    "sass": sass_map,
}
```

**改造后（动态处理任意 stages）**：

```python
# 动态发现 stages
stage_keys = _resolve_source_mappable_stage_keys(entry)

# 动态循环处理每个 stage
stage_maps = {}
for i, stage_name in enumerate(stage_names):
    artifact_key = stage_keys[stage_name]
    other_mappings = [stage_maps[prev] for prev in stage_names[:i]]
    stage_map = process_ir(artifact_key, file_content, file_path, other_mappings)
    stage_maps[stage_name] = stage_map

# 动态构建双向映射（支持任意 stage 数量）
for src_stage in stage_names:
    for tgt_stage in stage_names[i + 1:]:
        create_bidirectional_mapping(
            stage_maps[src_stage], stage_maps[tgt_stage],
            src_stage, tgt_stage
        )
```

**关键改进**：
1. **消除了硬编码的 stage 列表**：不再局限于 5 个固定 stages
2. **动态 stage 处理循环**：自动适配任意数量的 stages
3. **动态映射构建**：双向映射自动适配 stage 数量
4. **动态 Python 映射**：自动包含所有可用的 stages

**关键改进**：
1. **支持任意数量的 stages**：不再局限于 5 个硬编码 stages
2. **自动发现新 stages**：例如 AMDGCN、未来的 Ascend stages
3. **保持向后兼容**：降级策略确保旧 traces 仍然工作
4. **元数据优先**：新 traces 使用 writer-side 提供的完整元数据

---

## 架构改进

### 改造前的数据流

```
Trace Event
    ↓
trace_processor.py (硬编码 5 个 stages)
    ↓
process_ir() (手动调用 5 次)
    ↓
source_mappings (固定的 5 个 keys)
```

### 改造后的数据流

```
Trace Event
    ↓
get_backend_registry()
    ↓
resolve_from_trace(metadata)
    ↓
Adapter (NvidiaTritonAdapter / AmdTritonAdapter)
    ↓
_resolve_source_mappable_stage_keys() (包含降级策略)
    ↓
动态 stage 循环处理
    ↓
source_mappings (动态 keys)
```

---

## 测试验证

### 兼容性测试
为了验证当前parse能力在新旧trace场景下的兼容情况，在我fork仓库的logs_creator分支中完成了初版trace适配逻辑，生成了新版本的trace文件用于验证新trace场景下的适配情况。
新旧trace验证结果如下，均可以正确完成parse。

### 功能测试
make test-cuda结果如下：


### 多后端测试


---

## 总结

本 PR 完成了 **Flexible Backend Support RFC 的 PR 1**，建立了 **Reader-side 基础设施层与通用 Parse 逻辑改造**，为多后端支持打下了基础。

### 核心贡献

1. ✅ **Adapter 基础设施**：完整的 contract、registry、具体实现（NvidiaTritonAdapter、AmdTritonAdapter）
2. ✅ **动态 Stage 发现**：元数据优先 + 硬编码降级
3. ✅ **通用 Parse 调度逻辑改造**：
   - 动态 stage 发现（`_resolve_source_mappable_stage_keys`）
   - 动态 stage 处理循环（支持任意数量 stages）
   - 动态双向映射构建（自动适配 stage 数量）
   - 动态 Python 映射（自动包含所有可用 stages）
4. ✅ **消除硬编码**：`trace_processor.py` 不再硬编码 stage 查找和处理
5. ✅ **向后兼容**：100% 兼容现有 traces

---

## RFC Phase 1 完成情况 & 下一步计划

### ✅ PR 1（本 PR）：Reader-side 基础设施层与通用 Parse 逻辑改造
- Adapter 基础设施 + `trace_processor.py` 的通用 parse 调度逻辑改造
- 新增 `tritonparse/backend.py`：IRStageDescriptor、CompilationPipelineAdapter、NvidiaTritonAdapter、AmdTritonAdapter、PipelineAdapterRegistry
- 改造 `trace_processor.py`：动态 stage 发现（降级策略）、动态 stage 处理循环、动态映射构建

### 🔜 PR 2：Parse 与 source mapping
- `ir_parser.py` 的后端专属解析器改造（parser dispatch）

### 🔜 PR 3：Analysis 与 reproducer
- `ir_analysis.py` 和 `reproducer/` 模块迁移