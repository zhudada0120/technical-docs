

#ifndef TRITON_NPU_HEADERS
#define TRITON_NPU_HEADERS
#include <assert.h>
#include <stdbool.h>
#include <string>
#include <memory>
#include <sys/syscall.h>
#include <vector>
#include <Python.h>
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <dlfcn.h>
#include <functional>
#endif



#define PY_SSIZE_T_CLEAN

#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2


extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}



typedef void* (*triton_allocate_workspace_legacy_t)(uint64_t);
typedef void* (*triton_allocate_sync_block_lock_t)(uint64_t, void*, void**);
typedef void  (*triton_async_launch_t)(void*, const char*);
typedef void  (*triton_release_retained_tensor_t)(void*);

static triton_allocate_workspace_legacy_t g_allocate_workspace_legacy = nullptr;
static triton_allocate_sync_block_lock_t g_allocate_sync_block_lock = nullptr;
static triton_async_launch_t g_async_launch = nullptr;
static triton_release_retained_tensor_t g_release_retained_tensor = nullptr;

static bool npu_utils_ready() {
    return g_allocate_workspace_legacy &&
           g_allocate_sync_block_lock &&
           g_async_launch &&
           g_release_retained_tensor;
}

static void init_npu_utils() {
    if (npu_utils_ready()) return;
    const char* so_path = "/home/zhudada/.triton/cache/4UWHOTOVGGJKVPI3VQ2Q5UCEZI/npu_utils.so";
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error: dlopen %s failed: %s\n", so_path, dlerror());
        return;
    }
    g_allocate_workspace_legacy = (triton_allocate_workspace_legacy_t)dlsym(handle, "triton_allocate_workspace_legacy");
    g_allocate_sync_block_lock = (triton_allocate_sync_block_lock_t)dlsym(handle, "triton_allocate_sync_block_lock");
    g_async_launch = (triton_async_launch_t)dlsym(handle, "triton_async_launch");
    g_release_retained_tensor = (triton_release_retained_tensor_t)dlsym(handle, "triton_release_retained_tensor");
}

static void release_npu_tensor_handle(void* handle) {
    if (!handle) return;
    if (!g_release_retained_tensor) {
        fprintf(stderr, "Error: triton_release_retained_tensor is unavailable\n");
        return;
    }
    g_release_retained_tensor(handle);
}



typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  // Cache the interned "data_ptr" key once instead of rebuilding a temporary
  // PyUnicode on every call. Function-local static init is thread-safe in C++11
  // and the GIL is held here, so the one-time init is safe.
  static PyObject *data_ptr_str = PyUnicode_InternFromString("data_ptr");
  PyObject *ptr = PyObject_GetAttr(obj, data_ptr_str);
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}


static inline size_t _align_launch_offset(size_t offset, size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

extern "C" {
void triton_launch_kernel(
    const char* kernelName, const void* func, rtStream_t stream,
    int gridX, int gridY, int gridZ,
    const int64_t* shapes_data, const int* shape_dims, int num_tensors,
    const int* tensor_kinds,
    const void* const* kernel_args, const size_t* arg_sizes, int num_args) {
  if (gridX <=0 || gridY <=0 || gridZ <=0) {
    printf("WARNING: Skipping launch for kernel '%s' due to empty grid (gridX=%d, gridY=%d, gridZ=%d).\n", kernelName, gridX, gridY, gridZ);
    return;
  }
  std::vector<std::vector<int64_t>> tensorShapes;
  if (shapes_data != nullptr && shape_dims != nullptr) {
    int shapes_idx = 0;
    for (int tensor_idx = 0; tensor_idx < num_tensors; ++tensor_idx) {
      std::vector<int64_t> tensorShape;
      for (int dim_idx = 0; dim_idx < shape_dims[tensor_idx]; ++dim_idx) {
        tensorShape.push_back(shapes_data[shapes_idx++]);
      }
      tensorShapes.push_back(tensorShape);
    }
  }
  std::vector<int> tensorKinds;
  if (tensor_kinds != nullptr && num_tensors > 0) {
    tensorKinds.assign(tensor_kinds, tensor_kinds + num_tensors);
  }
  if (num_args > 0 && (kernel_args == nullptr || arg_sizes == nullptr)) {
    return;
  }
  std::vector<size_t> launch_arg_sizes;
  launch_arg_sizes.reserve(num_args);
  std::vector<std::vector<char>> copied_kernel_args;
  copied_kernel_args.reserve(num_args);
  for (int arg_idx = 0; arg_idx < num_args; ++arg_idx) {
    launch_arg_sizes.push_back(arg_sizes[arg_idx]);
    copied_kernel_args.emplace_back(arg_sizes[arg_idx]);
    memcpy(copied_kernel_args.back().data(), kernel_args[arg_idx], arg_sizes[arg_idx]);
  }

  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  void *workspace_addr_ptr = NULL;
  
  uint32_t blockNum4Workspace = gridX * gridY * gridZ;
  
  
  std::function<rtError_t()> launch_call = [=]() -> rtError_t {
    
    uint32_t blockNum = gridX * gridY * gridZ;

    #ifdef ENABLE_GRID_WARN_PRINT
      static bool warned = false;
      if (!warned && blockNum > (uint32_t)40) {
        printf("WARNING: Grid %u > physical limit 40, performance maybe reduced.\n",blockNum);
        warned = true;
    }
    #endif
    blockNum = std::min(blockNum, (uint32_t)40);
    // set mixBlockNumRation for nodeBasicBlockDim for msprof report
    uint32_t mixBlockNumRation = 0;
    uint32_t nodeBasicBlockDim = (mixBlockNumRation << 16) + blockNum;

    
    rtError_t ret = RT_ERROR_NONE;
    void *ffts_addr = NULL; uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) return ret;
    // stub argument for workspace
    void *syncBlockLock_ptr = NULL;
    void *syncBlockLock_handle = NULL;
    uint16_t ModuleId = 0;
    
    

    size_t args_offset = 0;
    auto reserve_slot = [&](size_t size, size_t alignment) -> size_t {
      args_offset = _align_launch_offset(args_offset, alignment);
      size_t current_offset = args_offset;
      args_offset += size;
      return current_offset;
    };
    size_t ffts_offset = reserve_slot(sizeof(void*), 8);
    size_t sync_block_lock_offset = reserve_slot(sizeof(void*), 8);
    size_t workspace_offset = reserve_slot(sizeof(void*), 8);
    size_t kernel_args_offset = args_offset;
    for (int arg_idx = 0; arg_idx < num_args; ++arg_idx) {
      size_t alignment = launch_arg_sizes[arg_idx] >= 8 ? 8 : (launch_arg_sizes[arg_idx] >= 4 ? 4 : 1);
      args_offset = _align_launch_offset(args_offset, alignment);
      args_offset += launch_arg_sizes[arg_idx];
    }
    size_t grid_offset = reserve_slot(sizeof(int32_t), 4);
    reserve_slot(sizeof(int32_t), 4);
    reserve_slot(sizeof(int32_t), 4);
    
    size_t total_size = args_offset;

    std::vector<char> launch_args(total_size, 0);
    memcpy(launch_args.data() + ffts_offset, &ffts_addr, sizeof(void*));
    memcpy(launch_args.data() + sync_block_lock_offset, &syncBlockLock_ptr, sizeof(void*));
    memcpy(launch_args.data() + workspace_offset, &workspace_addr_ptr, sizeof(void*));
    size_t kernel_arg_offset = kernel_args_offset;
    for (int arg_idx = 0; arg_idx < num_args; ++arg_idx) {
      size_t alignment = launch_arg_sizes[arg_idx] >= 8 ? 8 : (launch_arg_sizes[arg_idx] >= 4 ? 4 : 1);
      kernel_arg_offset = _align_launch_offset(kernel_arg_offset, alignment);
      memcpy(launch_args.data() + kernel_arg_offset, copied_kernel_args[arg_idx].data(), launch_arg_sizes[arg_idx]);
      kernel_arg_offset += launch_arg_sizes[arg_idx];
    }
    memcpy(launch_args.data() + grid_offset, &gridX, sizeof(int32_t));
    memcpy(launch_args.data() + grid_offset + sizeof(int32_t), &gridY, sizeof(int32_t));
    memcpy(launch_args.data() + grid_offset + 2 * sizeof(int32_t), &gridZ, sizeof(int32_t));
    

    
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }

    
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(launch_args.data()), launch_args.size(), NULL, stream);

    
    
    
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }
    if (__MsprofFlagL1)
    {
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AIV;
      nodeBasicInfo.data.nodeBasicInfo.blockDim = nodeBasicBlockDim;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // 'mix' kernel need to report the ctxID
      if (false > 0) {
        MsprofAdditionalInfo info;
        info.level = MSPROF_REPORT_NODE_LEVEL;
        info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
        info.threadId = threadId;
        info.timeStamp = endTime;
        MsprofContextIdInfo ctxId;
        ctxId.opName = opNameHashID;
        ctxId.ctxIdNum = 1;
        for (uint32_t i = 0; i < ctxId.ctxIdNum; i++) {
          ctxId.ctxIds[i] = i;
        }
        size_t copyLen = sizeof(MsprofContextIdInfo);
        if (copyLen > MSPROF_ADDTIONAL_INFO_DATA_LENGTH) {
          copyLen = MSPROF_ADDTIONAL_INFO_DATA_LENGTH;
        }
        memcpy(info.data, &ctxId, copyLen);
        MsprofReportAdditionalInfo(false, static_cast<void *>(&info), sizeof(MsprofAdditionalInfo));
      }

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {
        dataTypes[0] = 3;
dataTypes[1] = 1;
dataTypes[2] = 3;
dataTypes[4] = 1;
      }
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {
        auto fillTensorData = [&](int index, int tensorType) {
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            profTensorData->tensorData[index].shape[j] = 0;
          }
        };
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }
      }
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }

    return ret;
   };
   init_npu_utils();
   if (!g_async_launch) {
     fprintf(stderr, "Error: triton_async_launch is unavailable\n");
     return;
   }
   g_async_launch(static_cast<void*>(&launch_call), name.c_str());
  return;
}
} // extern "C"

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds, void* arg0, void* arg1, void* arg2, int32_t arg3, void* arg4, int32_t arg5, int32_t arg6) {
  // Keep Python launcher on the stable local packing path.
  if (gridX <=0 || gridY <=0 || gridZ <=0) {
    printf("WARNING: Skipping launch for kernel '%s' due to empty grid (gridX=%d, gridY=%d, gridZ=%d).\n", kernelName, gridX, gridY, gridZ);
    return;
  }
  std::string name = "";
  name.append(kernelName);
  void *workspace_addr_ptr = NULL;
  
  uint32_t blockNum4Workspace = gridX * gridY * gridZ;
  
  
  std::function<rtError_t()> launch_call = [=]() -> rtError_t {
    
    uint32_t blockNum = gridX * gridY * gridZ;

    #ifdef ENABLE_GRID_WARN_PRINT
      static bool warned = false;
      if (!warned && blockNum > (uint32_t)40) {
        printf("WARNING: Grid %u > physical limit 40, performance maybe reduced.\n",blockNum);
        warned = true;
    }
    #endif
    blockNum = std::min(blockNum, (uint32_t)40);
    uint32_t mixBlockNumRation = 0;
    uint32_t nodeBasicBlockDim = (mixBlockNumRation << 16) + blockNum;

    
    rtError_t ret = RT_ERROR_NONE;
    void *ffts_addr = NULL; uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) return ret;
    void *syncBlockLock_ptr = NULL;
    void *syncBlockLock_handle = NULL;
    uint16_t ModuleId = 0;
    
    
    struct __attribute__((packed)) {
      void* ffts_addr __attribute__((aligned(8)));
      void* syncBlockLock __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
      void* arg0 __attribute__((aligned(8))); void* arg1 __attribute__((aligned(8))); void* arg2 __attribute__((aligned(8))); int32_t arg3 __attribute__((aligned(4))); void* arg4 __attribute__((aligned(8))); int32_t arg5 __attribute__((aligned(4))); int32_t arg6 __attribute__((aligned(4)));
      int32_t gridX __attribute__((aligned(4))); int32_t gridY __attribute__((aligned(4))); int32_t gridZ __attribute__((aligned(4)));
      
    } args = {
      static_cast<void*>(ffts_addr),
      nullptr,
      nullptr,
      static_cast<void*>(arg0), static_cast<void*>(arg1), static_cast<void*>(arg2), static_cast<int32_t>(arg3), static_cast<void*>(arg4), static_cast<int32_t>(arg5), static_cast<int32_t>(arg6),
      static_cast<int32_t>(gridX), static_cast<int32_t>(gridY), static_cast<int32_t>(gridZ)
      
    };
    
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }

    
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);

    
    
    
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }
    if (__MsprofFlagL1)
    {
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AIV;
      nodeBasicInfo.data.nodeBasicInfo.blockDim = nodeBasicBlockDim;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // 'mix' kernel need to report the ctxID
      if (false > 0) {
        MsprofAdditionalInfo info;
        info.level = MSPROF_REPORT_NODE_LEVEL;
        info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
        info.threadId = threadId;
        info.timeStamp = endTime;
        MsprofContextIdInfo ctxId;
        ctxId.opName = opNameHashID;
        ctxId.ctxIdNum = 1;
        for (uint32_t i = 0; i < ctxId.ctxIdNum; i++) {
          ctxId.ctxIds[i] = i;
        }
        size_t copyLen = sizeof(MsprofContextIdInfo);
        if (copyLen > MSPROF_ADDTIONAL_INFO_DATA_LENGTH) {
          copyLen = MSPROF_ADDTIONAL_INFO_DATA_LENGTH;
        }
        memcpy(info.data, &ctxId, copyLen);
        MsprofReportAdditionalInfo(false, static_cast<void *>(&info), sizeof(MsprofAdditionalInfo));
      }

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {
        dataTypes[0] = 3;
dataTypes[1] = 1;
dataTypes[2] = 3;
dataTypes[4] = 1;
      }
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {
        auto fillTensorData = [&](int index, int tensorType) {
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            profTensorData->tensorData[index].shape[j] = 0;
          }
        };
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }
      }
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }

    return ret;
   };
   init_npu_utils();
   if (!g_async_launch) {
     fprintf(stderr, "Error: triton_async_launch is unavailable\n");
     return;
   }
   g_async_launch(static_cast<void*>(&launch_call), name.c_str());
  return;
}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_shape(PyObject *tensor) {
  std::vector<int64_t> shape;

  // Early return if tensor is None or null
  if (!tensor || tensor == Py_None) {
    return shape;
  }

  // Calling tensor.size()
  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {
    return shape;
  }
  // Using PySequence_Fast to improve access efficiency
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {
        shape.push_back(PyLong_AsLong(dim));
      }
    }
  }
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;

  PyObject* _arg0;
  PyObject* _arg1;
  PyObject* _arg2;
  int32_t _arg3;
  PyObject* _arg4;
  int32_t _arg5;
  int32_t _arg6;
  PyObject* _arg7;
  PyObject* _arg8;
  if(!PyArg_ParseTuple(
      args, "iiiKKOOOOOOOiOiiOO",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata, &launch_enter_hook, &launch_exit_hook
      , &_arg0, &_arg1, &_arg2, &_arg3, &_arg4, &_arg5, &_arg6, &_arg7, &_arg8
      )
    ) {
    return NULL;
  }
  if (__MsprofFlagL1)
  {
    { auto tmp = _get_tensor_shape(_arg0); if (!tmp.empty()) tensorShapes.push_back(tmp); }
{ auto tmp = _get_tensor_shape(_arg1); if (!tmp.empty()) tensorShapes.push_back(tmp); }
{ auto tmp = _get_tensor_shape(_arg2); if (!tmp.empty()) tensorShapes.push_back(tmp); }
{ auto tmp = _get_tensor_shape(_arg4); if (!tmp.empty()) tensorShapes.push_back(tmp); }
  }

  if (launch_enter_hook != Py_None){
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }


  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }
  }


  // raise exception asap
  DevicePtrInfo ptr_info0 = getPointer(_arg0, 0); if (!ptr_info0.valid) return NULL;
  DevicePtrInfo ptr_info1 = getPointer(_arg1, 1); if (!ptr_info1.valid) return NULL;
  DevicePtrInfo ptr_info2 = getPointer(_arg2, 2); if (!ptr_info2.valid) return NULL;
  DevicePtrInfo ptr_info4 = getPointer(_arg4, 4); if (!ptr_info4.valid) return NULL;
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds, ptr_info0.dev_ptr, ptr_info1.dev_ptr, ptr_info2.dev_ptr, _arg3, ptr_info4.dev_ptr, _arg5, _arg6);
  if (PyErr_Occurred()) {
    return NULL;
  }
  if(launch_exit_hook != Py_None){
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }
  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__triton_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___triton_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  
  MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h

  return m;
}
