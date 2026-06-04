# LinAlg Matrix MLP — Getting Started Guide

A step-by-step guide for using DirectX 12 [LinAlg Matrix][linalg-spec] to run MLP inference and training with the MiniDXNN library (`include/minidxnn/hlsl/mlp.hlsl`).

---

## Table of Contents

1. [Prerequisites & Installation](#prerequisites--installation)
2. [Creating a D3D12 Context with Experimental Features](#creating-a-d3d12-context-with-experimental-features)
3. [Feature Support Check](#feature-support-check)
4. [Preparing Weight Matrices for LinAlg Matrix](#preparing-weight-matrices-for-linalg-matrix)
5. [Preparing Bias Vectors](#preparing-bias-vectors)
6. [Compiling Compute Shaders with SM 6.10](#compiling-compute-shaders-with-sm-610)
7. [Using mlp.hlsl for Inference](#using-mlphlsl-for-inference)
8. [Using mlp.hlsl for Training](#using-mlphlsl-for-training)
9. [Source Code Reference Map](#source-code-reference-map)

---

## Prerequisites & Installation

### 1. GPU Driver

Install a driver that supports Shader Model 6.10 and LinAlg Matrix:

- **AMD**: [Radeon™ Software for RX 9000 Series][linalg-driver] or later
- **NVIDIA**: Check NVIDIA's developer portal for SM 6.10 support

### 2. Windows Developer Mode

LinAlg Matrix requires experimental shader models, which requires Windows Developer Mode.

1. Open **Settings → Update & Security → For developers**
2. Enable **Developer Mode**

See [Microsoft's guide][win-dev-mode] for details.

### 3. Agility SDK

MiniDXNN uses [Agility SDK 1.721-preview][dx12-agility-sdk-download] to access the latest D3D12 features. The SDK is auto-downloaded by CMake when building this project (placed in `third_party/gfx_dep/gfx/third_party/`).

If integrating manually, download the NuGet package and place the D3D12 runtime DLLs (`D3D12Core.dll`, `d3d12SDKLayers.dll`) in a `D3D12/` subdirectory next to your executable.

### 4. DirectX Shader Compiler (DXC)

Download [DXC v1.10.2605.4][dx12-dxc-download] or later. This version supports SM 6.10 and the `dx/linalg.h` system header.

Compile with:
```bash
dxc -I ./include/hlsl -T cs_6_10 -enable-16bit-types my_shader.hlsl
```

> **Note**: The `-I` path must include the directory containing `dx/linalg.h`, which ships with DXC 1.10+.

### 5. Build MiniDXNN

```bash
git clone --recursive https://github.com/amdadvtech/MiniDXNN.git
cd MiniDXNN
cmake -B build
cmake --build build --config Release
```

---

## Creating a D3D12 Context with Experimental Features

As of early 2026, LinAlg Matrix requires enabling experimental shader models **before** device creation.

### Raw D3D12 API

```cpp
#include <d3d12.h>

// Must be called BEFORE ID3D12Device creation
HRESULT enableExperimental()
{
    UUID features[] = { D3D12ExperimentalShaderModels };
    return D3D12EnableExperimentalFeatures(
        _countof(features), features, nullptr, nullptr);
}

// Then create the device normally
ComPtr<ID3D12Device> device;
D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
```

> ⚠️ **Windows Developer Mode must be enabled**, or `D3D12EnableExperimentalFeatures` will fail.

### Using the gfx Library (as in MiniDXNN)

The [gfx](https://github.com/gboisse/gfx) library wraps this with a single flag:

```cpp
#include "gfx.h"

GfxContext context = gfxCreateContext(
    window, kGfxCreateContextFlag_EnableExperimentalShaders);
```

Internally, gfx calls `D3D12EnableExperimentalFeatures` with `D3D12ExperimentalShaderModels` and checks for Developer Mode.

**Source reference**:
- [`third_party/gfx_dep/gfx/gfx.cpp` lines 961–968](../third_party/gfx_dep/gfx/gfx.cpp) — experimental features initialization
- [`example/common/gfx_utility.cpp`](../example/common/gfx_utility.cpp) — `createGfxContext()` usage

---

## Feature Support Check

Before using LinAlg Matrix, verify the device supports it.

### Tier Check (Recommended)

```cpp
// Using gfx wrapper
uint32_t tier = gfxGetLinearAlgebraTier(context);
if (tier == 0) {
    // D3D12_LINEAR_ALGEBRA_TIER_NOT_SUPPORTED
    printf("LinAlg Matrix not supported on this device.\n");
    return;
}
printf("LinAlg tier: %s\n", gfxGetLinearAlgebraTierName(context).c_str());
```

### Raw D3D12 Tier Check

```cpp
D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT linAlgSupport = {};
HRESULT hr = device->CheckFeatureSupport(
    D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT,
    &linAlgSupport, sizeof(linAlgSupport));

if (SUCCEEDED(hr) &&
    linAlgSupport.LinearAlgebraTier >= D3D12_LINEAR_ALGEBRA_TIER_1) {
    // Tier 1 supported — FP16 vector-matrix multiply guaranteed
}
```

### Granular Operation Support Check

For specific data type combinations (e.g., FP16 vector × FP16 matrix → FP16 result with FP16 bias):

```cpp
// Using gfx wrapper
GfxMatrixMultiplySupportResult result = gfxCheckMatrixMultiplyAddSupport(
    context,
    /* vectorInputType  */ 7,   // D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16
    /* matrixInputType  */ 7,   // D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16
    /* biasInputType    */ 7,   // D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16
    /* resultType       */ 7);  // D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16

if (result.supported && result.hardwareAccelerated) {
    printf("FP16 MatVecMulAdd: hardware accelerated!\n");
}
```

**Source reference**:
- [`third_party/gfx_dep/gfx/gfx.h`](../third_party/gfx_dep/gfx/gfx.h) — feature query API (`gfxGetLinearAlgebraTier`, `gfxCheckMatrixMultiplyAddSupport`, etc.)
- [D3D12 LinAlg Runtime Spec][linalg-overview] — full query structures

---

## Preparing Weight Matrices for LinAlg Matrix

LinAlg Matrix requires weight matrices in specific memory layouts with alignment constraints. The optimal approach uses `GetLinearAlgebraMatrixConversionDestinationInfo` and `ConvertLinearAlgebraMatrix` to convert CPU-side row-major matrices to the driver's optimal format.

### Alignment Requirements

| Requirement | Value |
|-------------|-------|
| Matrix base address | 128-byte aligned |
| Row/column stride | 16-byte aligned |
| Allocation size | Multiple of 16 bytes |

### Step 1: Pack CPU Data as Row-Major with Stride

```cpp
#include "common/d3d12_format.hpp"  // MiniDXNN utilities

using half_float::half;

// Prepare matrix info for each MLP layer
std::vector<ex::D3D12MatrixInfo<half>> matrixInfoList;
for (const auto& layer : mlpLayers) {
    ex::D3D12MatrixInfo<half> info;
    info.m_srcData   = layer.weightData();       // dense row-major weights
    info.m_rowSize   = layer.outputDimension();  // M (rows)
    info.m_columnSize = layer.inputDimension();  // K (columns)
    info.m_layout    = ex::MatrixLayout::MUL_OPTIMAL;  // target layout
    matrixInfoList.push_back(info);
}
```

### Step 2: GPU Conversion to Optimal Layout

The `packAsD3D12MatrixBuffer` function performs the full pipeline:
1. Packs source data with proper stride into a row-major GPU buffer
2. Queries destination size via `GetLinearAlgebraMatrixConversionDestinationInfo`
3. Performs GPU conversion via `ConvertLinearAlgebraMatrix`

```cpp
// Create GPU buffer with optimal matrix layout
// If conversion fails (e.g., no hardware support), falls back to ROW_MAJOR
std::shared_ptr<GfxBuffer> weightBuffer =
    ex::packAsD3D12MatrixBuffer<half>(context, matrixInfoList, /*allowFallback=*/true);

// After this call, matrixInfoList[i].m_layout reflects the actual layout used
// and matrixInfoList[i].m_dataSize reflects the per-matrix buffer size in bytes
```

### Direct D3D12 API (Without gfx)

```cpp
// 1. Query destination size
D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO destInfo = {};
destInfo.DestLayout  = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL;
destInfo.DestStride  = 0;  // driver default for optimal layouts
destInfo.NumRows     = numRows;
destInfo.NumColumns  = numColumns;
destInfo.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;

device->GetLinearAlgebraMatrixConversionDestinationInfo(&destInfo);
// destInfo.DestSize now contains the required buffer size

// 2. Create destination buffer (128-byte aligned)
// 3. Record conversion command
D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO convInfo = {};
convInfo.DestInfo = destInfo;
convInfo.SrcInfo.SrcSize     = srcSizeBytes;
convInfo.SrcInfo.SrcDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
convInfo.SrcInfo.SrcLayout   = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
convInfo.SrcInfo.SrcStride   = numColumns * sizeof(half);  // row stride
convInfo.DataDesc.DestVA     = destGpuVA;
convInfo.DataDesc.SrcVA      = srcGpuVA;

commandList->ConvertLinearAlgebraMatrix(&convInfo, 1);
```

> **Important**: Source buffer must be in `D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE` state, destination in `D3D12_RESOURCE_STATE_UNORDERED_ACCESS`.

**Source reference**:
- [`example/common/d3d12_format.hpp`](../example/common/d3d12_format.hpp) — alignment constants, `D3D12MatrixInfo`, `getD3D12MatrixInfo()`, `packAsD3D12Matrix()`
- [`example/common/gfx_utility.hpp`](../example/common/gfx_utility.hpp) — `packAsD3D12MatrixBuffer()` with GPU conversion
- [D3D12 LinAlg Runtime Spec — Convert Matrix][linalg-overview-convert] — full API specification

---

## Preparing Bias Vectors

Bias vectors also require alignment for LinAlg Matrix `VectorRef` in HLSL.

### Alignment Requirements

| Requirement | Value |
|-------------|-------|
| Bias vector base address | 128-byte aligned |

### Packing Bias Vectors

```cpp
std::vector<ex::D3D12VectorInfo<half>> vectorInfoList;
for (const auto& layer : mlpLayers) {
    ex::D3D12VectorInfo<half> info;
    info.m_srcData = layer.biasData();
    // info.m_alignment defaults to VECTOR_ALIGNMENT (128 bytes)
    vectorInfoList.push_back(info);
}

// Pack all bias vectors contiguously with alignment padding
std::shared_ptr<GfxBuffer> biasBuffer =
    ex::packAsD3D12VectorBuffer<half>(context, vectorInfoList);
```

The `packAsD3D12Vector` function:
1. Calls `getD3D12VectorInfo()` to compute aligned sizes
2. Copies each vector's data into an aligned buffer region
3. Zero-pads between vectors to satisfy alignment

**Source reference**:
- [`example/common/d3d12_format.hpp`](../example/common/d3d12_format.hpp) — `D3D12VectorInfo` struct, `getD3D12VectorInfo()`, and `packAsD3D12Vector()`
- [`example/common/gfx_utility.hpp`](../example/common/gfx_utility.hpp) — `packAsD3D12VectorBuffer()`

---

## Compiling Compute Shaders with SM 6.10

### Command-Line Compilation (DXC)

```bash
dxc -T cs_6_10 \
    -enable-16bit-types \
    -I ./include/hlsl \
    -D MINIDXNN_NUM_LAYERS=3 \
    -D MINIDXNN_HIDDEN_LAYER_DIMENSIONS=64 \
    -D MINIDXNN_WEIGHT_MATRIX_LAYOUT=2 \
    -E inferenceF16Kernel \
    my_shader.hlsl
```

Key flags:
- `-T cs_6_10` — target Shader Model 6.10 (required for `dx/linalg.h`)
- `-enable-16bit-types` — enable native `half` type
- `-I ./include/hlsl` — path to `dx/linalg.h` headers

### Runtime Compilation (gfx)

The gfx library compiles shaders at runtime using the specified shader model:

```cpp
// gfx sets shader model "6_10" for the program
const std::string_view shaderMode = "6_10";
GfxProgram program = gfxCreateProgram(
    context, "my_shader", "./shaders/", shaderMode.data(),
    includePaths.data(), includePaths.size());

// Create a compute kernel with compile-time definitions
std::vector<const char*> defs = {
    "MINIDXNN_NUM_LAYERS=3",
    "MINIDXNN_HIDDEN_LAYER_DIMENSIONS=64",
    "MINIDXNN_WEIGHT_MATRIX_LAYOUT=2",  // MUL_OPTIMAL
    "MINIDXNN_WEIGHT_MATRIX_ALIGNMENT=128",
    "MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT=16",
    "MINIDXNN_BIAS_VECTOR_ALIGNMENT=128",
    "MINIDXNN_HAS_BIAS=1",
    "MINIDXNN_NUM_THREADS_X=32",
    "MINIDXNN_NUM_TASKS=1024",
};
GfxKernel kernel = gfxCreateComputeKernel(
    context, program, "inferenceF16Kernel", defs.data(), defs.size());
```

**Source reference**:
- [`example/common/gfx_utility.cpp`](../example/common/gfx_utility.cpp) — `createGfxProgram()` with SM 6.10
- [`example/01_texture_inference/example.cpp`](../example/01_texture_inference/example.cpp) — `buildKernelDefinitions()`

---

## Using mlp.hlsl for Inference

### HLSL Shader Code

```hlsl
// my_inference.hlsl
#include <minidxnn/hlsl/mlp.hlsl>

// Architecture (set via compile definitions or hardcoded)
static const uint NUM_LAYERS = MINIDXNN_NUM_LAYERS;
static const int  HIDDEN_DIM = MINIDXNN_HIDDEN_LAYER_DIMENSIONS;

// Choose activation functions
using ActivationHidden = mininn::LeakyReluActivation;
using ActivationOutput = mininn::SigmoidActivation;

// Configure the layer data reference
using LayerData = mininn::InferenceLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,         // weight element type
    (dx::linalg::MatrixLayoutEnum)MINIDXNN_WEIGHT_MATRIX_LAYOUT,
    dx::linalg::DATA_TYPE_FLOAT16,         // bias type
    dx::linalg::DATA_TYPE_FLOAT16,         // accumulator type
    ActivationHidden,
    ActivationOutput,
    dx::linalg::DATA_TYPE_FLOAT16,         // activation element type
    MINIDXNN_WEIGHT_MATRIX_ALIGNMENT,      // matrix alignment (128)
    MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,  // stride alignment (16)
    MINIDXNN_BIAS_VECTOR_ALIGNMENT         // bias alignment (128)
>;

ByteAddressBuffer WeightBuffer : register(t0);
ByteAddressBuffer BiasBuffer   : register(t1);
ByteAddressBuffer InputBuffer  : register(t2);
RWByteAddressBuffer OutputBuffer : register(u0);

// firstLayerMatSize = size in bytes of the first layer's weight matrix
// hiddenLayerMatSize = size in bytes of hidden layer weight matrices
int firstLayerMatSize;
int hiddenLayerMatSize;

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // Load input vector
    vector<half, 2> input = InputBuffer.Load<half2>(tid.x * 4);

    // Set up layer data with weight and bias buffers
    LayerData layerData;
    layerData.setWeightData(WeightBuffer, uint2(firstLayerMatSize, hiddenLayerMatSize));
    layerData.setBiasData(BiasBuffer);

    // Run forward pass
    vector<half, 2> output;
    mininn::forward(output, input, layerData);

    // Store result
    OutputBuffer.Store<half2>(tid.x * 4, output);
}
```

### Key Points

- **`setWeightData(buffer, uint2(firstSize, hiddenSize))`**: The `uint2` contains the byte sizes of the first layer's weight matrix and the backbone (hidden) layers' weight matrices. These sizes come from `D3D12MatrixInfo::m_dataSize` after packing.
- **`setBiasData(buffer)`**: Bias data must be packed with 128-byte alignment between layers.
- **`mininn::forward(output, input, layerData)`**: Internally uses `dx::linalg::Matrix::Multiply` or `MultiplyAdd` for hardware-accelerated inference.

**Source reference**:
- [`include/minidxnn/hlsl/mlp.hlsl`](../include/minidxnn/hlsl/mlp.hlsl) — core library
- [`example/kernel/01_texture_inference.comp`](../example/kernel/01_texture_inference.comp) — complete inference shader
- [`example/kernel/texture_inference_common.hlsl`](../example/kernel/texture_inference_common.hlsl) — shared inference step

---

## Using mlp.hlsl for Training

Training requires additional buffers for gradient accumulation and logits caching.

### HLSL Shader Code

```hlsl
#include <minidxnn/hlsl/mlp.hlsl>

using TrainData = mininn::TrainingLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,         // weight type
    dx::linalg::MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL,  // gradient layout
    dx::linalg::DATA_TYPE_FLOAT16,         // weight gradient type
    dx::linalg::DATA_TYPE_FLOAT16,         // bias type
    dx::linalg::DATA_TYPE_FLOAT16,         // bias gradient type
    dx::linalg::DATA_TYPE_FLOAT16,         // accumulator type
    dx::linalg::DATA_TYPE_FLOAT16,         // logits cache type
    mininn::LeakyReluActivation,
    mininn::SigmoidActivation
>;

ByteAddressBuffer WeightBuffer;
RWByteAddressBuffer WeightGradBuffer;
ByteAddressBuffer BiasBuffer;
RWByteAddressBuffer BiasGradBuffer;
RWByteAddressBuffer LogitsCacheBuffer;

[numthreads(32, 1, 1)]
void trainStep(uint3 tid : SV_DispatchThreadID)
{
    TrainData layerData;
    layerData.setWeightData(WeightBuffer, uint2(firstMatSize, hiddenMatSize));
    layerData.setWeightGradientCache(WeightGradBuffer, uint2(firstGradMatSize, hiddenGradMatSize));
    layerData.setBiasData(BiasBuffer);
    layerData.setBiasGradientCache(BiasGradBuffer);
    layerData.setLogitsCache(LogitsCacheBuffer);

    vector<half, 2> input = /* load from buffer */;
    vector<half, 2> output;

    // Forward pass (caches logits for backward)
    mininn::forward(output, input, layerData);

    // Compute loss gradient
    vector<half, 2> lossGrad = /* e.g., MSE gradient */;

    // Backward pass (accumulates weight and bias gradients)
    mininn::backward(lossGrad, input, layerData);
}
```

**Source reference**:
- [`example/kernel/02_texture_training.comp`](../example/kernel/02_texture_training.comp) — training shader
- [`example/kernel/texture_training_common.hlsl`](../example/kernel/texture_training_common.hlsl) — shared training step
- [`docs/mlp_hlsl.md`](./mlp_hlsl.md) — full API reference

---

## Source Code Reference Map

### Project File Locations

| Component | Path | Description |
|-----------|------|-------------|
| **HLSL Library** | [`include/minidxnn/hlsl/mlp.hlsl`](../include/minidxnn/hlsl/mlp.hlsl) | Core MLP forward/backward with LinAlg Matrix |
| **D3D12 Format Utils** | [`example/common/d3d12_format.hpp`](../example/common/d3d12_format.hpp) | Alignment, stride, matrix/vector packing |
| **GPU Utilities** | [`example/common/gfx_utility.hpp`](../example/common/gfx_utility.hpp) | Buffer creation, matrix conversion, kernel dispatch |
| **GPU Utilities (impl)** | [`example/common/gfx_utility.cpp`](../example/common/gfx_utility.cpp) | Context creation, program/kernel creation |
| **Inference Example** | [`example/01_texture_inference/`](../example/01_texture_inference/) | Complete GPU inference pipeline |
| **Training Example** | [`example/02_texture_training/`](../example/02_texture_training/) | Complete GPU training pipeline |
| **Inference Kernel** | [`example/kernel/01_texture_inference.comp`](../example/kernel/01_texture_inference.comp) | HLSL inference compute shader |
| **Training Kernel** | [`example/kernel/02_texture_training.comp`](../example/kernel/02_texture_training.comp) | HLSL training compute shader |

### gfx Library Locations

| Component | Path | Description |
|-----------|------|-------------|
| **API Header** | [`third_party/gfx_dep/gfx/gfx.h`](../third_party/gfx_dep/gfx/gfx.h) | `gfxGetLinearAlgebraTier`, `gfxConvertMatrix`, etc. |
| **Implementation** | [`third_party/gfx_dep/gfx/gfx.cpp`](../third_party/gfx_dep/gfx/gfx.cpp) | D3D12 feature check, matrix conversion |

### External References

| Resource | Link |
|----------|------|
| D3D12 LinAlg Runtime Spec | [D3D12LinearAlgebraRuntimeFeatureSupport.html][linalg-overview] |
| HLSL LinAlg Matrix Spec | [hlsl-specs/proposals/0035-linalg-matrix.md][linalg-spec] |
| LinAlg Examples | [github.com/llvm-beanz/linalg-examples][linalg-examples] |
| Blog: D3D12 LinAlg Preview | [devblogs.microsoft.com/directx/d3d12-linalg-preview/][linalg-blog] |
| SM 6.10 / Agility SDK 721 Preview | [devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/][linalg-driver] |

---

## Full Pipeline Summary

```
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Install driver (SM 6.10 + LinAlg)  +  Enable Developer Mode         │
│  2. D3D12EnableExperimentalFeatures(D3D12ExperimentalShaderModels)       │
│  3. Create D3D12 device                                                  │
│  4. CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT)           │
│  5. Prepare weight matrices:                                             │
│     a. Pack as ROW_MAJOR with 16-byte stride alignment                  │
│     b. GetLinearAlgebraMatrixConversionDestinationInfo (query dest size) │
│     c. ConvertLinearAlgebraMatrix → MUL_OPTIMAL layout                  │
│  6. Prepare bias vectors with 128-byte alignment                         │
│  7. Compile shader with DXC: -T cs_6_10 -enable-16bit-types             │
│  8. #include <minidxnn/hlsl/mlp.hlsl> in your shader                    │
│  9. Dispatch compute shader → GPU-accelerated MLP inference/training    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

[linalg-spec]: https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md
[linalg-overview]: https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html
[linalg-overview-convert]: https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html#convert-matrix-to-desired-layout-and-type
[linalg-examples]: https://github.com/llvm-beanz/linalg-examples
[linalg-blog]: https://devblogs.microsoft.com/directx/d3d12-linalg-preview/
[linalg-driver]: https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/
[win-dev-mode]: https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
[dx12-agility-sdk-download]: https://devblogs.microsoft.com/directx/directx12agility/
[dx12-dxc-download]: https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.4
