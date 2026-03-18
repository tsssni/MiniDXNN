# MLP HLSL API Documentation

## Overview

API reference for [`include/hlsl/mlp.hlsl`](../include/hlsl/mlp.hlsl) — a header-only HLSL library for MLP inference using DirectX 12 Cooperative Vector.

For project overview, system requirements, and build instructions, see the [top-level README](../README.md).


## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Types](#core-types)
3. [Activation Functions](#activation-functions)
4. [Main API Functions](#main-api-functions)
5. [Usage Examples](#usage-examples)
6. [Network Architecture](#network-architecture)
7. [Memory Layout Considerations](#memory-layout-considerations)
8. [Performance Considerations](#performance-considerations)
9. [Advanced Features](#advanced-features)

---

## Quick Start

Here's a minimal example to get started:

```hlsl
#include <hlsl/mlp.hlsl>

// Define network: 2 inputs → 64 hidden → 2 outputs
static const uint NUM_HIDDEN_LAYERS = 1;
static const int INPUT_DIM = 2;
static const int HIDDEN_DIM = 64;
static const int OUTPUT_DIM = 2;

// Configure layer data
using LayerDataRef = mininn::InferenceLayerDataRef<
    NUM_HIDDEN_LAYERS,
    HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,      // weight storage type
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,      // bias storage type
    dx::linalg::DATA_TYPE_FLOAT16,      // accumulation type for matrix operations
    mininn::LeakyReluActivation,        // hidden activation
    mininn::SigmoidActivation,          // output activation
    dx::linalg::DATA_TYPE_FLOAT16       // computation type for activation functions
>;

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // Setup layer data
    LayerDataRef layerData;
    layerData.setWeightData(g_weightsBuffer);
    layerData.setBiasData(g_biasBuffer);
    
    // Run inference
    vector<half, INPUT_DIM> input = half2(tid.x * 0.01, tid.y * 0.01);
    vector<half, OUTPUT_DIM> output;
    
    mininn::forward(output, input, layerData);
    
    // Use output...
    g_outputBuffer[tid.x] = output;
}
```

---

## Core Types

### LayerDataRefImpl

The fundamental template structure that represents MLP layer data. It holds references to weight and bias buffers along with activation function instances.

```hlsl
template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          bool HAS_BIAS,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
struct LayerDataRefImpl
```

**Template Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUM_HIDDEN_LAYERS` | `uint` | — | Number of hidden layers in the network |
| `HIDDEN_LAYER_DIM` | `int` | — | Dimension of each hidden layer |
| `WeightBufferT` | typename | — | Buffer type for weight storage (`ByteAddressBuffer` or `RWByteAddressBuffer`) |
| `WEIGHT_ELEM_TYPE` | `dx::linalg::DataType` | — | Data type of weight elements |
| `WEIGHT_MATRIX_LAYOUT` | `dx::linalg::MatrixLayout` | — | Memory layout of weight matrices |
| `HAS_BIAS` | `bool` | — | Whether the network includes bias terms |
| `BiasBufferT` | typename | — | Buffer type for bias storage |
| `BIAS_ELEM_TYPE` | `dx::linalg::DataType` | `WEIGHT_ELEM_TYPE` | Data type of bias elements |
| `ACCUMULATOR_ELEM_TYPE` | `dx::linalg::DataType` | `WEIGHT_ELEM_TYPE` | Accumulation type for matrix operations |
| `ActivationHiddenT` | typename | `IdentityActivation` | Activation function type for hidden layers |
| `ActivationLastT` | typename | `IdentityActivation` | Activation function type for the output layer |
| `ACTIVATION_ELEM_TYPE` | `dx::linalg::DataType` | `WEIGHT_ELEM_TYPE` | Element type used for activation function computation |
| `IS_WEIGHT_MATRIX_TRANSPOSED` | `bool` | `false` | Whether weight matrices are stored transposed |
| `WEIGHT_ALIGNMENT` | `uint` | `128` | Memory alignment for weight matrices (bytes) |
| `WEIGHT_STRIDE_ALIGNMENT` | `uint` | `16` | Stride alignment for weight matrices (bytes) |
| `BIAS_ALIGNMENT` | `uint` | `64` | Memory alignment for bias vectors (bytes) |

**Methods:**

| Method | Description |
|--------|-------------|
| `setWeightData(WeightBufferT buffer, uint startOffset = 0)` | Sets the weight buffer and its start offset |
| `setBiasData(BiasBufferT buffer, uint startOffset = 0)` | Sets the bias buffer and its start offset |

**Members:**

| Member | Description |
|--------|-------------|
| `m_weight` | Weight buffer reference |
| `m_bias` | Bias buffer reference |
| `m_activationHidden` | Activation function instance for hidden layers |
| `m_activationLast` | Activation function instance for the output layer |

### Inference Helper Aliases

The following type aliases simplify `LayerDataRefImpl` for common inference use cases by fixing the buffer types and/or the bias flag.

#### InferenceLayerDataRefImpl

Direct alias of `LayerDataRefImpl` with the same template parameters. Serves as the base for the other inference aliases.

```hlsl
template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          bool HAS_BIAS,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
using InferenceLayerDataRefImpl = LayerDataRefImpl<...>;
```

#### InferenceLayerDataRef

Read-only inference with bias. Uses `ByteAddressBuffer` for both weight and bias buffers, and fixes `HAS_BIAS = true`. The `WeightBufferT`, `BiasBufferT`, and `HAS_BIAS` template parameters are omitted.

```hlsl
template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
using InferenceLayerDataRef = InferenceLayerDataRefImpl<
    ..., ByteAddressBuffer, ..., true, ByteAddressBuffer, ...>;
```

#### InferenceLayerDataRefNoBias

Read-only inference without bias. Uses `ByteAddressBuffer` and fixes `HAS_BIAS = false`. The `WeightBufferT`, `BiasBufferT`, `HAS_BIAS`, and `BIAS_ELEM_TYPE` template parameters are omitted.

```hlsl
template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
using InferenceLayerDataRefNoBias = InferenceLayerDataRefImpl<
    ..., ByteAddressBuffer, ..., false, ByteAddressBuffer, ...>;
```

#### RWInferenceLayerDataRef

Read-write inference with bias. Uses `RWByteAddressBuffer` for both weight and bias buffers, and fixes `HAS_BIAS = true`. Template parameters are the same as [`InferenceLayerDataRef`](#inferencelayerdataref).

```hlsl
template </* same as InferenceLayerDataRef */>
using RWInferenceLayerDataRef = InferenceLayerDataRefImpl<
    ..., RWByteAddressBuffer, ..., true, RWByteAddressBuffer, ...>;
```

#### RWInferenceLayerDataRefNoBias

Read-write inference without bias. Uses `RWByteAddressBuffer` and fixes `HAS_BIAS = false`. Template parameters are the same as [`InferenceLayerDataRefNoBias`](#inferencelayerdatarefnobias).

```hlsl
template </* same as InferenceLayerDataRefNoBias */>
using RWInferenceLayerDataRefNoBias = InferenceLayerDataRefImpl<
    ..., RWByteAddressBuffer, ..., false, RWByteAddressBuffer, ...>;
```

---

## Activation Functions

All activation functions implement a `forward` method with the following signature:

```hlsl
template <typename OutputElemT, typename InputElemT, int N>
void forward(out vector<OutputElemT, N> output, const vector<InputElemT, N> input)
```

### Built-in Activation Functions

#### IdentityActivation

Pass-through activation function.

**Formula:** `f(x) = x`

#### SigmoidActivation

Sigmoid activation function with numerically stable implementation.

**Formula:** `f(x) = 1 / (1 + e^(-x))`

**Implementation Details:**
- Uses `exp(-abs(x))` for numerical stability
- Handles positive and negative inputs separately using `select`
- Output range: (0, 1)

#### ReluActivation

Rectified Linear Unit activation function.

**Formula:** `f(x) = max(0, x)`

#### LeakyReluActivation

Leaky ReLU activation function with a fixed negative slope of 0.01.

**Formula:** `f(x) = max(0.01 * x, x)`

### Custom Activation Functions

You can define your own activation functions beyond those provided by `mlp.hlsl`. Any struct that implements the `forward` method matching the signature and output value type shown above can be used as an activation function:

```hlsl
struct MyCustomActivation
{
    template <typename OutputElemT, typename InputElemT, int N>
    void forward(out vector<OutputElemT, N> output, const vector<InputElemT, N> input)
    {
        // Your custom activation logic here
    }
};

// Use it with any layer data type
using MyLayerData = mininn::InferenceLayerDataRef<
    NUM_HIDDEN_LAYERS, HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,       // bias type
    dx::linalg::DATA_TYPE_FLOAT16,       // accumulator type
    MyCustomActivation,                  // custom hidden activation
    mininn::SigmoidActivation            // output activation
>;
```

---

## Main API Functions

### forward

Performs a forward pass through the MLP network.

```hlsl
template <typename OutputElemT, int OUTPUT_DIM,
          typename InputElemT, int INPUT_DIM,
          /* remaining template parameters deduced from layerData */>
void forward(out vector<OutputElemT, OUTPUT_DIM> output,
             const vector<InputElemT, INPUT_DIM> input,
             const LayerDataRefImpl<...> layerData)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `output` [out] | Output vector to store network results |
| `input` | Input vector to the network |
| `layerData` | `LayerDataRefImpl` (or any of its aliases) containing weight, bias, and activation data |

**Behavior:**
1. Computes matrix-vector products for each layer
2. Applies `m_activationHidden` after each hidden layer
3. Applies `m_activationLast` after the output layer
4. Supports networks with 0 or more hidden layers
5. Handles both biased and unbiased networks

The template parameters of `forward` are fully deduced from the types of `output`, `input`, and `layerData`, so you only need to specify the output element type and dimension explicitly when they cannot be inferred.

---

## Usage Examples

### Example 1: Simple 2-Layer MLP with ReLU

```hlsl
#include <hlsl/mlp.hlsl>

// Define network structure
static const uint NUM_HIDDEN_LAYERS = 1;
static const int INPUT_DIM = 16;
static const int HIDDEN_DIM = 32;
static const int OUTPUT_DIM = 8;

// Create layer data reference (read-only, with bias)
using MlpLayerData = mininn::InferenceLayerDataRef<
    NUM_HIDDEN_LAYERS,
    HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,      // bias type
    dx::linalg::DATA_TYPE_FLOAT16,      // accumulator type
    mininn::ReluActivation,             // hidden activation
    mininn::IdentityActivation          // output activation
>;

// Forward pass
void runMlp(ByteAddressBuffer weights, ByteAddressBuffer biases)
{
    MlpLayerData layerData;
    layerData.setWeightData(weights);
    layerData.setBiasData(biases);

    vector<float, INPUT_DIM> input = {...};  // your input data
    vector<float, OUTPUT_DIM> output;

    mininn::forward(output, input, layerData);
}
```

### Example 2: Deep Network with Leaky ReLU

```hlsl
// 3 hidden layers with Leaky ReLU activation
static const uint NUM_HIDDEN_LAYERS = 3;
static const int INPUT_DIM = 64;
static const int HIDDEN_DIM = 128;
static const int OUTPUT_DIM = 10;

using DeepMlpData = mininn::InferenceLayerDataRef<
    NUM_HIDDEN_LAYERS,
    HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,         // bias type
    dx::linalg::DATA_TYPE_FLOAT16,         // accumulator type
    mininn::LeakyReluActivation,           // hidden layers
    mininn::SigmoidActivation              // output layer
>;
```

### Example 3: Single-Layer Perceptron (No Hidden Layers)

```hlsl
// Linear transformation: input → output (no bias)
static const uint NUM_HIDDEN_LAYERS = 0;
static const int INPUT_DIM = 10;
static const int OUTPUT_DIM = 5;

using LinearLayerData = mininn::InferenceLayerDataRefNoBias<
    NUM_HIDDEN_LAYERS,
    0,  // hidden dim not used when no hidden layers
    dx::linalg::DATA_TYPE_FLOAT16,
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,      // accumulator type
    mininn::IdentityActivation,
    mininn::IdentityActivation
>;
```

---

## Network Architecture

The MLP implementation follows this architecture:

### For NUM_HIDDEN_LAYERS > 0:

```
Input (INPUT_DIM)
    ↓
[Weight₀ × Input + Bias₀]
    ↓
ActivationHidden
    ↓
Hidden Layer₁ (HIDDEN_DIM)
    ↓
[Weight₁ × Hidden₁ + Bias₁]
    ↓
ActivationHidden
    ↓
... (repeat for each hidden layer)
    ↓
[Weightₙ × Hiddenₙ₋₁ + Biasₙ]
    ↓
ActivationLast
    ↓
Output (OUTPUT_DIM)
```

### For NUM_HIDDEN_LAYERS == 0 (Single-Layer):

```
Input (INPUT_DIM)
    ↓
[Weight × Input + Bias]
    ↓
ActivationLast
    ↓
Output (OUTPUT_DIM)
```

---

## Memory Layout Considerations

### Weight Matrix Storage

The library currently supports **Row-Major** matrix layout only (`MATRIX_LAYOUT_ROW_MAJOR`): rows are contiguous in memory.

### Alignment Requirements

Cooperative Vector requires that weight and bias data in GPU buffers are properly aligned. The library uses the following default alignment values (configurable via template parameters):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WEIGHT_ALIGNMENT` | 128 bytes | Base address and per-layer offset alignment for weight matrices |
| `WEIGHT_STRIDE_ALIGNMENT` | 16 bytes | Row stride alignment for weight matrices |
| `BIAS_ALIGNMENT` | 64 bytes | Base address and per-layer offset alignment for bias vectors |

These alignment values must match between the HLSL shader and the host-side code that prepares the GPU buffers.

### Packing Weight and Bias Data for GPU Buffers

When uploading weight and bias data to GPU buffers, each layer's data must be aligned according to the alignment parameters. The library computes per-layer offsets internally, but the **host-side buffer packing must use the same alignment rules**.

#### Weight Matrix Packing

For each layer's weight matrix (with dimensions `outputDim × inputDim` in row-major order):

1. **Stride alignment**: Each row is padded so its stride (in bytes) is a multiple of `WEIGHT_STRIDE_ALIGNMENT`
   - `stride = align(inputDim * sizeof(element), WEIGHT_STRIDE_ALIGNMENT)`
2. **Matrix alignment**: The total size of each layer's matrix is padded to a multiple of `WEIGHT_ALIGNMENT`
   - `layerSize = align(outputDim * stride, WEIGHT_ALIGNMENT)`
3. All layers are packed contiguously in a single buffer with this per-layer padding

#### Bias Vector Packing

For each layer's bias vector (with dimension `outputDim`):

1. **Vector alignment**: Each layer's bias vector is padded to a multiple of `BIAS_ALIGNMENT`
   - `layerSize = align(outputDim * sizeof(element), BIAS_ALIGNMENT)`
2. All layers are packed contiguously in a single buffer with this per-layer padding

#### Example (C++ Host Side)

The example application in `example/common/mlp_layer.hpp` demonstrates the alignment logic with `packMatrixData()` and `packVectorData()`:

```cpp
// Alignment constants (must match HLSL template parameters)
constexpr size_t MATRIX_ALIGNMENT = 128;        // matches WEIGHT_ALIGNMENT
constexpr size_t MATRIX_STRIDE_ALIGNMENT = 16;   // matches WEIGHT_STRIDE_ALIGNMENT
constexpr size_t VECTOR_ALIGNMENT = 64;           // matches BIAS_ALIGNMENT

// Align a byte size up to the given alignment boundary
constexpr size_t align(size_t sizeInBytes, size_t alignmentInBytes) {
    return (sizeInBytes + alignmentInBytes - 1) & ~(alignmentInBytes - 1);
}

// Align element count so that (count * sizeof(Type)) meets the alignment
template <typename Type>
constexpr size_t alignN(size_t n, size_t alignmentInBytes) {
    return align(n * sizeof(Type), alignmentInBytes) / sizeof(Type);
}
```

The weight buffer is then created by packing each layer's rows with stride alignment and aligning each layer's total size:

```cpp
// For each layer:
size_t stride = alignN<half>(inputDim, MATRIX_STRIDE_ALIGNMENT);
size_t layerSize = alignN<half>(stride * outputDim, MATRIX_ALIGNMENT);
// Copy rows with padding, then advance offset by layerSize
```

See `example/common/gfx_utility.hpp` (`convertToMatrixBuffer`, `convertToVectorBuffer`) for the full GPU buffer creation flow.

### Buffer Offset Calculation

Weights and biases are stored in contiguous buffers. Use `setWeightData()` and `setBiasData()` to configure buffer references and start offsets. Internal layer offsets are computed automatically and account for alignment padding.

---

## Performance Considerations

1. **Data Types**: Currently only `half` (float16) is supported for MLP computation
2. **Matrix Layout**: Currently only Row-Major (`MATRIX_LAYOUT_ROW_MAJOR`) is supported
3. **Alignment**: Default values are optimized for AMD GPUs; adjust for other architectures
4. **Batch Processing**: For multiple inputs, consider calling `forward` in parallel threads

---

## Dependencies

This library requires:
- **dx/linalg.h**: DirectX linear algebra library (can be disabled with `MINIDXNN_NO_INCLUDE_DX_LINALG`)
- HLSL Shader Model 6.0+ (for template support)
- ByteAddressBuffer/RWByteAddressBuffer support

The `MINIDXNN_USE_SOFTWARE_LINALG_IMPL` option can be defined to use a software fallback for linear algebra operations instead of cooperative vector intrinsics.

---

## Error Handling

The library uses compile-time validation through template parameters. Common issues:

- **Alignment**: Ensure alignment parameters are powers of 2
- **Buffer Sizes**: Ensure weight and bias buffers are large enough for the network configuration
- **Type Compatibility**: Ensure input/output element types are compatible with buffer types
- **Dimension Mismatch**: Verify layer dimensions match between training and inference

---

## Advanced Features

### Transposed Weight Matrices

Set `IS_WEIGHT_MATRIX_TRANSPOSED = true` to work with pre-transposed weight matrices, which can improve memory access patterns for certain layouts.

### Supported Data Type

Currently only `half` (float16 / `DATA_TYPE_FLOAT16`) is supported for MLP computation. All weight, bias, accumulator, and activation element types should use `DATA_TYPE_FLOAT16`.

---

## Additional Resources

- **[Example Code](../example)**: Complete working examples
- **[Unit Tests](../unittest)**: Test cases demonstrating API usage
- **[Cooperative Vector Spec][coop-vec-spec]**: HLSL specification
- **[DirectX Blog Post][coop-vec-overview]**: Getting started with Cooperative Vector

[coop-vec-spec]: https://github.com/microsoft/hlsl-specs/blob/main/proposals/0029-cooperative-vector.md
[coop-vec-overview]: https://devblogs.microsoft.com/directx/cooperative-vector/

---

## License

MIT License - See file header for full license text.

Copyright (c) 2026 Advanced Micro Devices, Inc.
