# MLP HLSL API Reference

API reference for [`include/minidxnn/hlsl/mlp.hlsl`](../include/minidxnn/hlsl/mlp.hlsl) — a header-only HLSL library for MLP forward and backward passes using DirectX 12 Cooperative Vector.

For project overview and build instructions, see the [top-level README](../README.md).

---

## Quick Start

### Inference

```hlsl
#include <minidxnn/hlsl/mlp.hlsl>

static const uint NUM_LAYERS = 2;   // total layers (hidden + 1)
static const int  HIDDEN_DIM = 64;

using LayerData = mininn::InferenceLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,        // weight type
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,        // bias type
    dx::linalg::DATA_TYPE_FLOAT16,        // accumulator type
    mininn::LeakyReluActivation,          // hidden activation
    mininn::SigmoidActivation             // output activation
>;

ByteAddressBuffer g_weights : register(t0);
ByteAddressBuffer g_biases  : register(t1);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    LayerData layerData;
    layerData.setWeightData(g_weights, uint2(firstLayerMatSize, hiddenLayerMatSize));
    layerData.setBiasData(g_biases);

    vector<half, 2> input = half2(tid.x * 0.01, tid.y * 0.01);
    vector<half, 2> output;
    mininn::forward(output, input, layerData);
}
```

### Training (forward + backward)

```hlsl
using TrainData = mininn::TrainingLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,        // weight type
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,        // weight gradient type
    dx::linalg::DATA_TYPE_FLOAT16,        // bias type
    dx::linalg::DATA_TYPE_FLOAT16,        // bias gradient type
    dx::linalg::DATA_TYPE_FLOAT16,        // accumulator type
    dx::linalg::DATA_TYPE_FLOAT16,        // logits cache type
    mininn::LeakyReluActivation,
    mininn::SigmoidActivation
>;

// ... set up layerData with weight, bias, gradient, and logits cache buffers ...
mininn::forward(output, input, layerData);  // caches logits for backward
mininn::backward(lossGrad, input, layerData);  // accumulates gradients
```

---

## Core Types

### LayerDataRefImpl

The base template holding all buffer references for an MLP layer stack. You normally use one of the convenience aliases below.

**Key template parameters:**

| Parameter | Description |
|-----------|-------------|
| `NUM_LAYERS` | Total number of layers (hidden layers + 1 output layer) |
| `HIDDEN_LAYER_DIM` | Dimension of each hidden layer |
| `WEIGHT_ELEM_TYPE` | Data type for weight elements (e.g. `DATA_TYPE_FLOAT16`) |
| `WEIGHT_MATRIX_LAYOUT` | Memory layout (`MATRIX_LAYOUT_ROW_MAJOR`, `MATRIX_LAYOUT_COLUMN_MAJOR`, `MATRIX_LAYOUT_MUL_OPTIMAL`, `MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL`) |
| `BIAS_ELEM_TYPE` | Data type for bias elements (default: same as weight) |
| `ACCUMULATOR_ELEM_TYPE` | Accumulation type for matrix operations (default: same as weight) |
| `ActivationHiddenT` | Activation for hidden layers (default: `IdentityActivation`) |
| `ActivationLastT` | Activation for the output layer (default: `IdentityActivation`) |
| `ACTIVATION_ELEM_TYPE` | Element type for activation computation (default: same as weight) |
| `WEIGHT_MATRIX_ALIGNMENT` | Weight matrix alignment in bytes (default: 128) |
| `WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT` | Weight row stride alignment in bytes (default: 16) |
| `BIAS_VECTOR_ALIGNMENT` | Bias vector alignment in bytes (default: 64) |

**Methods:**

| Method | Description |
|--------|-------------|
| `setWeightData(buffer, uint2 matrixSize, startOffset=0)` | Set weight buffer. `matrixSize.x` = first layer matrix byte size, `.y` = hidden layer matrix byte size. |
| `setBiasData(buffer, startOffset=0)` | Set bias buffer |
| `setWeightGradientCache(buffer, uint2 matrixSize, startOffset=0)` | Set weight gradient buffer (training only) |
| `setBiasGradientCache(buffer, startOffset=0)` | Set bias gradient buffer (training only) |
| `setLogitsCache(buffer, startOffset=0)` | Set pre-activation logits cache (training only) |

### Inference Aliases

| Alias | Buffer | Bias | Description |
|-------|--------|------|-------------|
| `InferenceLayerDataRef` | `ByteAddressBuffer` | ✅ | Read-only inference with bias |
| `InferenceLayerDataRefNoBias` | `ByteAddressBuffer` | ❌ | Read-only inference without bias |
| `RWInferenceLayerDataRef` | `RWByteAddressBuffer` | ✅ | Read-write inference with bias |
| `RWInferenceLayerDataRefNoBias` | `RWByteAddressBuffer` | ❌ | Read-write inference without bias |

These aliases fix the buffer type and bias flag, so you only specify:

```hlsl
mininn::InferenceLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT,
    BIAS_ELEM_TYPE,           // default: WEIGHT_ELEM_TYPE
    ACCUMULATOR_ELEM_TYPE,    // default: WEIGHT_ELEM_TYPE
    ActivationHiddenT,        // default: IdentityActivation
    ActivationLastT,          // default: IdentityActivation
    ACTIVATION_ELEM_TYPE,     // default: WEIGHT_ELEM_TYPE
    WEIGHT_ALIGNMENT,         // default: 128
    WEIGHT_STRIDE_ALIGNMENT,  // default: 16
    BIAS_ALIGNMENT            // default: 64
>
```

### Training Aliases

| Alias | Bias | Description |
|-------|------|-------------|
| `TrainingLayerDataRef` | ✅ | Training with bias — enables weight/bias gradient caches and logits cache |
| `TrainingLayerDataRefNoBias` | ❌ | Training without bias |

Training aliases enable gradient accumulation buffers (`RWByteAddressBuffer`) and a logits cache for the backward pass. Additional template parameters:

| Parameter | Description |
|-----------|-------------|
| `WEIGHT_GRADIENT_CACHE_ELEM_TYPE` | Element type for cached weight gradients |
| `BIAS_GRADIENT_CACHE_ELEM_TYPE` | Element type for cached bias gradients |
| `LOGITS_CACHE_ELEM_TYPE` | Element type for cached pre-activation values |

---

## Activation Functions

All activations implement `forward` and `backward` with this signature:

```hlsl
template <typename OutputElemT, typename InputElemT, int N>
void forward(out vector<OutputElemT, N> output, const vector<InputElemT, N> input);

template <typename OutputElemT, typename InputElemT, int N>
void backward(out vector<OutputElemT, N> gradient, const vector<InputElemT, N> input);
```

### Built-in

| Type | Formula | Notes |
|------|---------|-------|
| `IdentityActivation` | f(x) = x | Pass-through |
| `SigmoidActivation` | f(x) = 1/(1+e⁻ˣ) | Numerically stable via `exp(-abs(x))` |
| `ReluActivation` | f(x) = max(0, x) | |
| `LeakyReluActivation` | f(x) = max(0.01x, x) | Fixed slope = 0.01 |

### Custom Activations

Any struct with matching `forward` (and optionally `backward`) methods can be used:

```hlsl
struct TanhActivation {
    template <typename OutputElemT, typename InputElemT, int N>
    void forward(out vector<OutputElemT, N> output, const vector<InputElemT, N> input) {
        output = (vector<OutputElemT, N>)tanh((vector<InputElemT, N>)input);
    }
};
```

---

## Functions

### forward

```hlsl
template <typename OutputElemT, int OUTPUT_DIM,
          typename InputElemT,  int INPUT_DIM, ...>
void mininn::forward(out vector<OutputElemT, OUTPUT_DIM> output,
                     const vector<InputElemT, INPUT_DIM> input,
                     const LayerDataRefImpl<...> layerData);
```

Runs a full forward pass: for each layer, computes `weight × input + bias` then applies the activation. Hidden layers use `ActivationHiddenT`; the final layer uses `ActivationLastT`. When using a training layer data type, pre-activation values (logits) are cached for the backward pass.

### backward

```hlsl
template <typename OutputElemT, int OUTPUT_DIM,
          typename InputElemT,  int INPUT_DIM, ...>
vector<OutputElemT, INPUT_DIM>
mininn::backward(const vector<OutputElemT, OUTPUT_DIM> lossGrad,
                 const vector<InputElemT, INPUT_DIM> input,
                 const LayerDataRefImpl<...> layerData);
```

Runs a full backward pass using cached logits from the preceding `forward` call. Accumulates weight and bias gradients into the gradient cache buffers (via atomic adds). Returns the upstream gradient with respect to the input.

---

## Network Architecture

```
NUM_LAYERS > 1:                          NUM_LAYERS == 1:

Input (INPUT_DIM)                        Input (INPUT_DIM)
  ↓                                        ↓
[W₀ × input + b₀] → ActivationHidden    [W × input + b] → ActivationLast
  ↓                                        ↓
Hidden (HIDDEN_DIM)                      Output (OUTPUT_DIM)
  ↓
  ... repeat for each hidden layer ...
  ↓
[Wₙ × hidden + bₙ] → ActivationLast
  ↓
Output (OUTPUT_DIM)
```

---

## Memory Layout

### Weight Matrix Packing

Each layer's weight matrix (`outputDim × inputDim`, row-major) is packed as follows:

1. **Row stride**: `align(inputDim * sizeof(elem), WEIGHT_STRIDE_ALIGNMENT)`
2. **Layer size**: `align(outputDim * stride, WEIGHT_ALIGNMENT)`
3. All layers concatenated in a single buffer

### Bias Vector Packing

Each layer's bias vector (`outputDim`) is padded to `align(outputDim * sizeof(elem), BIAS_ALIGNMENT)`, then concatenated.

### Host-Side Alignment

The host code must use the same alignment constants:

```cpp
constexpr size_t MATRIX_ALIGNMENT        = 128;  // matches WEIGHT_ALIGNMENT
constexpr size_t MATRIX_STRIDE_ALIGNMENT = 16;   // matches WEIGHT_STRIDE_ALIGNMENT
constexpr size_t VECTOR_ALIGNMENT        = 64;   // matches BIAS_ALIGNMENT
```

See `example/common/gfx_utility.hpp` (`convertToMatrixBuffer`, `convertToVectorBuffer`) for the full GPU buffer packing implementation.

---

## Preprocessor Options

| Define | Effect |
|--------|--------|
| `MINIDXNN_NO_INCLUDE_DX_LINALG` | Skip `#include <dx/linalg.h>` (provide it yourself) |
| `MINIDXNN_USE_SOFTWARE_LINALG_IMPL` | Use software fallback for matrix-vector ops instead of Cooperative Vector intrinsics |

---

## Additional Resources

- [Example Code](../example) — complete working examples
- [Unit Tests](../unittest) — test cases demonstrating API usage
- [Cooperative Vector Spec][coop-vec-spec] — HLSL specification
- [DirectX Blog][coop-vec-overview] — getting started with Cooperative Vector

[coop-vec-spec]: https://github.com/microsoft/hlsl-specs/blob/main/proposals/0029-cooperative-vector.md
[coop-vec-overview]: https://devblogs.microsoft.com/directx/cooperative-vector/

---

MIT License — Copyright (c) 2026 Advanced Micro Devices, Inc.
