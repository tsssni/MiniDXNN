# MiniDXNN — MLP Inference & Training on DirectX 12 with LinAlg Matrix

![CMake build on Windows](../../../actions/workflows/cmake.yaml/badge.svg)

<p align="center"><img src="./docs/minidxnn-logo.png" width="200"></p>

An implementation of **MLP** (Multi-Layer Perceptron) inference and training using DirectX 12 [LinAlg Matrix][linalg-spec]. This library demonstrates GPU-accelerated neural network inference and training with cutting-edge shader features.

- 🚀 **High Performance**: GPU-accelerated inference and training using [LinAlg Matrix][linalg-overview]
- 🔧 **Flexible Architecture**: Configurable layers, activations, and data types
- 🎯 **Single-header HLSL**: Easy to integrate into any DX12 project

## Requirements

- **OS**: Windows 11 with [Developer Mode][win-dev-mode] enabled
- **GPU**: Supports Shader Model 6.10 and LinAlg Matrix in D3D12 (AMD Radeon™ RX 9000 Series GPUs or equivalent NVIDIA)
- **Build**: CMake ≥ 3.21, Visual Studio 2022 (C++20), Windows SDK
- **DX12 Runtime**: [Agility SDK 1.721-preview][dx12-agility-sdk-download], [DXC v1.10.2605.4][dx12-dxc-download]
- **Python**: Python 3.8+ with PyTorch (optional, for example python training)

## Getting Started

```bash
# Clone with submodules (gfx, GoogleTest, CLI11)
git clone --recursive https://github.com/amdadvtech/MiniDXNN.git
cd MiniDXNN

# Build (library + examples)
cmake -B build
cmake --build build --config Release

# Optional: build & run tests
cmake -B build -DMINIDXNN_BUILD_TESTS=ON
cmake --build build --config Release
cd build/unittest && ctest -C Release
```

Example binaries are output to `build/example/Release/`. Run them from `build/example/` as the working directory. See [example/README.md](./example/README.md) for details.

## DX12 Setup

⚠️ **Important**: As of early 2026, LinAlg Matrix requires experimental feature support.

1. Install a [LinAlg Matrix supported driver][linalg-driver]
2. Enable [Experimental Shader Model][dx-experimental-shader-model] with [D3D12EnableExperimentalFeatures][dx-enable-experimental-features] **before** creating the device
3. Compile shaders with **Shader Model 6.10**

For a detailed walkthrough — including feature checks, weight matrix conversion (`GetLinearAlgebraMatrixConversionDestinationInfo` / `ConvertLinearAlgebraMatrix`), bias alignment, and full sample code — see the **[LinAlg Matrix MLP Guide](docs/linalg_matrix_mlp.md)**.

## HLSL Usage

Include the header-only library in your compute shader:

```hlsl
#include <minidxnn/hlsl/mlp.hlsl>

static const uint NUM_LAYERS = 3;       // total layers (hidden + 1)
static const int  INPUT_DIM  = 2;
static const int  HIDDEN_DIM = 64;
static const int  OUTPUT_DIM = 2;

using LayerData = mininn::InferenceLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    dx::linalg::DATA_TYPE_FLOAT16,
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,      // bias type
    dx::linalg::DATA_TYPE_FLOAT16,      // accumulator type
    mininn::LeakyReluActivation,        // hidden activation
    mininn::SigmoidActivation           // output activation
>;

ByteAddressBuffer g_weights : register(t0);
ByteAddressBuffer g_biases  : register(t1);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    LayerData layerData;
    layerData.setWeightData(g_weights, uint2(firstLayerMatSize, hiddenLayerMatSize));
    layerData.setBiasData(g_biases);

    vector<half, INPUT_DIM> input = ...;
    vector<half, OUTPUT_DIM> output;
    mininn::forward(output, input, layerData);
}
```

See the [HLSL API Reference](docs/mlp_hlsl.md) for the full API including training (`backward`).

## C++ Fallback Mode

MiniDXNN can be built **without** DirectX 12 or GPU dependencies, using a pure C++ fallback path. This is useful for CI, unit testing, or platforms without DX12 support (e.g. Linux).

```bash
# Build with CPU fallback only (no GPU/DX12 required)
cmake -B build \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMINIDXNN_CPP_FALLBACK_ONLY=ON \
  -DMINIDXNN_BUILD_TESTS=ON \
  -DMINIDXNN_BUILD_EXAMPLES=ON

cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure
```

### How It Works

The CPU fallback compiles `include/minidxnn/hlsl/mlp.hlsl` as standard C++ by providing HLSL-compatible shims in `include/minidxnn/cpp/hlsl_compat.hpp`. This header maps HLSL intrinsics (`vector`, `ByteAddressBuffer`, `dx::linalg::*`) to C++ equivalents. The half-precision type is configurable via the `MINIDXNN_CPP_FALLBACK_HALF_TYPE` CMake compile definition (defaults to `half_float::half`).

## Project Structure

```
MiniDXNN/
├── include/minidxnn/
│   ├── hlsl/mlp.hlsl              # Header-only HLSL library: MLP forward & backward
│   └── cpp/hlsl_compat.hpp         # HLSL → C++ shim for CPU-only builds
├── docs/                          # Documentation
│   └── mlp_hlsl.md                #   HLSL API reference
├── example/                       # Example applications
│   ├── common/                    #   Shared C++ utilities
│   │   ├── mlp_layer.hpp          #     MLP layer data, CPU forward/backward, weight init
│   │   ├── cpp_fallback.hpp       #     C++ fallback infrastructure (buffer packing, etc.)
│   │   ├── optimizer.hpp          #     Optimizer implementations (SGD, Adam, Lion)
│   │   ├── activation.hpp         #     Activation functions (Identity–LeakyReLU + Tanh)
│   │   ├── loss.hpp               #     Loss functions (MSE)
│   │   └── ...                    #     GPU helpers, image I/O, textures, matrix, RNG
│   ├── kernel/                    #   HLSL compute shaders
│   │   ├── optimizer.hlsl         #     GPU optimizer kernels (SGD, Adam, Lion)
│   │   └── ...                    #     Training, inference, and shared kernel headers
│   ├── 01_texture_inference/      #   Inference from a pre-trained MLP binary
│   ├── 02_texture_training/       #   On-GPU training + reconstruction
│   ├── 03_texture_compression_with_input_encoding/  # Training with positional/grid input encoding
│   ├── 04_texture_compression_app/  # Interactive GUI app with live display and ImGui controls
│   └── README.md                  #   Example documentation
├── scripts/reference/             # Python reference implementation
│   ├── texture_training.py            # Train & export MLP model
│   └── xoshiro128p.py                 # RNG matching C++ xoshiro128+
├── unittest/                      # GoogleTest unit tests
├── third_party/                   # Submodules & vendored deps
├── cmake/                         # CMake scripts
└── tools/natvis/                  # Debugging helpers (Visual Studio natvis)
```

## Features

| Category | Details |
|----------|---------|
| **Architecture** | MLP with 0–N hidden layers, independent input/hidden/output dimensions |
| **Operations** | Forward pass (inference), backward pass (training with gradient accumulation), wave-reduced vector accumulation (`WaveActiveSum`-based) |
| **Activations** | Identity, Sigmoid, ReLU, Leaky ReLU (custom activations supported — e.g. Tanh) |
| **Data type** | float16 (`DATA_TYPE_FLOAT16`) — currently the only tested type |
| **Matrix layout** | Row-major, Column-major, Mul-optimal, Outer-product-optimal |

## Examples

| # | Name | Description |
|---|------|-------------|
| 01 | [Texture Inference](./example/01_texture_inference) | Load a pre-trained MLP binary and reconstruct a texture on the GPU |
| 02 | [Texture Training](./example/02_texture_training) | Train an MLP on-GPU to learn a 2D texture pattern, then reconstruct it |
| 03 | [Texture Compression with Input Encoding](./example/03_texture_compression_with_input_encoding) | Train with positional/grid input encoding for higher-quality texture compression |
| 04 | [Texture Compression App](./example/04_texture_compression_app) | Interactive GUI app — incremental per-frame training with live display and ImGui controls |

See [example/README.md](./example/README.md) for step-by-step instructions.

## Documentation

- [LinAlg Matrix MLP Guide](docs/linalg_matrix_mlp.md) — Step-by-step setup, weight conversion, bias alignment, and shader compilation
- [HLSL API Reference](docs/mlp_hlsl.md) — `mlp.hlsl` types, functions, and memory layout
- [Example Guide](example/README.md) — building and running the examples
- [LinAlg Matrix Spec][linalg-spec] — HLSL specification
- [D3D12 LinAlg Matrix Overview][linalg-overview] — runtime feature support and getting started
- [LinAlg Examples][linalg-examples] — official example code

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

## Third-Party Notices

- [Half-precision floating-point library](https://half.sourceforge.net/) — MIT
- [gfx](https://github.com/gboisse/gfx) — MIT
- [CLI11](https://github.com/CLIUtils/CLI11) — BSD-3-Clause
- [GoogleTest](https://github.com/google/googletest) — BSD-3-Clause
- [stb_image](https://github.com/nothings/stb) — MIT/Public Domain

See [NOTICE.md](NOTICE.md) for details.

### CMake dependency downloads

When building with GPU support (without `MINIDXNN_CPP_FALLBACK_ONLY`), CMake auto-downloads dependencies to `third_party/gfx_dep/gfx/third_party/`.

---

[linalg-spec]: https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md
[linalg-overview]: https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html#tier-1-support
[linalg-driver]: https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/
[linalg-examples]: https://github.com/llvm-beanz/linalg-examples
[win-dev-mode]: https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
[dx-enable-experimental-features]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-d3d12enableexperimentalfeatures
[dx-experimental-shader-model]: https://devblogs.microsoft.com/directx/ser/#availability
[dx12-agility-sdk-download]: https://devblogs.microsoft.com/directx/directx12agility/
[dx12-dxc-download]: https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.4
