# MiniDxNn - GPU‑accelerated MLP inference on DirectX 12 with Cooperative Vector

![CMake build on Windows](../../actions/workflows/cmake.yaml/badge.svg)

<p align="center"><img src="./docs/minidxnn-logo.png" width="200"></p>

An implementation of **MLP** (Multi-Layer Perceptron) inference using DirectX 12 [Cooperative Vector][coop-vec-spec]. This library demonstrates GPU-accelerated neural network inference with cutting-edge shader features.

**Key Features:**
- 🚀 **High Performance**: GPU-accelerated inference using [DirectX 12 Cooperative Vector][coop-vec-overview]
- 🎯 **Flexible Architecture**: Configurable layer dimensions, activation functions, and data types
- 🔧 **Single-header HLSL**: Easy integration into existing DirectX 12 projects

For an overview of Cooperative Vector, please read [D3D12 Cooperative Vector][coop-vec-overview].

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/GPUOpen-LibrariesAndSDKs/MiniDxNn.git
cd MiniDxNn

# Build library and examples (examples are ON by default)
cmake -B build
cmake --build build --config Release

# Build with tests enabled
cmake -B build -DMINIDXNN_BUILD_TESTS=ON
cmake --build build --config Release

# Run unit tests (from build/unittest as CWD)
cd build/unittest
ctest -C Release
```

## Requirements

- **Operating System**: Windows 10/11 with [Developer Mode][win-dev-mode] enabled
- **GPU**: Supports Shader Model 6.9 and Cooperative Vector in D3D12 (AMD Radeon™ RX 9000 Series GPUs or equivalent NVIDIA)
- **Build Tools**:
  - CMake 3.21.0 or higher
  - Visual Studio 2022 (or compatible C++ compiler with C++20 support)
  - Windows SDK with DirectX 12
- **Python** (for training): Python 3.8+ with PyTorch (optional, for example training)

## Installation

### 1. Enable Experimental Features

⚠️ **Important**: As of early 2026, Cooperative Vector requires experimental feature support:

#### System setup

1. Install [Cooperative Vector supported driver][coop-vec-driver]
2. Enable [Windows Developer Mode][win-dev-mode]

### 2. Clone the Repository

```bash
git clone --recursive https://github.com/GPUOpen-LibrariesAndSDKs/MiniDxNn.git
cd MiniDxNn
```

⚠️ **Important**: Use `--recursive` flag to include submodules (gfx, GoogleTest, CLI11).

## Build and Run

### Basic Build (Library + Examples)

By default, the core library and examples are built (`MINIDXNN_BUILD_EXAMPLES=ON`). Tests are off by default.

```bash
# Configure and build (library + examples)
cmake -B build
cmake --build build --config Release

# Build with tests enabled
cmake -B build -DMINIDXNN_BUILD_TESTS=ON
cmake --build build --config Release

# Run tests (from build/unittest as CWD)
cd build/unittest
ctest -C Release
```

The example executables will be built in the `build/example/Release/` directory (or `build/example/Debug/` for Debug builds). Run examples from the `build/example` directory as working directory. See [example/README.md](./example/README.md) for details on running the examples.

Unit tests are run from the `build/unittest` directory as working directory:

```bash
cd build/unittest
ctest -C Release
```

## Usage

Please follow [the installation instruction](#Installation) to enable experimental feature support on your system.

### DX12 setup

1. Use **Direct 12 Agility SDK** [Version 1.717.1 preview][dx12-agility-sdk-download] and **DirectX Shader Compiler** [Version v1.8.2505.1][dx12-dxc-download] for DX12 runtime
2. Enable [Experimental Shader Model][dx-experimental-shader-model] using [D3D12EnableExperimentalFeatures][dx-enable-experimental-features] **before** creating the D3D12 device
3. Compile shaders with **Shader Model 6.9**

### HLSL Integration

The core MLP inference functionality is provided as a header-only HLSL library. Include it in your compute shaders:

```hlsl
#include <hlsl/mlp.hlsl>

// Define your network architecture
static const uint NUM_HIDDEN_LAYERS = 2;
static const int INPUT_DIM = 2;
static const int HIDDEN_DIM = 64;
static const int OUTPUT_DIM = 3;

// Create layer data reference using convenience alias (ByteAddressBuffer, with bias)
using MlpLayerData = mininn::InferenceLayerDataRef<
    NUM_HIDDEN_LAYERS,
    HIDDEN_DIM,
    half,  // staging element type
    dx::linalg::DATA_TYPE_FLOAT16,
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    dx::linalg::DATA_TYPE_FLOAT16,  // bias type
    dx::linalg::DATA_TYPE_FLOAT16,  // accumulator type
    mininn::ReluActivation,          // hidden layer activation
    mininn::SigmoidActivation        // output layer activation
>;

ByteAddressBuffer g_weights : register(t0);
ByteAddressBuffer g_biases  : register(t1);

[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    MlpLayerData layerData;
    layerData.setWeightData(g_weights);
    layerData.setBiasData(g_biases);

    vector<float, INPUT_DIM> input = ...;  // your input
    vector<float, OUTPUT_DIM> output;

    mininn::forward(output, input, layerData);

    // Use output...
}
```

See the [MLP HLSL API Documentation](docs/mlp_hlsl.md) for complete details.

## Project Structure

```
MiniDxNn/
├── include/hlsl/            # HLSL shader library (header-only)
│   └── mlp.hlsl            # MLP forward pass with Cooperative Vector
├── docs/                 # Documentation
│   └── mlp_hlsl.md       # HLSL API documentation
├── example/                 # Example applications
│   ├── common/             # Shared C++ utilities
│   ├── kernel/             # HLSL compute shaders for examples
│   ├── 01_texture_inference/  # Texture synthesis example
│   └── README.md           # Example documentation
├── scripts/                 # Training and utilities
│   └── pyreference/        # Python reference implementations
│       └── texture_reconstruction_mlp.py  # MLP training script
├── unittest/               # Unit tests
│   ├── kernel/            # HLSL compute shaders for tests
│   ├── test.cpp/hpp       # Test infrastructure
│   └── unittest.cpp       # GoogleTest main
├── third_party/           # External dependencies
│   ├── gfx/              # DirectX 12 framework (submodule)
│   ├── googletest/       # Unit testing framework (submodule)
│   ├── CLI11/            # Command line parser (submodule)
│   └── half-2.2.1.zip    # Half-precision float library
├── cmake/                # CMake build scripts
├── tools/                # 
├── CMakeLists.txt        # Root build configuration
├── LICENSE               # MIT License
└── NOTICE.md             # Third-party notices
```

## Features

### Supported Architectures
- **Multi-Layer Perceptron (MLP)**: 0 to N hidden layers with configurable dimensions
- **Flexible Layer Sizes**: Independent input, hidden, and output dimensions

### Activation Functions
- **Identity**: Linear pass-through (f(x) = x)
- **Sigmoid**: Numerically stable sigmoid (f(x) = 1/(1+e^(-x)))
- **ReLU**: Rectified Linear Unit (f(x) = max(0, x))
- **Leaky ReLU**: Leaky ReLU with configurable slope (f(x) = max(0.01x, x))

### Data Types
- **float16** (half): Currently only half precision (float16) is supported for MLP computation

### Matrix Layouts
- **Row-Major**: Rows contiguous in memory

## Examples

See the [example directory](./example) for complete working examples:

1. **[01_texture_inference](./example/01_texture_inference)**: Texture synthesis using MLP
   - Train a network with PyTorch, export to binary format
   - GPU inference with Cooperative Vector
   - Image generation and export

For detailed usage, see [example/README.md](./example/README.md).

## Performance

The library is designed for high-performance inference:
- **GPU Acceleration**: Parallel execution across all threads
- **Cooperative Vector**: Leverages DirectX 12's advanced shader features for efficient SIMD operations
- **Memory Efficiency**: Optimized buffer layouts with configurable alignment

Benchmark results vary by GPU architecture, network size, and data types. See the examples for practical performance characteristics.

## Documentation

- **[MLP HLSL API Reference](docs/mlp_hlsl.md)**: Complete HLSL API documentation
- **[Example Guide](example/README.md)**: Example applications and usage patterns
- **[Cooperative Vector Spec][coop-vec-spec]**: HLSL specification
- **[D3D12 Cooperative Vector Blog][coop-vec-overview]**: Overview and getting started

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

## Third-Party Notices

This project uses the following third-party libraries:

- [**Half-precision floating-point library**](https://half.sourceforge.net/) - MIT License
- [**gfx**](https://github.com/gboisse/gfx) - Direct3D12 framework - MIT License
- [**CLI11**](https://github.com/CLIUtils/CLI11) - Command line parser for C++ - BSD-3-Clause License
- [**GoogleTest**](https://github.com/google/googletest) - Testing framework - BSD-3-Clause License

See [NOTICE.md](NOTICE.md) for complete third-party notices.

## References

- [Cooperative Vector Specification][coop-vec-spec] - HLSL Spec Proposal
- [D3D12 Cooperative Vector Overview][coop-vec-overview] - DirectX Developer Blog

## Acknowledgments

Developed by Advanced Micro Devices, Inc. as a demonstration of DirectX 12 Cooperative Vector capabilities for GPU-accelerated machine learning workloads.

---

[coop-vec-spec]: https://github.com/microsoft/hlsl-specs/blob/main/proposals/0029-cooperative-vector.md
[coop-vec-overview]: https://devblogs.microsoft.com/directx/cooperative-vector/
[coop-vec-driver]: https://devblogs.microsoft.com/directx/cooperative-vector/#get-running
[win-dev-mode]: https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
[dx-enable-experimental-features]: https://learn.microsoft.com/ja-jp/windows/win32/api/d3d12/nf-d3d12-d3d12enableexperimentalfeatures
[dx-experimental-shader-model]: https://devblogs.microsoft.com/directx/ser/#availability
[dx12-agility-sdk-download]: https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.717.1-preview
[dx12-dxc-download]: https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2505.1/dxc_2025_07_14.zip
