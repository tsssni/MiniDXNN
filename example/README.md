# MiniDxNN Examples

This directory contains example applications demonstrating GPU-accelerated MLP inference using the MiniDxNN library with DirectX 12 Cooperative Vector.

## Overview

The examples showcase real-world use cases of MLP inference on the GPU. Each example demonstrates:
- Training a network using PyTorch and exporting model weights to binary format
- Loading the binary MLP data and running inference with DirectX 12 Cooperative Vector

## Table of Contents

- [Example 01: Texture Inference](#example-01-texture-inference)
- [MLP Model Format](#mlp-model-format)

## Example 01: Texture Inference

### Description

This example demonstrates neural implicit texture synthesis using an MLP that learns to map 2D UV coordinates to 2D output values (used as grayscale color). The workflow is:

1. **Train & Export**: Train an MLP using PyTorch and dump the trained weights/biases to a binary file
2. **Inference**: Load the binary MLP data and generate a texture using DirectX 12 compute shaders with Cooperative Vector
3. **Output**: Save the generated texture as a PPM image

### Step-by-Step Guide

Follow these steps to train a model and run GPU inference:

#### 1. Install Python Dependencies

The training script requires Python 3.8+ with PyTorch and related packages:

```bash
pip install torch numpy matplotlib
```

#### 2. Train the Model and Export Binary Data

Run the provided Python script to train an MLP and dump the model data as a binary file:

```bash
cd scripts/pyreference

# Basic training (defaults: 3 hidden layers, 64 neurons, leaky_relu activation)
python texture_reconstruction_mlp.py
```

The script will:
- Generate synthetic training data based on the chosen texture pattern
- Train the MLP and display training loss progression
- Save the trained model as `texture-mlp-data.bin` in the current directory
- Display original and reconstructed texture images for comparison

You can customize the training configuration:

```bash
python texture_reconstruction_mlp.py \
    --numHiddenLayers 4 \
    --hiddenLayerDim 128 \
    --activation leaky_relu \
    --texturePattern gradient \
    --epochs 50 \
    --learningRate 0.01
```

**Training Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--numHiddenLayers` | `3` | Number of hidden layers |
| `--hiddenLayerDim` | `64` | Neurons per hidden layer |
| `--activation` | `leaky_relu` | Activation function (`identity`, `sigmoid`, `tanh`, `relu`, `leaky_relu`) |
| `--texturePattern` | `gradient` | Pattern type (`gradient`, `checkerboard`, `stripes`, `circle`, `perlin`) |
| `--epochs` | `20` | Training iterations |
| `--learningRate` | `0.01` | Optimizer learning rate |
| `--optimizer` | `sgd` | Optimizer type (`sgd`, `adam`) |
| `--dtype` | `float` | Data type for training (`float`, `half`) |
| `--numSamples` | `10000` | Number of training samples |
| `--batchSize` | `10` | Training batch size |
| `--seed` | `987654321` | Random seed for reproducibility |
| `--textureWidth` | `512` | Texture width resolution |
| `--textureHeight` | `512` | Texture height resolution |

#### 3. Run GPU Inference

After [building the project](../README.md#build-and-run) with examples enabled and training a model, run the inference example:

```bash
cd build/example

# Run inference with the trained model binary
Release/01-texture-inference.exe ../../scripts/pyreference/texture-mlp-data.bin output.ppm
```

The application loads the binary MLP data, creates DirectX 12 buffers for weights and biases, compiles the HLSL compute kernel at runtime with the network configuration, and dispatches parallel inference across all pixels.

**Command Line Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `mlp-binary` | Yes | — | Path to the binary MLP model file |
| `output` | No | `mlp-inference-output.ppm` | Output PPM image path |
| `--texture-width` | No | `4096` | Image width in pixels |
| `--texture-height` | No | `4096` | Image height in pixels |
| `--software-linalg` | No | `false` | Use software linear algebra implementation on HLSL instead of Cooperative Vector intrinsics |
| `--debug` | No | `false` | Enable debug mode for detailed output |

For system requirements and environment setup (Cooperative Vector driver, Developer Mode, Shader Model 6.9), see the [top-level README](../README.md#requirements).

### GPU Kernel Details

The `kernel/01_texture_inference.comp` compute shader demonstrates:
- Including the MiniDxNN HLSL library (`#include "mlp.hlsl"`)
- Configuring MLP architecture using compile-time defines (e.g., `MINIDXNN_NUM_HIDDEN_LAYERS`, `MINIDXNN_HIDDEN_LAYER_DIMENSIONS`)
- Using `mininn::InferenceLayerDataRef` / `mininn::InferenceLayerDataRefNoBias` to bind DirectX 12 buffers
- Calling `mininn::forward()` for parallel inference across UV space
- Writing results to output buffers

Key features:
- Compile-time network configuration via shader defines
- Cooperative Vector acceleration
- Mixed precision support (float16 for MLP computation)

---

## MLP Model Format

The binary model file format used by the examples:

### File Structure

```
[Header]
- numHiddenLayers (int32): Number of hidden layers
- hiddenLayerDim  (int32): Dimension of each hidden layer
- activation      (int32): Activation function type (see below)

[For each layer: Input → Hidden₁ → ... → Hiddenₙ → Output]
  [Weight Matrix]
  - Data: float32 array [outputDim × inputDim] in row-major order
  
  [Bias Vector]
  - Data: float32 array [outputDim]
```

### Layer Configuration

For a network with `numHiddenLayers` hidden layers:
1. **First Layer**: Input (2) → Hidden (hiddenLayerDim)
2. **Inner Layers** (×(numHiddenLayers-1)): Hidden → Hidden
3. **Output Layer**: Hidden → Output (2)

### Activation Type Encoding

| Value | Activation |
|-------|------------|
| `0` | Identity (linear) |
| `1` | Sigmoid |
| `2` | Tanh |
| `3` | ReLU |
| `4` | Leaky ReLU |

The hidden layers use the activation specified in the header. The output layer always uses **Sigmoid** activation for mapping values to the [0, 1] range.

### Example

A 3-hidden-layer network (2→64→64→64→2) with Leaky ReLU:
```
numHiddenLayers = 3
hiddenLayerDim = 64
activation = 4  # Leaky ReLU

Layer 0: weights[64×2], bias[64]      # Input → Hidden₁
Layer 1: weights[64×64], bias[64]     # Hidden₁ → Hidden₂
Layer 2: weights[64×64], bias[64]     # Hidden₂ → Hidden₃
Layer 3: weights[2×64], bias[2]       # Hidden₃ → Output (sigmoid)
```

---

## Next Steps

- **Explore the code**: Study `example.cpp` and the HLSL kernel
- **Experiment with architectures**: Try different layer sizes and activation functions
- **Create custom patterns**: Modify the Python training script for your own textures
- **Integrate into your project**: Use the HLSL library in your DirectX 12 application

For HLSL API details, see [MLP HLSL API Documentation](../docs/mlp_hlsl.md).

## License

MIT License - see [LICENSE](../LICENSE) for details.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
