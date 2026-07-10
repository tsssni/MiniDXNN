# MiniDXNN Examples

Example applications for GPU-accelerated MLP inference and training with DirectX 12 LinAlg Matrix.

## Table of Contents

- [Example 01: Texture Inference](#example-01-texture-inference)
- [Example 02: Texture Training](#example-02-texture-training)
- [Example 03: Texture Compression with Input Encoding](#example-03-texture-compression-with-input-encoding)
- [Example 04: Texture Compression App](#example-04-texture-compression-app)
- [MLP Model Format](#mlp-model-format)

---

## Example 01: Texture Inference

Loads a pre-trained MLP binary and reconstructs a texture on the GPU.

**Workflow:** Train & export (Python) → Load binary → GPU inference → Save PPM image

### Step-by-Step

#### 1. Train a model (Python)

```bash
pip install torch numpy matplotlib
cd scripts/reference
python texture_training.py            # defaults: 4 hidden layers, 64 neurons, leaky_relu
```

<details>
<summary>Training options</summary>

| Option | Default | Description |
|--------|---------|-------------|
| `--backbone-layers` | `4` | Number of hidden layers |
| `--hidden-dim` | `64` | Neurons per hidden layer |
| `--activation` | `leaky_relu` | `identity`, `sigmoid`, `tanh`, `relu`, `leaky_relu` |
| `--texture-pattern` | `checkerboard` | `gradient`, `checkerboard`, `stripes`, `circle`, `perlin` |
| `--epochs` | `30` | Training iterations |
| `--learning-rate` | `0.0025` | Optimizer learning rate |
| `--optimizer` | `lion` | `sgd`, `adam`, `lion` |
| `--dtype` | `float` | `float`, `half` |
| `--samples` | `200000` | Number of training samples |
| `--batch-size` | `2000` | Training batch size |
| `--no-display` | `false` | Skip matplotlib display and exit automatically |
| `--seed` | `987654321` | Random seed |
| `--texture-width` | `2048` | Texture width |
| `--texture-height` | `2048` | Texture height |

</details>

#### 2. Run GPU inference

```bash
cd build/example
Release/01-texture-inference.exe ../../scripts/reference/texture-mlp-data.bin output.ppm
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `mlp-binary` | Yes | — | Path to MLP binary file |
| `output` | No | `mlp-inference-output.ppm` | Output PPM image |
| `--texture-width` | No | `4096` | Image width |
| `--texture-height` | No | `4096` | Image height |
| `--cpu` | No | `false` | Use CPU reference ML operations instead of GPU |
| `--cpp-fallback` | No | `false` | Use C++ fallback (mlp.hlsl compiled as C++) |
| `--software-linalg` | No | `false` | Use software linear algebra instead of LinAlg Matrix |
| `--debug` | No | `false` | Verbose output |

---

## Example 02: Texture Training

Trains an MLP **entirely on the GPU** to learn a 2D texture pattern, then reconstructs the texture using the trained model.

**Workflow:** Init weights → GPU mini-batch training (SGD/Adam/Lion) → Reconstruct texture → Save PPM image

### Run

```bash
cd build/example
Release/02-texture-training.exe --output-image training-result.ppm
```

| Option | Default | Description |
|--------|---------|-------------|
| `--backbone-layers` | `4` | Number of hidden layers |
| `--hidden-dim` | `64` | Neurons per hidden layer |
| `--activation` | `leaky_relu` | `identity`, `sigmoid`, `tanh`, `relu`, `leaky_relu` |
| `--bias` / `--no-bias` | `true` | Enable/disable bias in MLP layers |
| `--epochs` | `30` | Training epochs |
| `--batch-size` | `2000` | Mini-batch size |
| `--learning-rate` | `0.0025` | Optimizer learning rate |
| `--optimizer` | `lion` | `sgd`, `adam`, `lion` |
| `--samples` | `200000` | Number of training samples |
| `--texture-width` | `2048` | Texture width |
| `--texture-height` | `2048` | Texture height |
| `--texture-pattern` | `checkerboard` | `gradient`, `checkerboard`, `stripes`, `circle`, `perlin` |
| `--output-image` | `mlp-training-output.ppm` | Output PPM image |
| `--cpu` | `false` | Use CPU reference ML operations instead of GPU |
| `--cpp-fallback` | `false` | Use C++ fallback (mlp.hlsl compiled as C++) |
| `--software-linalg` | `false` | Use software linear algebra instead of LinAlg Matrix |
| `--debug` | `false` | Enable debug mode for detailed output |
| `--seed` | `987654321` | Random seed |

### GPU Kernel Details

The training compute shaders (`example/kernel/02_texture_training.comp`) demonstrate:
- Using `mininn::TrainingLayerDataRef` to bind weight, bias, gradient, and logits cache buffers
- Calling `mininn::forward()` followed by `mininn::backward()` for a full training step
- GPU-side optimizer kernels (SGD, Adam, Lion) that read gradients and update weights directly on the GPU
- Shared optimizer implementations in `kernel/optimizer.hlsl` that work on packed byte buffers

---

## Example 03: Texture Compression with Input Encoding

Trains an MLP on the GPU to map normalized UV coordinates `(u, v)` to RGB pixel values, optionally using positional or grid input encoding for higher-quality texture compression.

See [`03_texture_compression_with_input_encoding/README.md`](./03_texture_compression_with_input_encoding/README.md) for the full option list, input-encoding modes, and example recipes.

---

---

## Example 04: Texture Compression App

Real-time windowed application that trains an MLP to compress a 2D texture on the GPU. Training runs incrementally (a configurable number of epochs per frame), and the reconstructed texture is displayed live as a fullscreen quad with an ImGui overlay for monitoring and parameter control.

Based on Example 03 but restructured for interactive use. See [`04_texture_compression_app/README.md`](./04_texture_compression_app/README.md) for the full option list and ImGui controls.

### Build

```bash
cmake --build build --target 04-texture-compression-app --config Release
```

> Requires `GFX_ENABLE_GUI ON` in `third_party/gfx_dep/CMakeLists.txt` for ImGui support.

### Run

```bash
cd build/example

# Default: train on a checkerboard pattern
Release/04-texture-compression-app.exe

# Train on a PNG image
Release/04-texture-compression-app.exe --input-image photo.png

# Faster training: 5 epochs per frame with larger batch
Release/04-texture-compression-app.exe --input-image photo.png --epochs-per-frame 5 --batch-size 50000
```

| Option | Default | Description |
|---|---|---|
| `--backbone-layers` | 4 | Number of MLP backbone layers |
| `--hidden-dim` | 64 | Hidden layer dimension |
| `--activation` | leaky_relu | Activation: identity, sigmoid, tanh, relu, leaky_relu |
| `--bias` / `--no-bias` | true | Enable/disable bias in MLP layers |
| `--optimizer` | adam | Optimizer: sgd, adam, lion |
| `--learning-rate` | 0.005 | Learning rate |
| `--batch-size` | 20000 | Training batch size (adjustable at runtime via ImGui) |
| `--epochs-per-frame` | 1 | Training epochs per frame |
| `--input-encoding` | positional | Input encoding: none, positional |
| `--positional-frequencies` | 4 | Frequency bands for positional encoding |
| `--input-image` | — | PNG image to use as ground truth |
| `--texture-width` | 2048 | Generated texture width (ignored with `--input-image`) |
| `--texture-height` | 2048 | Generated texture height (ignored with `--input-image`) |
| `--texture-pattern` | checkerboard | Pattern: gradient, checkerboard, stripes, circle, perlin |
| `--window-width` | 1024 | Window width |
| `--window-height` | 1024 | Window height |
| `--loss-scale` | 512 | Loss scale for FP16 gradient stability |
| `--software-linalg` | false | Use software linear algebra |
| `--debug` | false | Enable GPU debug/shader debugging |

### Key Differences from Example 03

- Windowed with live display instead of headless
- GPU-only (no C++ fallback path)
- Incremental training across frames
- Interactive parameter tuning via ImGui
- No output PNG — display is live on screen

---

## MLP Model Format

Binary format used by Example 01 (produced by the Python training script):

```
Header (12 bytes):
  int32  numHiddenLayers
  int32  hiddenLayerDim
  int32  activationType    (0=Identity, 1=Sigmoid, 2=Tanh, 3=ReLU, 4=LeakyReLU)

Per layer (numHiddenLayers + 1 layers):
  float32[outputDim × inputDim]   Weight matrix (row-major)
  float32[outputDim]              Bias vector
```

Layer ordering: Input(2)→Hidden₁→…→Hiddenₙ→Output(2). Hidden layers use the header's activation; the output layer always uses Sigmoid.

---

For HLSL API details, see [MLP HLSL API Reference](../docs/mlp_hlsl.md).

## License

MIT License — see [LICENSE](../LICENSE).

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
