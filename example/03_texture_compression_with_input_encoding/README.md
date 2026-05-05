# Example 03 — Texture Compression with Input Encoding

Trains an MLP on the GPU to reconstruct a 2D texture. The MLP learns to map normalized UV coordinates `(u, v) ∈ [0,1]²` to RGB pixel values, compressing the texture into a compact neural network representation.

Training uses DirectX 12 LinAlg Matrix (SM6.10) for hardware-accelerated matrix-vector operations.

## Pipeline

1. Load a ground-truth texture (PNG image or procedural pattern)
2. Initialize the MLP with He/Kaiming weight initialization
3. Sample random `(u, v)` coordinates and look up corresponding RGB values
4. Train on GPU with mini-batch optimization (SGD / Adam / Lion)
5. Reconstruct the full texture by evaluating the MLP at every pixel
6. Write the result as a PNG image

## Building

```
cmake --build <build-dir> --target 03-texture-compression-with-input-encoding
```

## Basic Usage

```
03-texture-compression-with-input-encoding [options]
```

By default, runs with grid input encoding (feature-dim 8, resolution 64), Adam optimizer with loss scaling, and 30 epochs:

```
03-texture-compression-with-input-encoding
```

## Input Encodings

The `--input-encoding` option controls how UV coordinates are transformed before being fed into the MLP. More expressive encodings help the MLP learn high-frequency detail.

### None

Raw `(u, v)` coordinates passed directly as 2D input to the MLP.

```
03-texture-compression-with-input-encoding --input-encoding none
```

Best for smooth, low-frequency textures. Struggles with sharp edges or fine detail.

### Positional Encoding

Maps `(u, v)` to a vector of sinusoidal features at multiple frequencies. For each frequency band `k` in `0..F-1`, four features are produced: `sin(2^k π u)`, `cos(2^k π u)`, `sin(2^k π v)`, `cos(2^k π v)`. Total input dim = `4 * F`.

```
03-texture-compression-with-input-encoding --input-encoding positional
03-texture-compression-with-input-encoding --input-encoding positional --positional-frequencies 8
```

- `--positional-frequencies` — number of frequency bands `F` (default: 4, output dim = 4 * F, range: 1–16)

More frequency bands capture finer detail at the cost of a wider first MLP layer. Good for natural textures with moderate to high-frequency content.

### Grid Encoding (default)

Stores learnable feature vectors at vertices of a regular grid. For each query `(u, v)`, bilinearly interpolates the four surrounding corner vectors to produce the MLP input. Grid features are optimized jointly with the MLP weights.

```
03-texture-compression-with-input-encoding --input-encoding grid --grid-resolution 64 --grid-feature-dim 8
```

- `--grid-resolution` — number of grid cells along each axis (default: 64, range: 2–1024)
- `--grid-feature-dim` — feature vector dimension at each grid vertex (default: 8, range: 1–64)

Grid encoding is the most expressive option. It excels at sharp textures and natural images. Higher resolution and feature dim capture more detail at the cost of memory and training time.

## Ground-Truth Texture

### Procedural patterns

```
03-texture-compression-with-input-encoding --texture-pattern checkerboard --texture-width 512 --texture-height 512
03-texture-compression-with-input-encoding --texture-pattern gradient
03-texture-compression-with-input-encoding --texture-pattern stripes
03-texture-compression-with-input-encoding --texture-pattern circle
03-texture-compression-with-input-encoding --texture-pattern perlin
```

### PNG image (RGB)

Load any PNG as the training target. The image dimensions become the output resolution.

```
03-texture-compression-with-input-encoding --input-image photo.png
```

When `--input-image` is specified, `--texture-pattern`, `--texture-width`, and `--texture-height` are ignored.

## MLP Architecture

```
03-texture-compression-with-input-encoding --backbone-layers 4 --hidden-dim 64 --activation leaky_relu --bias
```

| Option | Default | Description |
|---|---|---|
| `--backbone-layers` | 4 | Total number of layers |
| `--hidden-dim` | 64 | Width of each hidden layer |
| `--activation` | `leaky_relu` | Hidden activation: `identity`, `sigmoid`, `tanh`, `relu`, `leaky_relu` |
| `--bias` / `--no-bias` | bias on | Whether layers use a bias vector |

## Training Parameters

```
03-texture-compression-with-input-encoding --epochs 50 --batch-size 2000 --samples 200000 --learning-rate 0.005 --optimizer adam
```

| Option | Default | Description |
|---|---|---|
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 2000 | Samples per mini-batch |
| `--samples` | 200000 | Total training samples drawn from the texture |
| `--learning-rate` | 0.005 | Learning rate |
| `--optimizer` | `adam` | Optimizer: `sgd`, `adam`, `lion` |
| `--loss-scale` | 512 | Loss scaling factor for FP16 gradient stability |
| `--adam-beta1` | 0.9 | Adam first moment decay |
| `--adam-beta2` | 0.999 | Adam second moment decay |
| `--adam-epsilon` | 1e-6 | Adam epsilon for numerical stability |
| `--lion-beta1` | 0.9 | Lion interpolation parameter (gradient) |
| `--lion-beta2` | 0.99 | Lion interpolation parameter (momentum) |
| `--lion-weight-decay` | 0.3 | Lion weight decay coefficient |
| `--seed` | 987654321 | RNG seed for reproducibility |

## Output

```
03-texture-compression-with-input-encoding --output-image result.png
```

Saves the reconstructed texture as an RGB PNG. Default filename: `mlp-training-output.png`.

## Example Recipes

Reconstruct a photo with grid encoding:
```
03-texture-compression-with-input-encoding \
  --input-image photo.png \
  --input-encoding grid --grid-resolution 64 --grid-feature-dim 8 \
  --epochs 50 --optimizer adam --learning-rate 0.005 --loss-scale 512 \
  --output-image photo-reconstructed.png
```

High-quality procedural checkerboard with positional encoding:
```
03-texture-compression-with-input-encoding \
  --texture-pattern checkerboard --texture-width 1024 --texture-height 1024 \
  --input-encoding positional --positional-frequencies 8 \
  --backbone-layers 6 --hidden-dim 128 \
  --epochs 100 --optimizer adam --learning-rate 0.001 --loss-scale 128 \
  --output-image checkerboard-reconstructed.png
```

Fast smoke test (small texture, few epochs):
```
03-texture-compression-with-input-encoding \
  --texture-pattern gradient --texture-width 64 --texture-height 64 \
  --epochs 5 --batch-size 500 --samples 5000
```

## Other Flags

| Option | Description |
|---|---|
| `--cpp-fallback` | Use C++ fallback (mlp.hlsl compiled as C++) instead of GPU |
| `--software-linalg` | Use software fallback for linear algebra instead of LinAlg Matrix |
| `--debug` | Enable verbose debug output |
