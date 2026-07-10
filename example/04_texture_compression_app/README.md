# 04 — Texture Compression App

Real-time windowed application that trains an MLP to compress a 2D texture on the GPU. Training runs incrementally (a configurable number of epochs per frame), and the reconstructed texture is displayed live as a fullscreen quad with an ImGui overlay for monitoring and parameter control.

Based on [Example 03](../03_texture_compression_with_input_encoding/) but restructured for interactive use.

## Building

```bash
cmake --build build --target 04-texture-compression-app --config Release
```

Requires `GFX_ENABLE_GUI ON` in `third_party/gfx_dep/CMakeLists.txt` for ImGui support.

## Usage

```
04-texture-compression-app [options]
```

### Examples

```bash
# Default: train on a checkerboard pattern
04-texture-compression-app

# Train on a PNG image
04-texture-compression-app --input-image photo.png

# Faster training: 5 epochs per frame with larger batch
04-texture-compression-app --input-image photo.png --epochs-per-frame 5 --batch-size 50000

# Positional encoding with more frequency bands
04-texture-compression-app --input-encoding positional --positional-frequencies 8

# Identity (raw UV) encoding
04-texture-compression-app --input-encoding none
```

### Command-Line Options

| Option | Default | Description |
|---|---|---|
| `--backbone-layers` | 4 | Number of MLP backbone layers |
| `--hidden-dim` | 64 | Hidden layer dimension |
| `--activation` | leaky_relu | Activation: identity, sigmoid, tanh, relu, leaky_relu |
| `--bias` / `--no-bias` | true | Enable/disable bias in MLP layers |
| `--seed` | 987654321 | Random seed |
| `--samples` | 200000 | Number of training samples |
| `--batch-size` | 20000 | Training batch size (adjustable at runtime via ImGui) |
| `--learning-rate` | 0.005 | Optimizer learning rate |
| `--optimizer` | adam | Optimizer: sgd, adam, lion |
| `--input-encoding` | positional | Input encoding: none, positional |
| `--positional-frequencies` | 4 | Frequency bands for positional encoding |
| `--input-image` | — | PNG image to use as ground truth |
| `--texture-width` | 2048 | Generated texture width (ignored with `--input-image`) |
| `--texture-height` | 2048 | Generated texture height (ignored with `--input-image`) |
| `--texture-pattern` | checkerboard | Pattern: gradient, checkerboard, stripes, circle, perlin |
| `--epochs-per-frame` | 1 | Training epochs per frame |
| `--window-width` | 1024 | Window width |
| `--window-height` | 1024 | Window height |
| `--loss-scale` | 512 | Loss scale for FP16 gradient stability |
| `--software-linalg` | false | Use software linear algebra on HLSL |
| `--debug` | false | Enable GPU debug/shader debugging |

## ImGui Controls

- **Start/Pause Training** — training starts paused; click to begin
- **Encoding** — switch between None (identity) and Positional encoding (requires reinit)
- **Frequencies** — positional encoding frequency count (shown when Positional is selected)
- **Batch Size** — adjustable at any time, takes effect immediately
- **Loss graph** — log-scale loss plotted over epochs
- **Stats** — current epoch, loss, training time, inference time, FPS

## Key Differences from Example 03

- Windowed with live display instead of headless
- GPU-only (no C++ fallback path)
- Incremental training across frames
- Interactive parameter tuning via ImGui
- No output PNG — display is live on screen
