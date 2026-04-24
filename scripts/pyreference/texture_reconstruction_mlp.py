#
# file: texture_reconstruction_mlp.py
# author: Sho Ikeda
# brief: MLP texture generation reference implementation
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

import argparse
import time

import numpy
import torch
import matplotlib.pyplot as plt

from xoshiro128p import Xoshiro128Plus


def createCommandLineParser():
  """Create a command-line argument parser."""
  parser = argparse.ArgumentParser(description='Texture learning with MLP')
  # MLP architecture
  parser.add_argument('--backbone-layers', type=int, default=4,
                      help='Number of backbone layers (default: 4)')
  parser.add_argument('--hidden-dim', type=int, default=64,
                      help='Dimension of each hidden layer (default: 64)')
  parser.add_argument('--activation', type=str, default='leaky_relu',
                      choices=['identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu'],
                      help='Activation function (default: leaky_relu)')
  parser.add_argument('--dtype', type=str, default='float',
                      choices=['float', 'half'],
                      help='Data type for MLP weights and computations (default: float)')
  # Training parameters
  parser.add_argument('--seed', type=int, default=987654321,
                      help='Random seed for xoshiro128+ (default: 987654321)')
  parser.add_argument('--samples', type=int, default=200000,
                      help='Number of training samples (default: 200000)')
  parser.add_argument('--batch-size', type=int, default=2000,
                      help='Batch size for training (default: 2000)')
  parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs (default: 30)')
  parser.add_argument('--learning-rate', type=float, default=0.0025,
                      help='Learning rate for optimizer (default: 0.0025)')
  parser.add_argument('--optimizer', type=str, default='lion',
                      choices=['sgd', 'adam', 'lion'],
                      help='Optimizer type (default: lion)')
  # Texture parameters
  parser.add_argument('--texture-width', type=int, default=2048,
                      help='Texture width resolution (default: 2048)')
  parser.add_argument('--texture-height', type=int, default=2048,
                      help='Texture height resolution (default: 2048)')
  parser.add_argument('--texture-pattern', type=str, default='checkerboard',
                      choices=['gradient', 'checkerboard', 'stripes', 'circle', 'perlin'],
                      help='Texture pattern type (default: checkerboard)')
  # Display
  parser.add_argument('--no-display', action='store_true',
                      help='Skip matplotlib display and exit automatically')
  return parser


# ============================================================================
# Random number generation
# ============================================================================

def randn(rng):
  """Generate a standard normal random float using the Box-Muller transform."""
  u1 = max(rng.draw(), 1e-10)
  u2 = rng.draw()
  return numpy.sqrt(-2.0 * numpy.log(u1)) * numpy.cos(2.0 * numpy.pi * u2)


def randnArray(rng, shape):
  """Generate an array of standard normal random floats."""
  size = numpy.prod(shape)
  result = numpy.zeros(size)
  for i in range(size):
    result[i] = randn(rng)
  return result.reshape(shape)


# ============================================================================
# MLP model
# ============================================================================

# Activation type enum values (matches ActivationType in C++)
ACTIVATION_MAP = {
  'identity':   (torch.nn.Identity,  0),
  'sigmoid':    (torch.nn.Sigmoid,   1),
  'tanh':       (torch.nn.Tanh,      2),
  'relu':       (torch.nn.ReLU,      3),
  'leaky_relu': (torch.nn.LeakyReLU, 4),
}


class Mlp(torch.nn.Module):
  """MLP for texture learning: maps (u,v) -> 2-channel texel values."""

  def __init__(self, numBackboneLayers, hiddenLayerDim, activation, rng, dtype):
    super().__init__()
    assert numBackboneLayers >= 1, "The number of backbone layers must be >= 1"

    self.m_numBackboneLayers = numBackboneLayers
    self.m_hiddenLayerDim = hiddenLayerDim
    self.m_dtype = torch.float16 if dtype == 'half' else torch.float32

    if activation not in ACTIVATION_MAP:
      raise ValueError(f"Unsupported activation function: {activation}")
    activationClass, self.m_activation_enum = ACTIVATION_MAP[activation]
    self.m_activation = activationClass()

    # Build layers: input -> backbone -> output
    self.m_layers = torch.nn.ModuleList()
    self.m_layers.append(torch.nn.Linear(2, hiddenLayerDim))
    for _ in range(numBackboneLayers - 1):
      self.m_layers.append(torch.nn.Linear(hiddenLayerDim, hiddenLayerDim))
    self.m_fcOut = torch.nn.Linear(hiddenLayerDim, 2)

    # He initialization with xoshiro128+
    for layer in self.m_layers:
      self._initHe(layer, rng)
    self._initHe(self.m_fcOut, rng)

    self.to(self.m_dtype)

  def _initHe(self, layer, rng):
    """He/Kaiming normal initialization using xoshiro128+."""
    fanIn = layer.weight.shape[1]
    std = numpy.sqrt(2.0 / fanIn)
    weights = randnArray(rng, layer.weight.shape) * std
    layer.weight.data = torch.from_numpy(weights).to(self.m_dtype)
    layer.bias.data.zero_()

  def forward(self, x):
    for layer in self.m_layers:
      x = self.m_activation(layer(x))
    return torch.sigmoid(self.m_fcOut(x))

  def dumpWeightsBiases(self, prefix=""):
    """Print weight/bias statistics for debugging."""
    print(f"\n{prefix}=== MLP Weights and Biases ===")
    for name, param in self.named_parameters():
      data = param.data
      print(f"\n{name}:")
      print(f"  Shape: {param.shape}")
      print(f"  Mean: {data.mean().item():.6f}")
      print(f"  Std: {data.std().item():.6f}")
      print(f"  Min: {data.min().item():.6f}")
      print(f"  Max: {data.max().item():.6f}")

  def dumpToBinary(self, filePath):
    """Dump MLP parameters to a binary file.

    Format (all little-endian):
      int32: numBackboneLayers
      int32: hiddenLayerDim
      int32: activation enum value
      float32[]: weights and biases for each layer (backbone then output), flattened
    """
    with open(filePath, "wb") as f:
      header = numpy.array([
        self.m_numBackboneLayers,
        self.m_hiddenLayerDim,
        self.m_activation_enum,
      ], dtype=numpy.int32)
      header.tofile(f)

      for layer in list(self.m_layers) + [self.m_fcOut]:
        layer.weight.detach().cpu().numpy().astype(numpy.float32).ravel().tofile(f)
        layer.bias.detach().cpu().numpy().astype(numpy.float32).ravel().tofile(f)


# ============================================================================
# Texture generation
# ============================================================================

def createGradientTexture(width, height):
  """Create 2-channel radial gradient texture (white center, black edges)."""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)
  centerX, centerY = width / 2.0, height / 2.0
  maxDist = numpy.sqrt(centerX**2 + centerY**2)
  for i in range(height):
    for j in range(width):
      dist = numpy.sqrt((j - centerX)**2 + (i - centerY)**2)
      value = 1.0 - min(dist / maxDist, 1.0)
      texture[i, j, :] = value
  return texture


def createCheckerboardTexture(width, height):
  """Create 2-channel checkerboard texture."""
  squareSize = 300
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)
  for i in range(height):
    for j in range(width):
      value = float(((j // squareSize) + (i // squareSize)) % 2)
      texture[i, j, :] = value
  return texture


def createStripesTexture(width, height):
  """Create 2-channel horizontal stripes texture with sinusoidal pattern."""
  stripeWidth = 160
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)
  for i in range(height):
    value = 0.5 + 0.5 * numpy.sin(2 * numpy.pi * i / stripeWidth)
    texture[i, :, :] = value
  return texture


def createCircleTexture(width, height):
  """Create 2-channel concentric circles texture."""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)
  centerX, centerY = width / 2.0, height / 2.0
  maxDist = numpy.sqrt(centerX**2 + centerY**2)
  for i in range(height):
    for j in range(width):
      dist = numpy.sqrt((j - centerX)**2 + (i - centerY)**2)
      value = 0.5 + 0.5 * numpy.sin(5 * numpy.pi * dist / maxDist)
      texture[i, j, :] = value
  return texture


def perlinNoise(x, y, seed):
  """Simple 2D hash-based noise."""
  n = int(x * 57 + y * 131 + seed * 13)
  n = (n << 13) ^ n
  return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)


def createPerlinTexture(width, height):
  """Create 2-channel Perlin-like noise texture."""
  scale = 0.01
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)
  for i in range(height):
    for j in range(width):
      value = (perlinNoise(j * scale, i * scale, 12345) + 1.0) * 0.5
      texture[i, j, :] = value
  return texture


def createTexture(pattern, width, height):
  """Create a 2-channel texture based on pattern type."""
  creators = {
    'gradient':     createGradientTexture,
    'checkerboard': createCheckerboardTexture,
    'stripes':      createStripesTexture,
    'circle':       createCircleTexture,
    'perlin':       createPerlinTexture,
  }
  if pattern not in creators:
    raise ValueError(f"Unsupported texture pattern: {pattern}")
  return creators[pattern](width, height)


# ============================================================================
# Training data generation
# ============================================================================

def generateTrainingData(texture, numSamples, rng):
  """Generate random UV coordinates and their corresponding texel values."""
  height, width = texture.shape[0], texture.shape[1]

  uv = numpy.zeros((numSamples, 2), dtype=numpy.float32)
  for i in range(numSamples):
    uv[i, 0] = rng.draw()
    uv[i, 1] = rng.draw()

  pixelX = (uv[:, 0] * (width - 1)).astype(numpy.int32)
  pixelY = (uv[:, 1] * (height - 1)).astype(numpy.int32)
  texels = texture[pixelY, pixelX, :]

  return uv, texels


# ============================================================================
# Optimizer
# ============================================================================

class LionOptimizer(torch.optim.Optimizer):
  """Lion optimizer (Chen et al., 2023)."""

  def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99):
    super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2))

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      lr, beta1, beta2 = group['lr'], group['beta1'], group['beta2']
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad
        state = self.state[p]
        if len(state) == 0:
          state['momentum'] = torch.zeros_like(p)
        m = state['momentum']
        p.data.sub_((beta1 * m + (1.0 - beta1) * grad).sign(), alpha=lr)
        m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
    return loss


# ============================================================================
# Training and reconstruction
# ============================================================================

def createOptimizer(model, optimizerType, learningRate):
  """Create an optimizer for the given model."""
  if optimizerType == 'adam':
    return torch.optim.Adam(model.parameters(), lr=learningRate)
  elif optimizerType == 'lion':
    return LionOptimizer(model.parameters(), lr=learningRate)
  else:
    return torch.optim.SGD(model.parameters(), lr=learningRate)


def trainModel(model, uvTrain, texelsTrain, epochs, batchSize, learningRate, optimizerType):
  """Train the MLP model."""
  criterion = torch.nn.MSELoss()
  optimizer = createOptimizer(model, optimizerType, learningRate)

  uvTensor = torch.from_numpy(uvTrain).to(model.m_dtype)
  texelsTensor = torch.from_numpy(texelsTrain).to(model.m_dtype)
  dataset = torch.utils.data.TensorDataset(uvTensor, texelsTensor)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)

  print("Starting training...")
  for epoch in range(epochs):
    epochLoss = 0.0
    numBatches = 0
    for batchUv, batchTexels in dataloader:
      optimizer.zero_grad()
      loss = criterion(model(batchUv), batchTexels)
      loss.backward()
      optimizer.step()
      epochLoss += loss.item()
      numBatches += 1
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epochLoss / numBatches:.6f}")
  print("Training completed!")


def reconstructTexture(model, width, height):
  """Reconstruct a texture by evaluating the trained MLP at every pixel."""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)
  with torch.no_grad():
    for i in range(height):
      for j in range(width):
        uv = torch.tensor([[j / (width - 1), i / (height - 1)]], dtype=model.m_dtype)
        texture[i, j, :] = model(uv).cpu().float().numpy()[0]
  return texture


# ============================================================================
# Main
# ============================================================================

def main():
  parser = createCommandLineParser()
  args = parser.parse_args()

  rng = Xoshiro128Plus(seed=args.seed)

  print(f"Creating {args.texture_pattern} texture ({args.texture_width}x{args.texture_height})...")
  originalTexture = createTexture(args.texture_pattern, args.texture_width, args.texture_height)

  print(f"Initializing MLP: backbone={args.backbone_layers}, hidden={args.hidden_dim}, "
        f"activation={args.activation}, dtype={args.dtype}")
  model = Mlp(args.backbone_layers, args.hidden_dim, args.activation, rng, args.dtype)
  model.dumpWeightsBiases(prefix="Initial ")

  print(f"Generating {args.samples} training samples...")
  uvTrain, texelsTrain = generateTrainingData(originalTexture, args.samples, rng)

  print(f"Training: epochs={args.epochs}, batch={args.batch_size}, "
        f"lr={args.learning_rate}, optimizer={args.optimizer}")
  print("Backend: Python (reference)")
  totalStart = time.perf_counter()
  trainingStart = time.perf_counter()
  trainModel(model, uvTrain, texelsTrain, args.epochs, args.batch_size,
             args.learning_rate, args.optimizer)
  trainingMs = (time.perf_counter() - trainingStart) * 1000.0
  print(f"Training time: {trainingMs:.3f} ms")

  model.dumpWeightsBiases(prefix="Final ")
  model.dumpToBinary("texture-mlp-data.bin")

  print("Reconstructing texture...")
  reconstructStart = time.perf_counter()
  reconstructedTexture = reconstructTexture(model, args.texture_width, args.texture_height)
  reconstructMs = (time.perf_counter() - reconstructStart) * 1000.0
  print(f"Reconstruction time: {reconstructMs:.3f} ms")
  totalMs = (time.perf_counter() - totalStart) * 1000.0
  print(f"Total time: {totalMs:.3f} ms")

  if not args.no_display:
    print("Displaying results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(originalTexture[:, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original Texture ({args.texture_pattern})')
    axes[0].axis('off')
    axes[1].imshow(reconstructedTexture[:, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('MLP Reconstructed Texture')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
  else:
    print("Skipping display (--no-display).")


if __name__ == "__main__":
  main()
