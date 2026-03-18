#
# file: texture_reconstruction_mlp.py
# author: Sho Ikeda
# brief: MLP texture generation reference implementation
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

import torch
import numpy
import matplotlib.pyplot as plt
import argparse
from xoshiro128p import Xoshiro128Plus


def createCommandLineParser():
  """Create a command-line argument parser"""
  parser = argparse.ArgumentParser(description='Texture learning with MLP')
  parser.add_argument('--numHiddenLayers', type=int, default=3,
                      help='Number of hidden layers (default: 3)')
  parser.add_argument('--hiddenLayerDim', type=int, default=64,
                      help='Dimension of each hidden layer (default: 64)')
  parser.add_argument('--activation', type=str, default='leaky_relu',
                      choices=['identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu'],
                      help='Activation function (default: leaky_relu)')
  parser.add_argument('--seed', type=int, default=987654321,
                      help='Random seed for xoshiro128+ (default: 987654321)')
  parser.add_argument('--numSamples', type=int, default=10000,
                      help='Number of training samples (default: 10000)')
  parser.add_argument('--batchSize', type=int, default=10,
                      help='Batch size for training (default: 10)')
  parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs (default: 20)')
  parser.add_argument('--learningRate', type=float, default=0.01,
                      help='Learning rate for optimizer (default: 0.01)')
  parser.add_argument('--optimizer', type=str, default='sgd',
                      choices=['sgd', 'adam'],
                      help='Optimizer type (default: sgd)')
  parser.add_argument('--dtype', type=str, default='float',
                      choices=['float', 'half'],
                      help='Data type for MLP weights, biases, and computations (default: flaot)')
  parser.add_argument('--textureWidth', type=int, default=512,
                      help='Texture width resolution (default: 512)')
  parser.add_argument('--textureHeight', type=int, default=512,
                      help='Texture height resolution (default: 512)')
  parser.add_argument('--texturePattern', type=str, default='gradient',
                      choices=['gradient', 'checkerboard', 'stripes', 'circle', 'perlin'],
                      help='Texture pattern type (default: gradient)')
  return parser


def randn(rng: Xoshiro128Plus):
  """Generate random float from standard normal distribution using Box-Muller transform"""
  u1 = rng.draw()
  u2 = rng.draw()
  # Avoid log(0)
  u1 = max(u1, 1e-10)
  return numpy.sqrt(-2.0 * numpy.log(u1)) * numpy.cos(2.0 * numpy.pi * u2)


def randnArray(rng: Xoshiro128Plus, shape):
  """Generate array of random floats from standard normal distribution"""
  size = numpy.prod(shape)
  result = numpy.zeros(size)
  for i in range(size):
    result[i] = randn(rng)
  return result.reshape(shape)


class Mlp(torch.nn.Module):
  """MLP for texture learning with custom initialization"""

  def __init__(self, numHiddenLayers, hiddenLayerDim, activation, rng, dtype='half'):
    """
    Args:
      numHiddenLayers: Number of hidden layers (must be >= 1)
      hiddenLayerDim: Dimension of each hidden layer
      activation: Activation function name ('identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu')
      rng: Random number generator for weight initialization
      dtype: Data type for weights and computations ('float' or 'half')
    """
    super(Mlp, self).__init__()

    assert numHiddenLayers >= 1, "The number of hidden layers must be greater or equal to 1"

    self.m_numHiddenLayers = numHiddenLayers
    self.m_hiddenLayerDim = hiddenLayerDim
    self.m_activation_name = activation
    self.m_dtype = torch.float16 if dtype == 'half' else torch.float32

    # Create activation function (order matches ActivationType enum in C++)
    # IDENTITY = 0, SIGMOID = 1, TANH = 2, RELU = 3, LEAKY_RELU = 4
    if activation == 'identity':
      self.m_activation = torch.nn.Identity()
      self.m_activation_enum = 0
    elif activation == 'sigmoid':
      self.m_activation = torch.nn.Sigmoid()
      self.m_activation_enum = 1
    elif activation == 'tanh':
      self.m_activation = torch.nn.Tanh()
      self.m_activation_enum = 2
    elif activation == 'relu':
      self.m_activation = torch.nn.ReLU()
      self.m_activation_enum = 3
    elif activation == 'leaky_relu':
      self.m_activation = torch.nn.LeakyReLU()
      self.m_activation_enum = 4
    else:
      raise ValueError(f"Unsupported activation function: {activation}")

    # Define layers
    self.m_layers = torch.nn.ModuleList()

    # Input layer
    self.m_layers.append(torch.nn.Linear(2, hiddenLayerDim))

    # Hidden layers
    for i in range(numHiddenLayers - 1):
      self.m_layers.append(torch.nn.Linear(hiddenLayerDim, hiddenLayerDim))

    # Output layer
    self.m_fcOut = torch.nn.Linear(hiddenLayerDim, 2)

    # Initialize weights using He initialization with xoshiro128+
    for layer in self.m_layers:
      self.initHe(layer, rng)
    self.initHe(self.m_fcOut, rng)

    # Convert model to specified dtype
    self.to(self.m_dtype)

  def initHe(self, layer, rng: Xoshiro128Plus):
    """He initialization using xoshiro128+ random generator"""
    fanIn = layer.weight.shape[1]
    std = numpy.sqrt(2.0 / fanIn)

    # Generate weights
    weightShape = layer.weight.shape
    weights = randnArray(rng, weightShape) * std
    layer.weight.data = torch.from_numpy(weights).to(self.m_dtype)

    # Initialize bias to zero
    layer.bias.data.zero_()

  def forward(self, x):
    # Forward through hidden layers with activation
    for layer in self.m_layers:
      x = layer(x)
      x = self.m_activation(x)
    # Output layer with sigmoid
    x = self.m_fcOut(x)
    x = torch.sigmoid(x)
    return x

  def dumpWeightsBiases(self, prefix=""):
    """Debug function to dump weights and biases"""
    print(f"\n{prefix}=== MLP Weights and Biases ===")
    for name, param in self.named_parameters():
      print(f"\n{name}:")
      print(f"  Shape: {param.shape}")
      print(f"  Mean: {param.data.mean().item():.6f}")
      print(f"  Std: {param.data.std().item():.6f}")
      print(f"  Min: {param.data.min().item():.6f}")
      print(f"  Max: {param.data.max().item():.6f}")

  def dumpToBinary(self, filePath="texture-mlp-data.bin"):
    """Dump MLP configuration and parameters to a binary file.

    Format (all little-endian):
      int32: numHiddenLayers
      int32: hiddenLayerDim
      int32: activation (ActivationType enum value)
      float32[]: weights and biases for each layer in order
                 (all hidden layers, then output layer), flattened.
    """
    with open(filePath, "wb") as f:
      header = numpy.array([
        self.m_numHiddenLayers,
        self.m_hiddenLayerDim,
        self.m_activation_enum,
      ], dtype=numpy.int32)
      header.tofile(f)

      layers = list(self.m_layers) + [self.m_fcOut]
      for layer in layers:
        weight = layer.weight.detach().cpu().numpy().astype(numpy.float32).ravel()
        bias = layer.bias.detach().cpu().numpy().astype(numpy.float32).ravel()
        weight.tofile(f)
        bias.tofile(f)


def createGradientTexture(width=512, height=512):
  """Create 2-channel gradient texture (white center to black edges)"""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)
  centerX = width / 2.0
  centerY = height / 2.0
  maxDist = numpy.sqrt(centerX**2 + centerY**2)

  for i in range(height):
    for j in range(width):
      # Distance from center
      dist = numpy.sqrt((j - centerX)**2 + (i - centerY)**2)
      # Normalize to [0, 1], white (1) at center, black (0) at edges
      value = 1.0 - min(dist / maxDist, 1.0)
      texture[i, j, 0] = value
      texture[i, j, 1] = value

  return texture


def createCheckerboardTexture(width=512, height=512, squareSize=300):
  """Create 2-channel checkerboard texture"""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)

  for i in range(height):
    for j in range(width):
      # Checkerboard pattern
      checkX = (j // squareSize) % 2
      checkY = (i // squareSize) % 2
      value = float((checkX + checkY) % 2)
      texture[i, j, 0] = value
      texture[i, j, 1] = value

  return texture


def createStripesTexture(width=512, height=512, stripeWidth=160):
  """Create 2-channel horizontal stripes texture with gradient"""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)

  for i in range(height):
    for j in range(width):
      # Horizontal stripes with sinusoidal pattern
      value = 0.5 + 0.5 * numpy.sin(2 * numpy.pi * i / stripeWidth)
      texture[i, j, 0] = value
      texture[i, j, 1] = value

  return texture


def createCircleTexture(width=512, height=512):
  """Create 2-channel concentric circles texture"""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)
  centerX = width / 2.0
  centerY = height / 2.0
  maxDist = numpy.sqrt(centerX**2 + centerY**2)

  for i in range(height):
    for j in range(width):
      # Distance from center
      dist = numpy.sqrt((j - centerX)**2 + (i - centerY)**2)
      # Concentric circles with sinusoidal pattern
      normalizedDist = dist / maxDist
      value = 0.5 + 0.5 * numpy.sin(5 * numpy.pi * normalizedDist)
      texture[i, j, 0] = value
      texture[i, j, 1] = value

  return texture


def perlinNoise(x, y, seed=0):
  """Simple 2D Perlin-like noise (simplified version)"""
  # Simple hash-based noise for demonstration
  n = int(x * 57 + y * 131 + seed * 13)
  n = (n << 13) ^ n
  return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)


def createPerlinTexture(width=512, height=512, scale=0.05):
  """Create 2-channel Perlin noise texture"""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)

  for i in range(height):
    for j in range(width):
      # Simplified Perlin-like noise
      noise = perlinNoise(j * scale, i * scale, seed=12345)
      # Normalize to [0, 1]
      value = (noise + 1.0) * 0.5
      texture[i, j, 0] = value
      texture[i, j, 1] = value

  return texture


def createTexture(pattern='gradient', width=512, height=512):
  """Create texture based on pattern type"""
  if pattern == 'gradient':
    return createGradientTexture(width, height)
  elif pattern == 'checkerboard':
    return createCheckerboardTexture(width, height)
  elif pattern == 'stripes':
    return createStripesTexture(width, height)
  elif pattern == 'circle':
    return createCircleTexture(width, height)
  elif pattern == 'perlin':
    return createPerlinTexture(width, height)
  else:
    raise ValueError(f"Unsupported texture pattern: {pattern}")


def generateTrainingData(texture, numSamples, rng):
  """Generate UV coordinates and corresponding texel values"""
  height = texture.shape[0]
  width = texture.shape[1]

  # Random UV coordinates in [0, 1] using rng
  uv = numpy.zeros((numSamples, 2), dtype=numpy.float32)
  for i in range(numSamples):
    uv[i, 0] = rng.draw()
    uv[i, 1] = rng.draw()

  # Convert UV to pixel coordinates
  pixelX = (uv[:, 0] * (width - 1)).astype(numpy.int32)
  pixelY = (uv[:, 1] * (height - 1)).astype(numpy.int32)

  # Get corresponding texel values (2D)
  texels = texture[pixelY, pixelX, :]

  return uv, texels


def trainModel(model, uvTrain, texelsTrain, epochs=20, batchSize=10, learningRate=0.01, optimizerType='sgd'):
  """Train the MLP model"""
  criterion = torch.nn.MSELoss()
  if optimizerType == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
  else:
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

  # Convert training data to model's dtype
  uvTensor = torch.from_numpy(uvTrain).to(model.m_dtype)
  texelsTensor = torch.from_numpy(texelsTrain).to(model.m_dtype)
  
  dataset = torch.utils.data.TensorDataset(uvTensor, texelsTensor)
  dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batchSize, shuffle=False
  )

  print("Starting training...")
  for epoch in range(epochs):
    epochLoss = 0.0
    numBatches = 0

    for batchUv, batchTexels in dataloader:
      optimizer.zero_grad()
      outputs = model(batchUv)
      loss = criterion(outputs, batchTexels)
      loss.backward()
      optimizer.step()

      epochLoss += loss.item()
      numBatches += 1

    avgLoss = epochLoss / numBatches
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avgLoss:.6f}")

  print("Training completed!")


def reconstructTexture(model, width=512, height=512):
  """Reconstruct texture using trained MLP"""
  texture = numpy.zeros((height, width, 2), dtype=numpy.float32)

  with torch.no_grad():
    for i in range(height):
      for j in range(width):
        u = j / (width - 1)
        v = i / (height - 1)
        uv = torch.tensor([[u, v]], dtype=model.m_dtype)
        texel = model(uv).cpu().float().numpy()[0]
        texture[i, j, :] = texel

  return texture


def main():
  # Parse command-line arguments
  parser = createCommandLineParser()
  args = parser.parse_args()

  # Initialize xoshiro128+ random generator
  rng = Xoshiro128Plus(seed=args.seed)

  # Create original texture
  print(f"Creating {args.texturePattern} texture (resolution: {args.textureWidth}x{args.textureHeight})...")
  originalTexture = createTexture(pattern=args.texturePattern, width=args.textureWidth, height=args.textureHeight)

  # Create and initialize model
  print("Initializing MLP...")
  print(f"Configuration: numHiddenLayers={args.numHiddenLayers}, hiddenLayerDim={args.hiddenLayerDim}, activation={args.activation}, dtype={args.dtype}")
  model = Mlp(numHiddenLayers=args.numHiddenLayers, hiddenLayerDim=args.hiddenLayerDim, activation=args.activation, rng=rng, dtype=args.dtype)

  # Dump initial weights
  model.dumpWeightsBiases(prefix="Initial ")

  # Generate training data
  print("Generating training data...")
  print(f"numSamples={args.numSamples}")
  uvTrain, texelsTrain = generateTrainingData(originalTexture, numSamples=args.numSamples, rng=rng)

  # Train model
  print(f"Training: epochs={args.epochs}, batchSize={args.batchSize}, learningRate={args.learningRate}, optimizer={args.optimizer}")
  trainModel(model, uvTrain, texelsTrain, epochs=args.epochs, batchSize=args.batchSize, learningRate=args.learningRate, optimizerType=args.optimizer)

  # Dump final weights
  model.dumpWeightsBiases(prefix="Final ")
  model.dumpToBinary()

  # Reconstruct texture
  print("Reconstructing texture...")
  reconstructedTexture = reconstructTexture(model, width=args.textureWidth, height=args.textureHeight)

  # Display results
  print("Displaying results...")
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))

  # Show first channel of 2D texel
  axes[0].imshow(originalTexture[:, :, 0], cmap='gray', vmin=0, vmax=1)
  axes[0].set_title(f'Original Texture ({args.texturePattern})')
  axes[0].axis('off')

  axes[1].imshow(reconstructedTexture[:, :, 0], cmap='gray', vmin=0, vmax=1)
  axes[1].set_title('MLP Reconstructed Texture')
  axes[1].axis('off')

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  main()
