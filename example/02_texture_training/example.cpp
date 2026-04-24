/*!
  \file example.cpp
  \author Sho Ikeda
  \brief Texture MLP training example
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This example demonstrates how to train an MLP to reconstruct 2D texture patterns.

  Overview:
    1. Create a ground-truth texture pattern (gradient, checkerboard, etc.)
    2. Initialize an MLP with He/Kaiming weight initialization
    3. Generate random (u,v) training samples from the texture
    4. Train the MLP using mini-batch SGD/Adam/Lion optimization
    5. Reconstruct the texture by evaluating the trained MLP at every pixel
    6. Save the result as a PPM image

  The MLP learns to map 2D coordinates (u,v) -> pixel intensity, effectively
  compressing a texture into a compact neural network representation.

  Usage:
    02-texture-training
        [--backbone-layers N] [--hidden-dim N] [--activation TYPE]
        [--epochs N] [--batch-size N] [--learning-rate F] [--optimizer TYPE]
        [--texture-width N] [--texture-height N] [--texture-pattern TYPE]
        [--output-image FILE]
        [--cpu] [--software-linalg] [--debug] [--seed N]
*/

// Standard C++ library
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <chrono>
// Half
#include "half.hpp"
// CLI
#include "CLI/CLI.hpp"
// Example
#include "hlsl_include_dirs.hpp"
#include "common/activation.hpp"
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
#include "common/gfx_utility.hpp"
#endif
#include "common/image.hpp"
#include "common/loss.hpp"
#include "common/matrix.hpp"
#include "common/optimizer.hpp"
#include "common/pixmap.hpp"
#include "common/texture.hpp"
#include "common/xoshiro128plus.hpp"
// C++ fallback infrastructure (includes hlsl_compat.hpp, mlp.hlsl, utility, mlp_layer)
#include "common/cpp_fallback.hpp"
#include "kernel/texture_training_common.hlsl"
#include "kernel/texture_inference_common.hlsl"
#include "kernel/optimizer.hlsl"

namespace {

// ============================================================================
// Command-line options
// ============================================================================

struct CliOptions
{
  size_t m_numBackboneLayers = 4;
  size_t m_hiddenLayerDim = 64;
  std::string m_activation = "leaky_relu";
  bool m_hasBias = true;
  std::uint32_t m_seed = 987654321;
  size_t m_numSamples = 200000;
  size_t m_batchSize = 2000;
  size_t m_epochs = 30;
  double m_learningRate = 0.0025;
  std::string m_optimizer = "lion";
  size_t m_textureWidth = 2048;
  size_t m_textureHeight = 2048;
  std::string m_texturePattern = "checkerboard";
  std::string m_outputImagePath = "mlp-training-output.ppm";
  bool m_useCpuMlpOperations = false;
  bool m_useCppFallback = false;
  bool m_useSoftwareLinalg = false;
  bool m_enableDebugMode = true;
};

auto createCommandLineParser(CliOptions& options) -> std::unique_ptr<CLI::App>
{
  auto parser = std::make_unique<CLI::App>(
      "Texture MLP training - Train MLP to reconstruct texture patterns");

  // MLP architecture
  parser->add_option("--backbone-layers", options.m_numBackboneLayers,
      "Number of backbone layers (default: 4)")
      ->default_val(options.m_numBackboneLayers)
      ->check(CLI::PositiveNumber);
  parser->add_option("--hidden-dim", options.m_hiddenLayerDim,
      "Dimension of each hidden layer (default: 64)")
      ->default_val(options.m_hiddenLayerDim)
      ->check(CLI::PositiveNumber);
  parser->add_option("--activation", options.m_activation,
      "Activation function (identity, sigmoid, tanh, relu, leaky_relu)")
      ->default_val(options.m_activation)
      ->check(CLI::IsMember({"identity", "sigmoid", "tanh", "relu", "leaky_relu"}));
  parser->add_flag("--bias,!--no-bias", options.m_hasBias,
      "Use bias in MLP layers (default: true, use --no-bias to disable)")
      ->default_val(options.m_hasBias);

  // Training parameters
  parser->add_option("--seed", options.m_seed,
      "Random seed for xoshiro128+ (default: 987654321)")
      ->default_val(options.m_seed);
  parser->add_option("--samples", options.m_numSamples,
      "Number of training samples (default: 200000)")
      ->default_val(options.m_numSamples)
      ->check(CLI::PositiveNumber);
  parser->add_option("--batch-size", options.m_batchSize,
      "Batch size for training (default: 2000)")
      ->default_val(options.m_batchSize)
      ->check(CLI::PositiveNumber);
  parser->add_option("--epochs", options.m_epochs,
      "Number of training epochs (default: 30)")
      ->default_val(options.m_epochs)
      ->check(CLI::PositiveNumber);
  parser->add_option("--learning-rate", options.m_learningRate,
      "Learning rate for optimizer (default: 0.0025)")
      ->default_val(options.m_learningRate)
      ->check(CLI::PositiveNumber);
  parser->add_option("--optimizer", options.m_optimizer,
      "Optimizer type: sgd, adam, lion (default: lion)")
      ->default_val(options.m_optimizer)
      ->check(CLI::IsMember({"sgd", "adam", "lion"}));

  // Texture parameters
  parser->add_option("--texture-width", options.m_textureWidth,
      "Texture width resolution (default: 2048)")
      ->default_val(options.m_textureWidth)
      ->check(CLI::PositiveNumber);
  parser->add_option("--texture-height", options.m_textureHeight,
      "Texture height resolution (default: 2048)")
      ->default_val(options.m_textureHeight)
      ->check(CLI::PositiveNumber);
  parser->add_option("--texture-pattern", options.m_texturePattern,
      "Texture pattern type (gradient, checkerboard, stripes, circle, perlin)")
      ->default_val(options.m_texturePattern)
      ->check(CLI::IsMember({"gradient", "checkerboard", "stripes", "circle", "perlin"}));

  // Output
  parser->add_option("--output-image", options.m_outputImagePath,
      "Output reconstructed image path in PPM format (default: mlp-training-output.ppm)")
      ->default_val(options.m_outputImagePath);

  // Execution mode
  parser->add_flag("--cpu", options.m_useCpuMlpOperations,
      "Use CPU reference ML operations instead of GPU")
      ->default_val(options.m_useCpuMlpOperations);
  parser->add_flag("--cpp-fallback", options.m_useCppFallback,
      "Use C++ fallback (mlp.hlsl compiled as C++)")
      ->default_val(options.m_useCppFallback);
  parser->add_flag("--software-linalg", options.m_useSoftwareLinalg,
      "Use software-implementation linear algebra functions on HLSL")
      ->default_val(options.m_useSoftwareLinalg);
  parser->add_flag("--debug", options.m_enableDebugMode,
      "Enable debug mode for detailed output")
      ->default_val(options.m_enableDebugMode);

  return parser;
}

// ============================================================================
// MLP configuration
// ============================================================================

template <ex::Arithmetic Type>
struct MlpConfig
{
  std::uint32_t m_numBackboneLayers;
  std::uint32_t m_hiddenLayerDim;
  ex::ActivationType m_activation;
  bool m_hasBias;
  std::vector<ex::MlpLayer<Type, Type, Type, Type>> m_layers;
};

// ============================================================================
// MLP initialization
// ============================================================================

/*!
  \brief Create and initialize an MLP based on command-line options.

  Builds the MLP layer stack from the CLI configuration:
    Layer 0:       input(2) -> hidden(hiddenLayerDim)   [user-specified activation]
    Layer 1..N-1:  hidden   -> hidden                   [user-specified activation]
    Layer N:       hidden   -> output(2)                [Sigmoid — maps output to [0,1]]

  Weights are initialized using He/Kaiming normal initialization.
  Biases are initialized to zero.

  \param options  CLI options specifying architecture (backbone layers, hidden dim, activation)
  \param rng      Random number generator for weight initialization
  \return Vector of initialized MLP layers ready for training
*/
template <ex::Arithmetic DataT>
auto initializeMlp(const CliOptions& options, ex::Xoshiro128Plus& rng)
    -> MlpConfig<DataT>
{
  const auto activationType = ex::getActivationTypeFromString(options.m_activation);

  // Build layer configurations: input -> hidden layers -> output
  std::vector<ex::LayerConfiguration> configs;
  configs.push_back({2, options.m_hiddenLayerDim, activationType});
  for (size_t i = 1; i < options.m_numBackboneLayers; ++i) {
    configs.push_back({options.m_hiddenLayerDim, options.m_hiddenLayerDim, activationType});
  }
  configs.push_back({options.m_hiddenLayerDim, 2, ex::ActivationType::SIGMOID});

  MlpConfig<DataT> config;
  config.m_numBackboneLayers = static_cast<std::uint32_t>(options.m_numBackboneLayers);
  config.m_hiddenLayerDim = static_cast<std::uint32_t>(options.m_hiddenLayerDim);
  config.m_activation = activationType;
  config.m_hasBias = options.m_hasBias;
  // Initialize weights (He/Kaiming normal), biases = 0
  config.m_layers = ex::createMlp<DataT, DataT, DataT, DataT>(configs, false, rng);

  return config;
}

// ============================================================================
// Training data generation
// ============================================================================

/*!
  \brief Generate training data by randomly sampling UV coordinates from the texture.

  Draws random (u,v) coordinates and looks up the corresponding texel values
  from the ground-truth texture. These pairs form the training dataset:
    input: (u, v)  ->  target: texture(u, v)

  \param texture     Ground-truth texture to sample from
  \param numSamples  Number of (u,v) samples to generate
  \param rng         Random number generator for uniform sampling
  \return Pair of vectors: (uvData, texelData), each containing 2 * numSamples elements
*/
template <ex::Arithmetic DataT>
auto generateTrainingData(const ex::Texture2Ch& texture,
                          const size_t numSamples,
                          ex::Xoshiro128Plus& rng)
    -> std::pair<std::vector<DataT>, std::vector<DataT>>
{
  std::vector<DataT> uvData(numSamples * 2);
  std::vector<DataT> texelData(numSamples * 2);

  for (size_t i = 0; i < numSamples; ++i) {
    const float u = rng.draw();
    const float v = rng.draw();
    uvData[i * 2 + 0] = static_cast<DataT>(u);
    uvData[i * 2 + 1] = static_cast<DataT>(v);

    const auto texel = texture.sample(u, v);
    texelData[i * 2 + 0] = static_cast<DataT>(texel[0]);
    texelData[i * 2 + 1] = static_cast<DataT>(texel[1]);
  }

  return {std::move(uvData), std::move(texelData)};
}

// ============================================================================
// UV coordinate generation
// ============================================================================

/*!
  \brief Generate normalized UV coordinates for every pixel in the texture.

  Creates a flat array of interleaved (u, v) pairs where u,v in [0,1], ordered
  row-by-row from top-left to bottom-right.

  \return Vector of size (width * height * 2) containing [u0, v0, u1, v1, ...].
*/
template <ex::Arithmetic Type>
auto createUvData(const size_t width, const size_t height) -> std::vector<Type>
{
  const size_t numPixels = width * height;
  std::vector<Type> uvData;
  uvData.reserve(numPixels * 2);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      const float u = static_cast<float>(j) / static_cast<float>(width - 1);
      const float v = static_cast<float>(i) / static_cast<float>(height - 1);
      uvData.push_back(static_cast<Type>(u));
      uvData.push_back(static_cast<Type>(v));
    }
  }

  return uvData;
}

// ============================================================================
// HDR to LDR conversion
// ============================================================================

/*!
  \brief Convert floating-point MLP output to 8-bit grayscale.

  The MLP outputs 2 channels per pixel. This function extracts the first channel,
  clamps it to [0,1], and quantizes to [0,255] for image output.

  \param hdr  MLP output (2 values per pixel: [ch0, ch1, ch0, ch1, ...])
  \param ldr  Target pixmap for 8-bit grayscale output
*/
template <ex::Arithmetic Type>
auto mapToLdr(const std::span<const Type> hdr, ex::PixmapU8& ldr) noexcept
{
  std::span out = ldr.data();
  const size_t numPixels = ldr.width() * ldr.height();
  for (size_t i = 0; i < numPixels; ++i) {
    using half_float::round;
    using std::round;
    using std::clamp;
    Type x = hdr[2 * i];
    x = clamp(x, static_cast<Type>(0), static_cast<Type>(1));
    x = round(x * static_cast<Type>(255));
    out[i] = {{static_cast<std::uint8_t>(x)}};
  }
}

// ============================================================================
// Training and texture reconstruction
// ============================================================================

/*!
  \brief Train the MLP on CPU and reconstruct the texture.

  Performs mini-batch stochastic gradient descent training:
    1. For each epoch, iterate over training data in batches
    2. For each batch: forward pass -> MSE loss -> backward pass -> optimizer step
    3. After training completes, evaluate the trained MLP at every pixel

  \param mlpData    MLP layers (modified in-place during training)
  \param uvData     Training UV coordinates (2 * numSamples elements)
  \param texelData  Ground-truth texel values (2 * numSamples elements)
  \param options    Training hyperparameters and output configuration
  \return Reconstructed texture as an 8-bit grayscale pixmap
*/
template <ex::Arithmetic DataT>
auto trainAndReconstructTextureCpu(
    std::span<ex::MlpLayer<DataT, DataT, DataT, DataT>> mlpData,
    const std::vector<DataT>& uvData,
    const std::vector<DataT>& texelData,
    const bool hasBias,
    const CliOptions& options) -> ex::PixmapU8
{
  const size_t numSamples = uvData.size() / 2;
  const size_t numLayers = mlpData.size();
  constexpr size_t outputDim = 2;
  const float lr = static_cast<float>(options.m_learningRate);

  // Create optimizer (SGD, Adam, or Lion)
  const auto optimizerType = ex::getOptimizerTypeFromString(options.m_optimizer);
  auto optimizer = ex::createOptimizer<DataT, DataT, DataT, DataT>(optimizerType);

  // Pre-allocate logits cache for forward/backward passes
  std::vector<DataT> logitsCache;
  std::array<DataT, outputDim> output;
  const size_t cacheSize = (numLayers - 1) * mlpData.back().inputDimension()
                           + mlpData.back().outputDimension();
  logitsCache.resize(cacheSize);

  // --- Training loop ---
  std::cout << "Backend: CPU (reference)" << std::endl;
  std::cout << "Starting training..." << std::endl;
  const auto trainingStart = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < options.m_epochs; ++epoch) {
    float epochLoss = 0.0f;
    size_t numBatches = 0;

    for (size_t batchStart = 0; batchStart < numSamples; batchStart += options.m_batchSize) {
      const size_t batchEnd = std::min(batchStart + options.m_batchSize, numSamples);
      const size_t currentBatchSize = batchEnd - batchStart;

      // Zero gradients before each batch
      using LayerT = std::remove_cvref_t<typename decltype(mlpData)::value_type>;
      std::ranges::for_each(mlpData, [](LayerT& layer) { layer.resetGrads(); });

      float batchLoss = 0.0f;

      for (size_t sampleIdx = batchStart; sampleIdx < batchEnd; ++sampleIdx) {
        const std::span<const DataT> input{uvData.data() + 2 * sampleIdx, 2};
        const std::span<const DataT> target{texelData.data() + 2 * sampleIdx, 2};

        // Forward pass (stores pre-activation logits in cache for backward pass)
        ex::forward<DataT, DataT, DataT, DataT, DataT>(output, input, mlpData, logitsCache);
        batchLoss += ex::mseLoss<DataT>(output, target);

        // MSE gradient scaled by 1/batchSize for batch averaging
        const float batchScale = 1.0f / static_cast<float>(currentBatchSize);
        std::vector lossGradient = ex::mseLossGradient<DataT>(output, target);
        std::ranges::for_each(lossGradient, [batchScale](DataT& v) { v *= batchScale; });

        // Backward pass (accumulates gradients into layer members)
        [[maybe_unused]] const std::vector upstreamGrad =
            ex::backward<DataT, DataT, DataT, DataT, DataT>(
                lossGradient, input, mlpData, logitsCache);
      }

      batchLoss /= static_cast<float>(currentBatchSize);
      epochLoss += batchLoss;
      numBatches++;

      // If bias is disabled, zero out bias gradients to prevent bias updates
      if (!hasBias) {
        std::ranges::for_each(mlpData, [](auto& layer) {
          std::ranges::fill(layer.biasGrads(), static_cast<DataT>(0));
        });
      }

      // Update weights using the selected optimizer
      optimizer->step(mlpData, lr);
    }

    const float avgLoss = epochLoss / static_cast<float>(numBatches);
    std::cout << std::format("Epoch [{}/{}], Loss: {:.6f}",
        epoch + 1, options.m_epochs, avgLoss) << std::endl;
  }
  std::cout << "Training completed!" << std::endl;
  {
    const auto trainingEnd = std::chrono::high_resolution_clock::now();
    const auto trainingMs = std::chrono::duration<double, std::milli>(trainingEnd - trainingStart).count();
    std::cout << std::format("Training time: {:.3f} ms", trainingMs) << std::endl;
  }

  // --- Reconstruct texture using the trained MLP ---
  std::cout << "Reconstructing texture..." << std::endl;
  ex::PixmapU8 texture{options.m_textureWidth, options.m_textureHeight};
  const std::vector reconstructUv = createUvData<DataT>(texture.width(), texture.height());
  const auto reconstructStart = std::chrono::high_resolution_clock::now();
  const std::vector inferenceOutput =
      ex::forwardBatch<DataT, DataT, DataT, DataT, DataT>(reconstructUv, mlpData);
  const auto reconstructEnd = std::chrono::high_resolution_clock::now();
  const auto reconstructMs = std::chrono::duration<double, std::milli>(reconstructEnd - reconstructStart).count();
  std::cout << std::format("Reconstruction time: {:.3f} ms", reconstructMs) << std::endl;
  mapToLdr<DataT>(inferenceOutput, texture);

  return texture;
}

// ============================================================================
// C++ fallback training (mlp.hlsl compiled as C++)
// ============================================================================

// Pack MLP layer weights/biases and training buffers into flat byte buffers
// matching the layout expected by mlp.hlsl.
// Extends the base PackedMlpBuffers with gradient, logits, and optimizer state.
template <ex::Arithmetic Type>
struct PackedTrainingBuffers
{
  std::vector<std::uint8_t> weightBuf;
  std::vector<std::uint8_t> biasBuf;
  std::vector<std::uint8_t> weightGradBuf;
  std::vector<std::uint8_t> biasGradBuf;
  std::vector<std::uint8_t> logitsCacheBuf;
  float lossValue = 0.0f;
  size_t biasStride = 0;
  [[maybe_unused]] uint32_t m_padd[2];
  uint2 matrixSizes{};

  // Optimizer state buffers (moment values are always float, one per parameter element)
  std::vector<std::uint8_t> weightFirstMomentBuf;
  std::vector<std::uint8_t> weightSecondMomentBuf;
  std::vector<std::uint8_t> biasFirstMomentBuf;
  std::vector<std::uint8_t> biasSecondMomentBuf;
  size_t timestep = 0;

  ByteAddressBuffer weightBAB() const { return ByteAddressBuffer{weightBuf}; }
  ByteAddressBuffer biasBAB() const { return ByteAddressBuffer{biasBuf}; }
  RWByteAddressBuffer weightRWBAB() { return RWByteAddressBuffer{weightBuf}; }
  RWByteAddressBuffer biasRWBAB() { return RWByteAddressBuffer{biasBuf}; }
  RWByteAddressBuffer weightGradRWBAB() { return RWByteAddressBuffer{weightGradBuf}; }
  ByteAddressBuffer weightGradBAB() const { return ByteAddressBuffer{weightGradBuf}; }
  RWByteAddressBuffer biasGradRWBAB() { return RWByteAddressBuffer{biasGradBuf}; }
  ByteAddressBuffer biasGradBAB() const { return ByteAddressBuffer{biasGradBuf}; }
  RWByteAddressBuffer logitsCacheRWBAB() { return RWByteAddressBuffer{logitsCacheBuf}; }
  RWByteAddressBuffer lossRWBAB()
  {
    return RWByteAddressBuffer{reinterpret_cast<std::uint8_t*>(&lossValue), sizeof(float)};
  }
  RWByteAddressBuffer weightFirstMomentRWBAB() { return RWByteAddressBuffer{weightFirstMomentBuf}; }
  RWByteAddressBuffer weightSecondMomentRWBAB() { return RWByteAddressBuffer{weightSecondMomentBuf}; }
  RWByteAddressBuffer biasFirstMomentRWBAB() { return RWByteAddressBuffer{biasFirstMomentBuf}; }
  RWByteAddressBuffer biasSecondMomentRWBAB() { return RWByteAddressBuffer{biasSecondMomentBuf}; }

  void pack(const std::span<const ex::MlpLayer<Type, Type, Type, Type>> mlpData, const bool hasBias)
  {
    const size_t numLayers = mlpData.size();
    const size_t hiddenDim = (numLayers > 1) ? mlpData[0].outputDimension()
                                             : mlpData[0].inputDimension();

    // Delegate weight/bias packing to shared PackedMlpBuffers
    ex::PackedMlpBuffers<Type> base;
    base.pack(mlpData, hasBias);
    weightBuf = std::move(base.weightBuf);
    biasBuf = std::move(base.biasBuf);
    matrixSizes = base.matrixSizes;

    biasStride = ex::alignBytes(hiddenDim * sizeof(Type), ex::VECTOR_ALIGNMENT);

    // Allocate gradient buffers (same size as weight/bias buffers)
    weightGradBuf.assign(weightBuf.size(), 0);
    biasGradBuf.assign(biasBuf.size(), 0);
  }

  void allocateLogitsCache(const size_t batchSize, const size_t numLayers)
  {
    const size_t perSample = biasStride * numLayers;
    logitsCacheBuf.assign(perSample * batchSize, 0);
  }

  void allocateOptimizerState(const ex::OptimizerType optType)
  {
    const size_t weightElements = weightBuf.size() / sizeof(Type);
    const size_t biasElements = biasBuf.size() / sizeof(Type);

    if (optType == ex::OptimizerType::ADAM) {
      weightFirstMomentBuf.assign(weightElements * sizeof(float), 0);
      weightSecondMomentBuf.assign(weightElements * sizeof(float), 0);
      biasFirstMomentBuf.assign(biasElements * sizeof(float), 0);
      biasSecondMomentBuf.assign(biasElements * sizeof(float), 0);
    } else if (optType == ex::OptimizerType::LION) {
      // Lion uses only first moment (momentum)
      weightFirstMomentBuf.assign(weightElements * sizeof(float), 0);
      biasFirstMomentBuf.assign(biasElements * sizeof(float), 0);
    }
    timestep = 0;
  }

  [[nodiscard]] size_t logitsPerSample(const size_t numLayers) const
  {
    return biasStride * numLayers;
  }

  void zeroGradients()
  {
    std::ranges::fill(weightGradBuf, std::uint8_t{0});
    std::ranges::fill(biasGradBuf, std::uint8_t{0});
    lossValue = 0.0f;
  }

  // Apply optimizer directly on packed buffers (no unpack/repack)
  void applyOptimizer(const ex::OptimizerType optType, const float lr)
  {
    const uint wSize = static_cast<uint>(weightBuf.size());
    const uint bSize = static_cast<uint>(biasBuf.size());

    switch (optType) {
      case ex::OptimizerType::SGD:
        optimizer::sgdUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(), lr, wSize);
        optimizer::sgdUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(), lr, bSize);
        break;
      case ex::OptimizerType::ADAM: {
        ++timestep;
        const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));
        optimizer::adamUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(),
            weightFirstMomentRWBAB(), weightSecondMomentRWBAB(),
            lr, beta1, beta2, epsilon, bc1, bc2, wSize);
        optimizer::adamUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(),
            biasFirstMomentRWBAB(), biasSecondMomentRWBAB(),
            lr, beta1, beta2, epsilon, bc1, bc2, bSize);
        break;
      }
      case ex::OptimizerType::LION: {
        const float beta1 = 0.9f, beta2 = 0.99f, weightDecay = 0.3f;
        optimizer::lionUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(),
            weightFirstMomentRWBAB(), lr, beta1, beta2, weightDecay, wSize);
        optimizer::lionUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(),
            biasFirstMomentRWBAB(), lr, beta1, beta2, weightDecay, bSize);
        break;
      }
      default:
        break;
    }
  }
};

// ----------------------------------------------------------------------------
// Training kernel: forward + MSE loss + backward for one batch
// Delegates to shared texkernel::trainingStep from texture_training_common.hlsl.
// ----------------------------------------------------------------------------

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          typename ActivationHiddenT, typename ActivationLastT>
void cppFallbackTrainingKernel(PackedTrainingBuffers<Type>& packed,
                               const std::vector<Type>& uvData,
                               const std::vector<Type>& texelData,
                               const size_t batchSize,
                               const size_t batchIndex,
                               const size_t currentBatchSize)
{
  constexpr auto DT = ex::DxLinalgDataTypeOf<Type>::value;

  ByteAddressBuffer uvBuf{uvData};
  ByteAddressBuffer targetBuf{texelData};

  const uint totalTasks = static_cast<uint>(currentBatchSize);
  const uint numThreads = std::max(1u, std::thread::hardware_concurrency());
  const uint tasksPerThread = totalTasks / numThreads;
  const uint remainder = totalTasks % numThreads;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  uint taskStart = 0;
  for (uint t = 0; t < numThreads; ++t) {
    const uint taskEnd = taskStart + tasksPerThread + (t < remainder ? 1 : 0);
    threads.emplace_back([&, taskStart, taskEnd]() {
      for (uint threadId = taskStart; threadId < taskEnd; ++threadId) {
        texkernel::trainingStep<Type, NUM_LAYERS, HIDDEN_DIM,
            DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
            128, 16, 64>(
            threadId, uvBuf, targetBuf, packed.weightBAB(), packed.biasBAB(),
            packed.weightGradRWBAB(), packed.biasGradRWBAB(),
            packed.logitsCacheRWBAB(), packed.lossRWBAB(),
            packed.matrixSizes, static_cast<uint>(batchSize),
            static_cast<uint>(batchIndex), static_cast<uint>(currentBatchSize),
            static_cast<uint>(packed.biasStride));
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Dispatch activation types for training
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM>
bool dispatchTrainingActivation(const ex::ActivationType hiddenAct,
                                PackedTrainingBuffers<Type>& packed,
                                const std::vector<Type>& uvData,
                                const std::vector<Type>& texelData,
                                const size_t batchSize,
                                const size_t batchIndex,
                                const size_t currentBatchSize)
{
  // Last activation is always Sigmoid for this example
  using Sigmoid = mininn::SigmoidActivation;

  #define DISPATCH_TRAIN_ACT(HiddenT, hiddenE) \
    if (hiddenAct == (hiddenE)) { \
      cppFallbackTrainingKernel<Type, NUM_LAYERS, HIDDEN_DIM, HiddenT, Sigmoid>( \
          packed, uvData, texelData, batchSize, batchIndex, currentBatchSize); \
      return true; \
    }

  DISPATCH_TRAIN_ACT(mininn::IdentityActivation,  ex::ActivationType::IDENTITY)
  DISPATCH_TRAIN_ACT(mininn::SigmoidActivation,   ex::ActivationType::SIGMOID)
  DISPATCH_TRAIN_ACT(mininn::ReluActivation,       ex::ActivationType::RELU)
  DISPATCH_TRAIN_ACT(mininn::LeakyReluActivation,  ex::ActivationType::LEAKY_RELU)

  #undef DISPATCH_TRAIN_ACT
  return false;
}

// Dispatch NUM_LAYERS and HIDDEN_DIM for training
template <ex::Arithmetic Type>
bool dispatchTraining(const size_t numLayers, const size_t hiddenDim,
                      const ex::ActivationType hiddenAct,
                      PackedTrainingBuffers<Type>& packed,
                      const std::vector<Type>& uvData,
                      const std::vector<Type>& texelData,
                      const size_t batchSize,
                      const size_t batchIndex,
                      const size_t currentBatchSize)
{
  #define DISPATCH_TRAIN(NL, HD) \
    if (numLayers == (NL) && hiddenDim == (HD)) \
      return dispatchTrainingActivation<Type, (NL), (HD)>( \
          hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize);

  // 2 layers (1 backbone)
  DISPATCH_TRAIN(2, 8)   DISPATCH_TRAIN(2, 16)
  DISPATCH_TRAIN(2, 32)  DISPATCH_TRAIN(2, 64)
  // 3 layers (2 backbone)
  DISPATCH_TRAIN(3, 8)   DISPATCH_TRAIN(3, 16)
  DISPATCH_TRAIN(3, 32)  DISPATCH_TRAIN(3, 64)
  // 4 layers (3 backbone) — default configuration
  DISPATCH_TRAIN(4, 8)   DISPATCH_TRAIN(4, 16)
  DISPATCH_TRAIN(4, 32)  DISPATCH_TRAIN(4, 64)
  // 5 layers (4 backbone)
  DISPATCH_TRAIN(5, 8)   DISPATCH_TRAIN(5, 16)
  DISPATCH_TRAIN(5, 32)  DISPATCH_TRAIN(5, 64)

  #undef DISPATCH_TRAIN
  return false;
}

// ----------------------------------------------------------------------------
// Forward inference kernel for texture reconstruction
// ----------------------------------------------------------------------------

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          typename ActivationHiddenT, typename ActivationLastT>
void cppFallbackForwardKernel(const PackedTrainingBuffers<Type>& packed,
                              const std::vector<Type>& uvData,
                              std::vector<Type>& output,
                              const size_t numTasks)
{
  constexpr auto DT = ex::DxLinalgDataTypeOf<Type>::value;

  ByteAddressBuffer uvBuf{uvData};
  RWByteAddressBuffer outBuf{output};

  const uint totalTasks = static_cast<uint>(numTasks);
  const uint numThreads = std::max(1u, std::thread::hardware_concurrency());
  const uint tasksPerThread = totalTasks / numThreads;
  const uint remainder = totalTasks % numThreads;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  uint taskStart = 0;
  for (uint t = 0; t < numThreads; ++t) {
    const uint taskEnd = taskStart + tasksPerThread + (t < remainder ? 1 : 0);
    threads.emplace_back([&, taskStart, taskEnd]() {
      for (uint task = taskStart; task < taskEnd; ++task) {
        texkernel::inferenceStep<Type, NUM_LAYERS, HIDDEN_DIM,
            DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
            128, 16, 64, true>(
            task, uvBuf, outBuf, packed.weightBAB(), packed.biasBAB(),
            packed.matrixSizes, totalTasks);
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Dispatch activation types for forward inference
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM>
bool dispatchForwardActivation(const ex::ActivationType hiddenAct,
                               const PackedTrainingBuffers<Type>& packed,
                               const std::vector<Type>& uvData,
                               std::vector<Type>& output,
                               const size_t numTasks)
{
  using Sigmoid = mininn::SigmoidActivation;

  #define DISPATCH_FWD_ACT(HiddenT, hiddenE) \
    if (hiddenAct == (hiddenE)) { \
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, HiddenT, Sigmoid>( \
          packed, uvData, output, numTasks); \
      return true; \
    }

  DISPATCH_FWD_ACT(mininn::IdentityActivation,  ex::ActivationType::IDENTITY)
  DISPATCH_FWD_ACT(mininn::SigmoidActivation,   ex::ActivationType::SIGMOID)
  DISPATCH_FWD_ACT(mininn::ReluActivation,       ex::ActivationType::RELU)
  DISPATCH_FWD_ACT(mininn::LeakyReluActivation,  ex::ActivationType::LEAKY_RELU)

  #undef DISPATCH_FWD_ACT
  return false;
}

// Dispatch NUM_LAYERS and HIDDEN_DIM for forward inference
template <ex::Arithmetic Type>
bool dispatchForward(const size_t numLayers, const size_t hiddenDim,
                     const ex::ActivationType hiddenAct,
                     const PackedTrainingBuffers<Type>& packed,
                     const std::vector<Type>& uvData,
                     std::vector<Type>& output,
                     const size_t numTasks)
{
  #define DISPATCH_FWD(NL, HD) \
    if (numLayers == (NL) && hiddenDim == (HD)) \
      return dispatchForwardActivation<Type, (NL), (HD)>( \
          hiddenAct, packed, uvData, output, numTasks);

  DISPATCH_FWD(2, 8)   DISPATCH_FWD(2, 16)
  DISPATCH_FWD(2, 32)  DISPATCH_FWD(2, 64)
  DISPATCH_FWD(3, 8)   DISPATCH_FWD(3, 16)
  DISPATCH_FWD(3, 32)  DISPATCH_FWD(3, 64)
  DISPATCH_FWD(4, 8)   DISPATCH_FWD(4, 16)
  DISPATCH_FWD(4, 32)  DISPATCH_FWD(4, 64)
  DISPATCH_FWD(5, 8)   DISPATCH_FWD(5, 16)
  DISPATCH_FWD(5, 32)  DISPATCH_FWD(5, 64)

  #undef DISPATCH_FWD
  return false;
}

// ----------------------------------------------------------------------------
// C++ fallback training loop + reconstruction
// ----------------------------------------------------------------------------

/*!
  \brief Train the MLP using C++ fallback and reconstruct the texture.

  Uses mlp.hlsl compiled as C++ (via hlsl_compat.hpp) for forward/backward passes.
  The optimizer operates directly on packed byte buffers, eliminating the need
  to unpack gradients or re-pack weights between batches.
*/
template <ex::Arithmetic DataT>
auto trainAndReconstructTextureCppFallback(
    std::span<ex::MlpLayer<DataT, DataT, DataT, DataT>> mlpData,
    const std::vector<DataT>& uvData,
    const std::vector<DataT>& texelData,
    const bool hasBias,
    const CliOptions& options) -> ex::PixmapU8
{
  const size_t numSamples = uvData.size() / 2;
  const size_t numLayers = mlpData.size();
  const size_t hiddenDim = mlpData.front().outputDimension();
  const auto hiddenAct = mlpData.front().configuration().m_activation;
  const float lr = static_cast<float>(options.m_learningRate);
  const auto optimizerType = ex::getOptimizerTypeFromString(options.m_optimizer);

  // Pack weights/biases into byte buffers
  PackedTrainingBuffers<DataT> packed;
  packed.pack(mlpData, hasBias);
  packed.allocateLogitsCache(options.m_batchSize, numLayers);
  packed.allocateOptimizerState(optimizerType);

  // --- Training loop ---
  std::cout << "Backend: C++ fallback" << std::endl;
  std::cout << "Starting training..." << std::endl;
  const auto trainingStart = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < options.m_epochs; ++epoch) {
    float epochLoss = 0.0f;
    size_t numBatches = 0;

    for (size_t batchStart = 0; batchStart < numSamples; batchStart += options.m_batchSize) {
      const size_t batchEnd = std::min(batchStart + options.m_batchSize, numSamples);
      const size_t currentBatchSize = batchEnd - batchStart;
      const size_t batchIndex = batchStart / options.m_batchSize;

      // Zero gradients and loss
      packed.zeroGradients();

      // Run training kernel (forward + backward for all samples in batch)
      if (!dispatchTraining<DataT>(numLayers, hiddenDim, hiddenAct,
              packed, uvData, texelData, options.m_batchSize, batchIndex, currentBatchSize)) {
        std::cerr << std::format("[Error] C++ fallback: unsupported training config (layers={}, hiddenDim={})\n",
            numLayers, hiddenDim);
        std::abort();
      }

      // Accumulate loss
      const float batchLoss = packed.lossValue / static_cast<float>(currentBatchSize);
      epochLoss += batchLoss;
      numBatches++;

      // Apply optimizer directly on packed buffers (no unpack/repack)
      packed.applyOptimizer(optimizerType, lr);
    }

    const float avgLoss = epochLoss / static_cast<float>(numBatches);
    std::cout << std::format("Epoch [{}/{}], Loss: {:.6f}",
        epoch + 1, options.m_epochs, avgLoss) << std::endl;
  }
  std::cout << "Training completed!" << std::endl;
  {
    const auto trainingEnd = std::chrono::high_resolution_clock::now();
    const auto trainingMs = std::chrono::duration<double, std::milli>(trainingEnd - trainingStart).count();
    std::cout << std::format("Training time: {:.3f} ms", trainingMs) << std::endl;
  }

  // --- Reconstruct texture using the trained MLP ---
  std::cout << "Reconstructing texture..." << std::endl;
  ex::PixmapU8 texture{options.m_textureWidth, options.m_textureHeight};
  const std::vector reconstructUv = createUvData<DataT>(texture.width(), texture.height());
  const size_t numPixels = texture.width() * texture.height();
  std::vector<DataT> output(numPixels * 2);

  const auto reconstructStart = std::chrono::high_resolution_clock::now();
  if (!dispatchForward<DataT>(numLayers, hiddenDim, hiddenAct,
          packed, reconstructUv, output, numPixels)) {
    std::cerr << std::format("[Error] C++ fallback: unsupported inference config (layers={}, hiddenDim={})\n",
        numLayers, hiddenDim);
    std::abort();
  }
  const auto reconstructEnd = std::chrono::high_resolution_clock::now();
  const auto reconstructMs = std::chrono::duration<double, std::milli>(reconstructEnd - reconstructStart).count();
  std::cout << std::format("Reconstruction time: {:.3f} ms", reconstructMs) << std::endl;

  mapToLdr<DataT>(output, texture);
  return texture;
}

template <ex::Arithmetic Type>
auto buildKernelDefinitions(std::span<ex::MlpLayer<Type, Type, Type, Type>> mlpData,
                            const size_t batchSize,
                            const float learningRate,
                            const size_t weightBufferSize,
                            const size_t biasBufferSize,
                            const size_t weightChunkSize,
                            const size_t biasChunkSize,
                            const ex::MatrixLayout weightMatrixLayout,
                            const bool useSoftwareLinalg,
                            const bool hasBias,
                            const float optimizerBeta1 = 0.0f,
                            const float optimizerBeta2 = 0.0f,
                            const float optimizerEpsilon = 0.0f,
                            const float optimizerWeightDecay = 0.0f) -> std::vector<ex::OptionString>
{
  const size_t inputDim = mlpData.front().inputDimension();
  const size_t outputDim = mlpData.back().outputDimension();
  const size_t numLayers = mlpData.size();
  const size_t hiddenLayerDim = mlpData.front().outputDimension();
  const ex::ActivationType activationHidden = mlpData.front().configuration().m_activation;
  const ex::ActivationType activationLast = mlpData.back().configuration().m_activation;
  constexpr size_t numThreadsX = 32;

  std::vector<ex::OptionString> defs;
  defs.reserve(19);

  // MLP architecture
  defs.push_back(ex::createOptionString("MINIDXNN_INPUT_DIMENSION={}", inputDim));
  defs.push_back(ex::createOptionString("MINIDXNN_OUTPUT_DIMENSION={}", outputDim));
  defs.push_back(ex::createOptionString("MINIDXNN_NUM_LAYERS={}", numLayers));
  defs.push_back(ex::createOptionString("MINIDXNN_HIDDEN_LAYER_DIMENSIONS={}", hiddenLayerDim));
  defs.push_back(ex::createOptionString("MINIDXNN_HAS_BIAS={}", hasBias ? 1 : 0));
  defs.push_back(ex::createOptionString("MINIDXNN_LEARNING_RATE={}", learningRate));

  // Activation functions
  defs.push_back(ex::createOptionString("MINIDXNN_ACTIVATION_HIDDEN_TYPE={}", ex::getActivationTypeString(activationHidden)));
  defs.push_back(ex::createOptionString("MINIDXNN_ACTIVATION_LAST_TYPE={}", ex::getActivationTypeString(activationLast)));

  // Weight matrix memory layout and alignment
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", static_cast<int>(weightMatrixLayout)));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_ALIGNMENT={}", ex::MATRIX_ALIGNMENT));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", ex::MATRIX_VECTOR_STRIDE_ALIGNMENT));
  defs.push_back(ex::createOptionString("MINIDXNN_BIAS_VECTOR_ALIGNMENT={}", ex::VECTOR_ALIGNMENT));

  // Dispatch configuration
  defs.push_back(ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", numThreadsX));
  defs.push_back(ex::createOptionString("MINIDXNN_BATCH_SIZE={}", batchSize));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_BUFFER_SIZE={}", weightBufferSize));
  defs.push_back(ex::createOptionString("MINIDXNN_BIAS_BUFFER_SIZE={}", biasBufferSize));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_CHUNK_SIZE={}", weightChunkSize));
  defs.push_back(ex::createOptionString("MINIDXNN_BIAS_CHUNK_SIZE={}", biasChunkSize));
  defs.push_back(ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", useSoftwareLinalg ? 1 : 0));

  // Optimizer hyperparameters (passed as compile-time defines to avoid float uniform binding issues)
  defs.push_back(ex::createOptionString("MINIDXNN_OPTIMIZER_BETA1={:.10f}f", optimizerBeta1));
  defs.push_back(ex::createOptionString("MINIDXNN_OPTIMIZER_BETA2={:.10f}f", optimizerBeta2));
  defs.push_back(ex::createOptionString("MINIDXNN_OPTIMIZER_EPSILON={:.10e}f", optimizerEpsilon));
  defs.push_back(ex::createOptionString("MINIDXNN_OPTIMIZER_WEIGHT_DECAY={:.10f}f", optimizerWeightDecay));

  return defs;
}

/*!
  \brief Train the MLP on GPU and reconstruct the texture.

  Uses DirectX compute shaders for parallel batch training with SGD/Adam/Lion
  optimizers, then performs forward inference to reconstruct the full texture.
*/
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
template <ex::Arithmetic Type>
auto trainAndReconstructTextureGpu(std::span<ex::MlpLayer<Type, Type, Type, Type>> mlpData,
                                   const std::vector<Type>& uvData,
                                   const std::vector<Type>& texelData,
                                   const bool hasBias,
                                   const CliOptions& options) -> ex::PixmapU8
{
  const ex::MatrixLayout weightMatrixLayout = ex::MatrixLayout::ROW_MAJOR;
  constexpr size_t weightChunkSize = ex::MATRIX_ALIGNMENT;
  constexpr size_t biasChunkSize = ex::VECTOR_ALIGNMENT;
  const auto optimizerType = ex::getOptimizerTypeFromString(options.m_optimizer);

  // Initialize GFX context
  std::shared_ptr context = ex::createGfxContext(options.m_enableDebugMode);
  const std::filesystem::path shaderDir = ex::getComputeShaderDir();
  const std::array includeDirList = ex::getHlslIncludeDirList();

  constexpr size_t numThreadsX = 32;

  //
  std::vector<size_t> matrixSizeList;
  matrixSizeList.resize(mlpData.size());

  // Create buffers
  std::shared_ptr uvBuffer = ex::createGfxBuffer<Type>(*context, uvData);
  std::shared_ptr targetBuffer = ex::createGfxBuffer<Type>(*context, texelData);
  std::shared_ptr weightBuffer = ex::convertToMatrixBuffer<Type>(*context, mlpData, weightMatrixLayout, matrixSizeList, ex::MATRIX_ALIGNMENT, ex::MATRIX_VECTOR_STRIDE_ALIGNMENT);
  std::shared_ptr weightGradBuffer = ex::createGfxBuffer<Type>(*context, weightBuffer->getSize() / sizeof(Type));
  std::shared_ptr biasBuffer = ex::convertToVectorBuffer<Type>(*context, mlpData, ex::VECTOR_ALIGNMENT);
  std::shared_ptr biasGradBuffer = ex::createGfxBuffer<Type>(*context, biasBuffer->getSize() / sizeof(Type));
  std::shared_ptr logitsCacheBuffer = ex::createGfxBuffer<Type>(*context, options.m_batchSize * biasBuffer->getSize() / sizeof(Type));
  std::shared_ptr lossBuffer = ex::createGfxBuffer<float>(*context, 1);

  // Adam/Lion optimizer moment buffers
  const size_t weightElements = weightBuffer->getSize() / sizeof(Type);
  const size_t biasElements = biasBuffer->getSize() / sizeof(Type);
  std::shared_ptr<GfxBuffer> weightFirstMomentBuffer;
  std::shared_ptr<GfxBuffer> weightSecondMomentBuffer;
  std::shared_ptr<GfxBuffer> biasFirstMomentBuffer;
  std::shared_ptr<GfxBuffer> biasSecondMomentBuffer;

  if (optimizerType == ex::OptimizerType::ADAM) {
    weightFirstMomentBuffer = ex::createGfxBuffer<float>(*context, weightElements);
    weightSecondMomentBuffer = ex::createGfxBuffer<float>(*context, weightElements);
    biasFirstMomentBuffer = ex::createGfxBuffer<float>(*context, biasElements);
    biasSecondMomentBuffer = ex::createGfxBuffer<float>(*context, biasElements);
    // Zero-initialize moment buffers (Adam requires m=0, v=0 at start)
    gfxCommandClearBuffer(*context, *weightFirstMomentBuffer);
    gfxCommandClearBuffer(*context, *weightSecondMomentBuffer);
    gfxCommandClearBuffer(*context, *biasFirstMomentBuffer);
    gfxCommandClearBuffer(*context, *biasSecondMomentBuffer);
    gfxFinish(*context);
  } else if (optimizerType == ex::OptimizerType::LION) {
    weightFirstMomentBuffer = ex::createGfxBuffer<float>(*context, weightElements);
    biasFirstMomentBuffer = ex::createGfxBuffer<float>(*context, biasElements);
    // Zero-initialize momentum buffers
    gfxCommandClearBuffer(*context, *weightFirstMomentBuffer);
    gfxCommandClearBuffer(*context, *biasFirstMomentBuffer);
    gfxFinish(*context);
  }

  // Optimizer hyperparameters
  constexpr float adamBeta1 = 0.9f, adamBeta2 = 0.999f, adamEpsilon = 1e-8f;
  constexpr float lionBeta1 = 0.9f, lionBeta2 = 0.99f, lionWeightDecay = 0.3f;

  // Create and run the example kernel
  {
    const size_t numOptElements = (std::max)(weightElements, biasElements);
    const auto [optBeta1, optBeta2, optEpsilon, optWeightDecay] = [&]() -> std::tuple<float, float, float, float> {
      if (optimizerType == ex::OptimizerType::ADAM)
        return {adamBeta1, adamBeta2, adamEpsilon, 0.0f};
      else if (optimizerType == ex::OptimizerType::LION)
        return {lionBeta1, lionBeta2, 0.0f, lionWeightDecay};
      else
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }();
    const std::vector definitions = buildKernelDefinitions(mlpData, options.m_batchSize, static_cast<float>(options.m_learningRate), weightBuffer->getSize(), biasBuffer->getSize(), weightChunkSize, biasChunkSize, weightMatrixLayout, options.m_useSoftwareLinalg, hasBias, optBeta1, optBeta2, optEpsilon, optWeightDecay);

    // Create shader programs and kernels for training
    // Use separate programs for backward and optimizer to isolate parameter bindings
    std::shared_ptr program = ex::createGfxProgram(*context, "02_texture_training", shaderDir, includeDirList);
    const ex::OptionString backwardKernelName = ex::createOptionString("trainingF{}Kernel", 8 * sizeof(Type));
    std::shared_ptr backwardKernel = ex::createGfxComputeKernel(*context, *program, backwardKernelName.data(), definitions);

    const ex::OptionString optimizationKernelName = ex::createOptionString("{}StepF{}Kernel", options.m_optimizer, 8 * sizeof(Type));
    std::shared_ptr<GfxKernel> optimizationKernel;

    const size_t numSamples = uvData.size() / 2;
    size_t timestep = 0;

    float totalTrainingKernelTimeMs = 0.0f;
    float totalOptimizerKernelTimeMs = 0.0f;

    std::cout << "Backend: GPU" << std::endl;
    std::cout << "Starting training..." << std::endl;
    for (size_t epoch = 0; epoch < options.m_epochs; ++epoch) {
      float epochLoss = 0.0f;
      size_t numBatches = 0;

      for (size_t batchStart = 0; batchStart < numSamples; batchStart += options.m_batchSize) {
        const size_t batchEnd = std::min(batchStart + options.m_batchSize, numSamples);
        const size_t currentBatchSize = batchEnd - batchStart;

        // Zero gradients
        gfxCommandClearBuffer(*context, *weightGradBuffer);
        gfxCommandClearBuffer(*context, *biasGradBuffer);
        gfxCommandClearBuffer(*context, *lossBuffer);
        gfxFinish(*context);

        const size_t batchIndex = batchStart / options.m_batchSize;
        {
          float kernelTimeMs = 0.0f;
          const size_t threadGroupSize = ex::align(currentBatchSize, numThreadsX) / numThreadsX;
          ex::runKernel(*context, *program, *backwardKernel, threadGroupSize,
              {
                ex::bind(*uvBuffer, "UvBuffer"),
                ex::bind(*targetBuffer, "TargetBuffer"),
                ex::bind(*weightBuffer, "WeightBuffer"),
                ex::bind(*biasBuffer, "BiasBuffer"),
                ex::bind(*weightGradBuffer, "WeightGradBuffer"),
                ex::bind(*biasGradBuffer, "BiasGradBuffer"),
                ex::bind(*logitsCacheBuffer, "LogitsCacheBuffer"),
                ex::bind(*lossBuffer, "LossBuffer"),
              },
              {
                ex::bind(static_cast<std::int32_t>(matrixSizeList.front()), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
                ex::bind(static_cast<std::int32_t>((matrixSizeList.size() > 1) ? matrixSizeList.at(1) : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
                ex::bind(static_cast<std::int32_t>(currentBatchSize), "TEST_CURRENT_BATCH_SIZE"),
                ex::bind(static_cast<std::int32_t>(batchIndex), "TEST_BATCH_INDEX"),
                ex::bind(static_cast<std::int32_t>(biasElements * sizeof(Type)), "TEST_BIAS_STRIDE"),
              },
              kernelTimeMs);
          totalTrainingKernelTimeMs += kernelTimeMs;
        }
        {
          std::shared_ptr staging = ex::createGfxBuffer<float>(*context, 1, kGfxCpuAccess_Read);
          ex::copyBuffer(*context, *lossBuffer, *staging);
          const std::span loss = ex::mapToCpu<float>(*context, *staging);
          // Average loss for this batch
          const float batchLoss = loss[0] / static_cast<float>(currentBatchSize);
          epochLoss += batchLoss;
          numBatches++;
        }

        // Update weights using optimizer
        {
          ++timestep;
          // Create the optimizer kernel lazily on first use
          if (!optimizationKernel) {
            optimizationKernel = ex::createGfxComputeKernel(*context, *program, optimizationKernelName.data(), definitions);
          }
          const size_t threadGroupSize = ex::align(numOptElements, numThreadsX) / numThreadsX;

          // Build buffer bindings for optimizer dispatch
          std::vector<ex::BufferBindingDataT> optBuffersVec;
          if (optimizerType == ex::OptimizerType::ADAM) {
            optBuffersVec = {
              ex::bind(*weightBuffer, "RWWeightBuffer"),
              ex::bind(*biasBuffer, "RWBiasBuffer"),
              ex::bind(*weightGradBuffer, "WeightGradBuffer"),
              ex::bind(*biasGradBuffer, "BiasGradBuffer"),
              ex::bind(*weightFirstMomentBuffer, "WeightFirstMoment"),
              ex::bind(*weightSecondMomentBuffer, "WeightSecondMoment"),
              ex::bind(*biasFirstMomentBuffer, "BiasFirstMoment"),
              ex::bind(*biasSecondMomentBuffer, "BiasSecondMoment"),
            };
          } else if (optimizerType == ex::OptimizerType::LION) {
            optBuffersVec = {
              ex::bind(*weightBuffer, "RWWeightBuffer"),
              ex::bind(*biasBuffer, "RWBiasBuffer"),
              ex::bind(*weightGradBuffer, "WeightGradBuffer"),
              ex::bind(*biasGradBuffer, "BiasGradBuffer"),
              ex::bind(*weightFirstMomentBuffer, "WeightFirstMoment"),
              ex::bind(*biasFirstMomentBuffer, "BiasFirstMoment"),
            };
          } else {
            optBuffersVec = {
              ex::bind(*weightBuffer, "RWWeightBuffer"),
              ex::bind(*biasBuffer, "RWBiasBuffer"),
              ex::bind(*weightGradBuffer, "WeightGradBuffer"),
              ex::bind(*biasGradBuffer, "BiasGradBuffer"),
            };
          }

          // Int bindings for optimizer (timestep)
          std::initializer_list<ex::IntBindingDataT> optInts = {
            ex::bind(static_cast<std::int32_t>(timestep), "OptimizerTimestep"),
          };

          float optKernelTimeMs = 0.0f;
          ex::runKernel(*context, *program, *optimizationKernel, threadGroupSize,
              std::span<const ex::BufferBindingDataT>{optBuffersVec}, optInts,
              optKernelTimeMs);
          totalOptimizerKernelTimeMs += optKernelTimeMs;
        }
      }

      const float avgLoss = epochLoss / static_cast<float>(numBatches);
      std::cout << std::format("Epoch [{}/{}], Loss: {:.6f}", epoch + 1, options.m_epochs, avgLoss) << std::endl;
    }
    std::cout << "Training completed!" << std::endl;
    std::cout << std::format("Training kernel time: {:.3f} ms", totalTrainingKernelTimeMs) << std::endl;
    std::cout << std::format("Optimizer kernel time: {:.3f} ms", totalOptimizerKernelTimeMs) << std::endl;
  }

  // --- Reconstruct texture using the trained MLP ---
  std::cout << "Reconstructing texture..." << std::endl;
  ex::PixmapU8 texture{options.m_textureWidth, options.m_textureHeight};
  const size_t numPixels = static_cast<size_t>(options.m_textureWidth) * static_cast<size_t>(options.m_textureHeight);
  const std::vector reconstructUv = createUvData<Type>(texture.width(), texture.height());
  std::shared_ptr reconstructUvBuffer = ex::createGfxBuffer<Type>(*context, reconstructUv);
  std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(*context, numPixels * 2);

  {
    const std::vector definitions = buildKernelDefinitions(mlpData, options.m_batchSize, static_cast<float>(options.m_learningRate), weightBuffer->getSize(), biasBuffer->getSize(), weightChunkSize, biasChunkSize, weightMatrixLayout, options.m_useSoftwareLinalg, hasBias);
    std::shared_ptr program = ex::createGfxProgram(*context, "02_texture_training", shaderDir, includeDirList);
    const ex::OptionString inferenceKernelName = ex::createOptionString("inferenceF{}Kernel", 8 * sizeof(Type));
    std::shared_ptr inferenceKernel = ex::createGfxComputeKernel(*context, *program, inferenceKernelName.data(), definitions);

    const size_t threadGroupSize = ex::align(numPixels, numThreadsX) / numThreadsX;
    float inferenceKernelTimeMs = 0.0f;
    ex::runKernel(*context, *program, *inferenceKernel, threadGroupSize,
        {
          ex::bind(*reconstructUvBuffer, "UvBuffer"),
          ex::bind(*outputBuffer, "OutputBuffer"),
          ex::bind(*weightBuffer, "WeightBuffer"),
          ex::bind(*biasBuffer, "BiasBuffer"),
        },
        {
          ex::bind(static_cast<std::int32_t>(matrixSizeList.front()), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
          ex::bind(static_cast<std::int32_t>((matrixSizeList.size() > 1) ? matrixSizeList.at(1) : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
          ex::bind(static_cast<std::int32_t>(numPixels), "TEST_NUM_INFERENCE_TASKS"),
        },
        inferenceKernelTimeMs);
    std::cout << std::format("Reconstruction time: {:.3f} ms", inferenceKernelTimeMs) << std::endl;
  }

  // Read back results
  std::shared_ptr stagingOutput = ex::createGfxBuffer<Type>(*context, numPixels * 2, kGfxCpuAccess_Read);
  ex::copyBuffer(*context, *outputBuffer, *stagingOutput);
  const std::span outputData = ex::mapToCpu<Type>(*context, *stagingOutput);
  mapToLdr<Type>(outputData, texture);

  return texture;
}
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

/*!
  \brief Train the MLP and reconstruct the texture.

  Dispatches to CPU or GPU implementation based on the --cpu flag.

  \param mlpData    MLP layers (modified in-place during training)
  \param uvData     Training UV coordinates
  \param texelData  Ground-truth texel values
  \param options    CLI options controlling execution path and hyperparameters
  \return Reconstructed texture as an 8-bit grayscale pixmap
*/
template <ex::Arithmetic DataT>
auto trainAndReconstructTexture(
    std::span<ex::MlpLayer<DataT, DataT, DataT, DataT>> mlpData,
    const std::vector<DataT>& uvData,
    const std::vector<DataT>& texelData,
    const bool hasBias,
    const CliOptions& options) -> ex::PixmapU8
{
  if (options.m_useCpuMlpOperations) {
    return trainAndReconstructTextureCpu<DataT>(mlpData, uvData, texelData, hasBias, options);
  }
  else if (ex::isCppFallbackForced || options.m_useCppFallback) {
    return trainAndReconstructTextureCppFallback<DataT>(mlpData, uvData, texelData, hasBias, options);
  }
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  else {
    return trainAndReconstructTextureGpu<DataT>(mlpData, uvData, texelData, hasBias, options);
  }
#endif
}

} // namespace

// ============================================================================
// Entry point
// ============================================================================

auto main(const int argc, const char** argv) -> int
{
  // Parse command-line arguments
  CliOptions options{};
  std::unique_ptr cliParser = createCommandLineParser(options);
  CLI11_PARSE(*cliParser, argc, argv)

  using DataT = half_float::half;

  // Step 1: Create ground-truth texture pattern
  std::cout << std::format("Creating {} texture ({}x{})...",
      options.m_texturePattern, options.m_textureWidth, options.m_textureHeight)
            << std::endl;
  const auto texturePattern = ex::getTexturePatternFromString(options.m_texturePattern);
  const auto texture = ex::createTexture(
      texturePattern, options.m_textureWidth, options.m_textureHeight);

  // Step 2: Initialize RNG and create MLP with He/Kaiming initialization
  //   The RNG seed order matters: weight init first, then training data generation,
  //   matching the Python reference implementation.
  ex::Xoshiro128Plus rng{options.m_seed};
  std::cout << std::format("Initializing MLP: backbone={}, hidden={}, activation={}, bias={}",
      options.m_numBackboneLayers, options.m_hiddenLayerDim, options.m_activation,
      options.m_hasBias ? "true" : "false")
            << std::endl;
  MlpConfig<DataT> mlpConfig = initializeMlp<DataT>(options, rng);

  // Step 3: Generate training samples by randomly sampling from the texture
  std::cout << std::format("Generating {} training samples...",
      options.m_numSamples) << std::endl;
  auto [uvData, texelData] = generateTrainingData<DataT>(
      texture, options.m_numSamples, rng);

  // Step 4: Train the MLP and reconstruct the texture
  std::cout << std::format("Training: epochs={}, batch={}, lr={}, optimizer={}",
      options.m_epochs, options.m_batchSize, options.m_learningRate, options.m_optimizer)
            << std::endl;
  const ex::PixmapU8 outputTexture = trainAndReconstructTexture<DataT>(
      mlpConfig.m_layers, uvData, texelData, mlpConfig.m_hasBias, options);

  // Step 5: Save the reconstructed texture as a PPM image
  {
    std::ofstream textureOut{options.m_outputImagePath, std::ios::binary};
    if (!textureOut.is_open()) {
      std::cerr << std::format("[Error] Failed to open output file: {}", options.m_outputImagePath)
                << std::endl;
      return 1;
    }
    ex::writeAsPpm(outputTexture, textureOut);
    std::cout << std::format("Output image saved to: {}", options.m_outputImagePath) << std::endl;
  }

  std::cout << std::flush;

  return 0;
}
