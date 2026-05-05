/*!
  \file example.cpp
  \author Sho Ikeda
  \brief Texture MLP GPU training example
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This example demonstrates how to train an MLP on the GPU to reconstruct 2D texture patterns.

  Overview:
    1. Create a ground-truth texture pattern (gradient, checkerboard, etc.)
    2. Initialize an MLP with He/Kaiming weight initialization
    3. Generate random (u,v) training samples from the texture
    4. Train the MLP on GPU using mini-batch SGD/Adam/Lion optimization
    5. Reconstruct the texture by evaluating the trained MLP at every pixel
    6. Save the result as a PPM image

  The MLP learns to map 2D coordinates (u,v) -> pixel intensity, effectively
  compressing a texture into a compact neural network representation.

  Usage:
    03-texture-compression-with-input-encoding
        [--backbone-layers N] [--hidden-dim N] [--activation TYPE]
        [--epochs N] [--batch-size N] [--learning-rate F] [--optimizer TYPE]
        [--texture-width N] [--texture-height N] [--texture-pattern TYPE]
        [--output-image FILE]
        [--input-encoding TYPE] [--grid-resolution N] [--grid-feature-dim N]
        [--input-image FILE]
        [--software-linalg] [--debug] [--seed N]
*/

// Standard C++ library
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <string>
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
#include "common/mlp_layer.hpp"
#include "common/optimizer.hpp"
#include "common/pixmap.hpp"
#include "common/texture.hpp"
#include "common/utility.hpp"
#include "common/xoshiro128plus.hpp"
// C++ fallback infrastructure (includes hlsl_compat.hpp, mlp.hlsl, utility, mlp_layer)
#include "common/cpp_fallback.hpp"
#include "kernel/input_encoding_common.hlsl"
#include "kernel/texture_inference_with_encoding_common.hlsl"
#include "kernel/texture_training_with_encoding_common.hlsl"
#include "kernel/texture_inference_common.hlsl"
#include "kernel/optimizer.hlsl"

namespace {

// ============================================================================
// Input encoding
// ============================================================================

enum class InputEncoding { NONE = 0, POSITIONAL = 1, GRID = 2 };

constexpr size_t encodedInputDim(const InputEncoding enc, const size_t positionalFrequencies = 4, const size_t gridFeatureDim = 0) noexcept {
  switch (enc) {
    case InputEncoding::POSITIONAL: return 4u * positionalFrequencies;
    case InputEncoding::GRID: return gridFeatureDim;
    case InputEncoding::NONE: [[fallthrough]];
    default: return 2u;
  }
}

InputEncoding inputEncodingFromString(const std::string& s) noexcept {
  if (s == "positional") return InputEncoding::POSITIONAL;
  if (s == "grid") return InputEncoding::GRID;
  return InputEncoding::NONE;
}

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
  double m_learningRate = 0.005;
  std::string m_optimizer = "adam";
  size_t m_textureWidth = 2048;
  size_t m_textureHeight = 2048;
  std::string m_texturePattern = "checkerboard";
  std::string m_outputImagePath = "mlp-training-output.png";
  std::string m_inputImage;
  std::string m_inputEncoding = "grid";
  size_t m_positionalFrequencies = 4;
  size_t m_gridResolution = 64;
  size_t m_gridFeatureDim = 8;
  // Optimizer hyperparameters
  double m_adamBeta1 = 0.9;
  double m_adamBeta2 = 0.999;
  double m_adamEpsilon = 1e-6;
  double m_lionBeta1 = 0.9;
  double m_lionBeta2 = 0.99;
  double m_lionWeightDecay = 0.3;
  double m_lossScale = 512.0;

  bool m_useCppFallback = false;
  bool m_useSoftwareLinalg = false;
  bool m_enableDebugMode = false;
  bool m_shuffle = true;
};

auto createCommandLineParser(CliOptions& options) -> std::unique_ptr<CLI::App>
{
  auto parser = std::make_unique<CLI::App>(
      "Texture compression with input encoding - Train MLP on GPU to reconstruct texture patterns");

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

  // Optimizer hyperparameters
  parser->add_option("--adam-beta1", options.m_adamBeta1,
      "Adam first moment decay rate (default: 0.9)")
      ->default_val(options.m_adamBeta1);
  parser->add_option("--adam-beta2", options.m_adamBeta2,
      "Adam second moment decay rate (default: 0.999)")
      ->default_val(options.m_adamBeta2);
  parser->add_option("--adam-epsilon", options.m_adamEpsilon,
      "Adam epsilon for numerical stability (default: 1e-8)")
      ->default_val(options.m_adamEpsilon);
  parser->add_option("--lion-beta1", options.m_lionBeta1,
      "Lion interpolation coefficient (default: 0.9)")
      ->default_val(options.m_lionBeta1);
  parser->add_option("--lion-beta2", options.m_lionBeta2,
      "Lion momentum decay rate (default: 0.99)")
      ->default_val(options.m_lionBeta2);
  parser->add_option("--lion-weight-decay", options.m_lionWeightDecay,
      "Lion weight decay coefficient (default: 0.3)")
      ->default_val(options.m_lionWeightDecay);
  parser->add_option("--loss-scale", options.m_lossScale,
      "Loss scale factor for FP16 gradient stability (default: 1.0)")
      ->default_val(options.m_lossScale)
      ->check(CLI::PositiveNumber);

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
      "Output reconstructed image path in PNG format (default: mlp-training-output.png)")
      ->default_val(options.m_outputImagePath);

  // Input image
  parser->add_option("--input-image", options.m_inputImage,
      "Input PNG image to use as ground truth (overrides --texture-pattern)")
      ->check(CLI::ExistingFile);

  // Input encoding
  parser->add_option("--input-encoding", options.m_inputEncoding,
      "Input encoding applied to UV coordinates before the MLP (none, positional, grid)")
      ->default_val(options.m_inputEncoding)
      ->check(CLI::IsMember({"none", "positional", "grid"}));
  parser->add_option("--positional-frequencies", options.m_positionalFrequencies,
      "Number of frequency bands for positional encoding (default: 4, output dim = 4 * F)")
      ->default_val(options.m_positionalFrequencies)
      ->check(CLI::Range(static_cast<size_t>(1), static_cast<size_t>(16)));
  parser->add_option("--grid-resolution", options.m_gridResolution,
      "Grid resolution for grid encoding (default: 32)")
      ->default_val(options.m_gridResolution)
      ->check(CLI::Range(static_cast<size_t>(2), static_cast<size_t>(1024)));
  parser->add_option("--grid-feature-dim", options.m_gridFeatureDim,
      "Feature vector dimension per grid vertex (default: 4)")
      ->default_val(options.m_gridFeatureDim)
      ->check(CLI::Range(static_cast<size_t>(1), static_cast<size_t>(64)));

  // Execution mode
  parser->add_flag("--cpp-fallback", options.m_useCppFallback,
      "Use C++ fallback (mlp.hlsl compiled as C++)")
      ->default_val(options.m_useCppFallback);
  parser->add_flag("--software-linalg", options.m_useSoftwareLinalg,
      "Use software-implementation linear algebra functions on HLSL")
      ->default_val(options.m_useSoftwareLinalg);
  parser->add_flag("--debug", options.m_enableDebugMode,
      "Enable debug mode for detailed output")
      ->default_val(options.m_enableDebugMode);
  parser->add_flag("--shuffle,!--no-shuffle", options.m_shuffle,
      "Shuffle training data before training (default: true, use --no-shuffle to disable)")
      ->default_val(options.m_shuffle);

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
    Layer N:       hidden   -> output(4)                [Sigmoid — maps output to [0,1]; channel 3 is a dummy]

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
  const size_t inputDim = encodedInputDim(inputEncodingFromString(options.m_inputEncoding), options.m_positionalFrequencies, options.m_gridFeatureDim);
  configs.push_back({inputDim, options.m_hiddenLayerDim, activationType});
  for (size_t i = 1; i < options.m_numBackboneLayers; ++i) {
    configs.push_back({options.m_hiddenLayerDim, options.m_hiddenLayerDim, activationType});
  }
  configs.push_back({options.m_hiddenLayerDim, 4, ex::ActivationType::SIGMOID});

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
auto generateTrainingData(const ex::Texture3Ch& texture,
                          const size_t numSamples,
                          ex::Xoshiro128Plus& rng)
    -> std::pair<std::vector<DataT>, std::vector<DataT>>
{
  std::vector<DataT> uvData(numSamples * 2);
  std::vector<DataT> texelData(numSamples * 4);

  for (size_t i = 0; i < numSamples; ++i) {
    const float u = rng.draw();
    const float v = rng.draw();
    uvData[i * 2 + 0] = static_cast<DataT>(u);
    uvData[i * 2 + 1] = static_cast<DataT>(v);

    const auto texel = texture.sample(u, v);
    texelData[i * 4 + 0] = static_cast<DataT>(texel[0]);
    texelData[i * 4 + 1] = static_cast<DataT>(texel[1]);
    texelData[i * 4 + 2] = static_cast<DataT>(texel[2]);
    texelData[i * 4 + 3] = static_cast<DataT>(0);  // dummy, not optimized
  }

  return {std::move(uvData), std::move(texelData)};
}

// ============================================================================
// Shuffle training data
// ============================================================================

/*!
  \brief Shuffle UV and texel arrays together using Fisher-Yates algorithm.

  Uses the provided RNG for deterministic shuffling. Both arrays are shuffled
  with the same permutation so that corresponding (uv, texel) pairs remain paired.

  \param uvData     Training UV coordinates (2 or ENCODED_DIM elements per sample)
  \param texelData  Ground-truth texel values (outputDim elements per sample)
  \param rng        Random number generator for shuffle
  \param uvStride   Number of elements per UV sample (default 2)
  \param texelStride Number of elements per texel sample (default 4)
*/
template <ex::Arithmetic DataT>
auto shuffleTrainingData(std::vector<DataT>& uvData,
                         std::vector<DataT>& texelData,
                         ex::Xoshiro128Plus& rng,
                         const size_t uvStride = 2,
                         const size_t texelStride = 4) -> void
{
  const size_t numSamples = uvData.size() / uvStride;
  // Fisher-Yates shuffle
  for (size_t i = numSamples - 1; i > 0; --i) {
    const size_t j = static_cast<size_t>(rng.draw() * static_cast<float>(i + 1));
    // Swap UV
    for (size_t k = 0; k < uvStride; ++k)
      std::swap(uvData[i * uvStride + k], uvData[j * uvStride + k]);
    // Swap texel
    for (size_t k = 0; k < texelStride; ++k)
      std::swap(texelData[i * texelStride + k], texelData[j * texelStride + k]);
  }
}

// ============================================================================
// HDR to LDR conversion
// ============================================================================

/*!
  \brief Convert floating-point MLP output to 8-bit RGB.

  The MLP outputs 3 channels per pixel (RGB). Each channel is clamped to [0,1]
  and quantized to [0,255].

  \param hdr  MLP output (3 values per pixel: [r, g, b, r, g, b, ...])
  \param ldr  Target pixmap for 8-bit RGB output
*/
template <ex::Arithmetic Type>
auto mapToLdr(const std::span<const Type> hdr, ex::PixmapRgb& ldr) noexcept
{
  std::span out = ldr.data();
  const size_t numPixels = ldr.width() * ldr.height();
  for (size_t i = 0; i < numPixels; ++i) {
    using half_float::round;
    using std::round;
    using std::clamp;
    auto toU8 = [&](Type v) -> std::uint8_t {
      v = clamp(v, static_cast<Type>(0), static_cast<Type>(1));
      return static_cast<std::uint8_t>(round(v * static_cast<Type>(255)));
    };
    out[i] = {{toU8(hdr[4 * i + 0]), toU8(hdr[4 * i + 1]), toU8(hdr[4 * i + 2])}};
  }
}

// ============================================================================
// Shader kernel definitions
// ============================================================================

template <ex::Arithmetic Type>
auto buildKernelDefinitions(std::span<ex::MlpLayer<Type, Type, Type, Type>> mlpData,
                            const size_t batchSize,
                            const float learningRate,
                            const size_t weightBufferSize,
                            const size_t biasBufferSize,
                            const size_t weightChunkSize,
                            const size_t biasChunkSize,
                            const ex::MatrixLayout weightMatrixLayout,
                            const size_t matrixAlignment,
                            const size_t vectorStrideAlignment,
                            const size_t biasAlignment,
                            const bool useSoftwareLinalg,
                            const bool hasBias,
                            const InputEncoding inputEncoding = InputEncoding::NONE,
                            const size_t positionalFrequencies = 4,
                            const size_t gridResolution = 0,
                            const size_t gridBufferSize = 0,
                            const float optimizerBeta1 = 0.0f,
                            const float optimizerBeta2 = 0.0f,
                            const float optimizerEpsilon = 0.0f,
                            const float optimizerWeightDecay = 0.0f,
                            const float lossScale = 1.0f) -> std::vector<ex::OptionString>
{
  const size_t inputDim = mlpData.front().inputDimension();
  const size_t outputDim = mlpData.back().outputDimension();
  const size_t numLayers = mlpData.size();
  const size_t hiddenLayerDim = mlpData.front().outputDimension();
  const ex::ActivationType activationHidden = mlpData.front().configuration().m_activation;
  const ex::ActivationType activationLast = mlpData.back().configuration().m_activation;
  constexpr size_t numThreadsX = 32;

  std::vector<ex::OptionString> defs;
  defs.reserve(26);

  // MLP architecture
  defs.push_back(ex::createOptionString("MINIDXNN_INPUT_DIMENSION={}", inputDim));
  defs.push_back(ex::createOptionString("MINIDXNN_OUTPUT_DIMENSION={}", outputDim));
  defs.push_back(ex::createOptionString("MINIDXNN_NUM_LAYERS={}", numLayers));
  defs.push_back(ex::createOptionString("MINIDXNN_HIDDEN_LAYER_DIMENSIONS={}", hiddenLayerDim));
  defs.push_back(ex::createOptionString("MINIDXNN_HAS_BIAS={}", hasBias ? 1 : 0));
  defs.push_back(ex::createOptionString("MINIDXNN_INPUT_ENCODING={}", static_cast<int>(inputEncoding)));
  if (inputEncoding == InputEncoding::POSITIONAL) {
    defs.push_back(ex::createOptionString("MINIDXNN_POSITIONAL_ENCODING_NUM_FREQUENCIES={}", positionalFrequencies));
  }
  if (inputEncoding == InputEncoding::GRID) {
    defs.push_back(ex::createOptionString("MINIDXNN_GRID_RESOLUTION={}", gridResolution));
    defs.push_back(ex::createOptionString("MINIDXNN_GRID_BUFFER_SIZE={}", gridBufferSize));
  }
  defs.push_back(ex::createOptionString("MINIDXNN_LEARNING_RATE={}", learningRate));

  // Activation functions
  defs.push_back(ex::createOptionString("MINIDXNN_ACTIVATION_HIDDEN_TYPE={}", ex::getActivationTypeString(activationHidden)));
  defs.push_back(ex::createOptionString("MINIDXNN_ACTIVATION_LAST_TYPE={}", ex::getActivationTypeString(activationLast)));

  // Weight matrix memory layout and alignment
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", ex::toHlslMatrixLayout(weightMatrixLayout)));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_ALIGNMENT={}", matrixAlignment));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", vectorStrideAlignment));
  defs.push_back(ex::createOptionString("MINIDXNN_BIAS_VECTOR_ALIGNMENT={}", biasAlignment));

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
  defs.push_back(ex::createOptionString("MINIDXNN_LOSS_SCALE={:.10f}f", lossScale));

  return defs;
}

// ============================================================================
// GPU training and texture reconstruction
// ============================================================================

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
                                   const CliOptions& options) -> ex::PixmapRgb
{
  const bool useSoftwareLinalg = options.m_useSoftwareLinalg;
  ex::MatrixLayout weightMatrixLayout = useSoftwareLinalg ? ex::MatrixLayout::ROW_MAJOR : ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL;
  constexpr size_t weightChunkSize = ex::MATRIX_ALIGNMENT;
  constexpr size_t biasChunkSize = ex::VECTOR_ALIGNMENT;
  const auto optimizerType = ex::getOptimizerTypeFromString(options.m_optimizer);
  const InputEncoding inputEncoding = inputEncodingFromString(options.m_inputEncoding);

  // Initialize GFX context
  std::shared_ptr context = ex::createGfxContext(options.m_enableDebugMode);
  const std::filesystem::path shaderDir = ex::getComputeShaderDir();
  const std::array includeDirList = ex::getHlslIncludeDirList();

  constexpr size_t numThreadsX = 32;

  // Build D3D12 matrix/vector info lists from MLP layers
  std::vector<ex::D3D12MatrixInfo<Type>> matrixInfoList;
  matrixInfoList.reserve(mlpData.size());
  for (const auto& layer : mlpData) {
    ex::D3D12MatrixInfo<Type> info;
    info.m_srcData = layer.weightData();
    info.m_rowSize = layer.outputDimension();
    info.m_columnSize = layer.inputDimension();
    info.m_layout = weightMatrixLayout;
    matrixInfoList.push_back(info);
  }
  std::vector<ex::D3D12VectorInfo<Type>> vectorInfoList;
  vectorInfoList.reserve(mlpData.size());
  for (const auto& layer : mlpData) {
    ex::D3D12VectorInfo<Type> info;
    info.m_srcData = layer.biasData();
    vectorInfoList.push_back(info);
  }

  // Create buffers
  std::shared_ptr uvBuffer = ex::createGfxBuffer<Type>(*context, uvData);
  std::shared_ptr targetBuffer = ex::createGfxBuffer<Type>(*context, texelData);
  std::shared_ptr weightBuffer = ex::packAsD3D12MatrixBuffer<Type>(*context, matrixInfoList, true);
  weightMatrixLayout = matrixInfoList.front().m_layout;
  std::shared_ptr weightGradBuffer = ex::createGfxBuffer<Type>(*context, weightBuffer->getSize() / sizeof(Type));
  std::shared_ptr biasBuffer = ex::packAsD3D12VectorBuffer<Type>(*context, vectorInfoList);
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

  // Grid encoding buffers
  const size_t gridResolution = options.m_gridResolution;
  const size_t gridFeatureDim = options.m_gridFeatureDim;
  const size_t gridElements = gridResolution * gridResolution * gridFeatureDim;
  const size_t gridBufferSize = gridElements * sizeof(float);
  std::shared_ptr<GfxBuffer> gridBuffer;
  std::shared_ptr<GfxBuffer> gridGradBuffer;
  std::shared_ptr<GfxBuffer> gridFirstMomentBuffer;
  std::shared_ptr<GfxBuffer> gridSecondMomentBuffer;

  if (inputEncoding == InputEncoding::GRID) {
    ex::Xoshiro128Plus gridRng{options.m_seed ^ 0xA5A5A5A5u};
    std::vector<float> gridFeatures(gridElements);
    for (auto& f : gridFeatures) {
      f = (gridRng.draw() - 0.5f) * 0.02f;
    }
    gridBuffer = ex::createGfxBuffer<float>(*context, gridFeatures);
    gridGradBuffer = ex::createGfxBuffer<float>(*context, gridElements);

    if (optimizerType == ex::OptimizerType::ADAM) {
      gridFirstMomentBuffer = ex::createGfxBuffer<float>(*context, gridElements);
      gridSecondMomentBuffer = ex::createGfxBuffer<float>(*context, gridElements);
      gfxCommandClearBuffer(*context, *gridFirstMomentBuffer);
      gfxCommandClearBuffer(*context, *gridSecondMomentBuffer);
      gfxFinish(*context);
    } else if (optimizerType == ex::OptimizerType::LION) {
      gridFirstMomentBuffer = ex::createGfxBuffer<float>(*context, gridElements);
      gfxCommandClearBuffer(*context, *gridFirstMomentBuffer);
      gfxFinish(*context);
    }
  }

  // Optimizer hyperparameters (from CLI)
  const float adamBeta1 = static_cast<float>(options.m_adamBeta1);
  const float adamBeta2 = static_cast<float>(options.m_adamBeta2);
  const float adamEpsilon = static_cast<float>(options.m_adamEpsilon);
  const float lionBeta1 = static_cast<float>(options.m_lionBeta1);
  const float lionBeta2 = static_cast<float>(options.m_lionBeta2);
  const float lionWeightDecay = static_cast<float>(options.m_lionWeightDecay);

  // Create and run the training kernel
  {
    const size_t numOptElements = (inputEncoding == InputEncoding::GRID)
        ? (std::max)({weightElements, biasElements, gridElements})
        : (std::max)(weightElements, biasElements);
    const auto [optBeta1, optBeta2, optEpsilon, optWeightDecay] = [&]() -> std::tuple<float, float, float, float> {
      if (optimizerType == ex::OptimizerType::ADAM)
        return {adamBeta1, adamBeta2, adamEpsilon, 0.0f};
      else if (optimizerType == ex::OptimizerType::LION)
        return {lionBeta1, lionBeta2, 0.0f, lionWeightDecay};
      else
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }();
    const std::vector definitions = buildKernelDefinitions(mlpData, options.m_batchSize, static_cast<float>(options.m_learningRate), weightBuffer->getSize(), biasBuffer->getSize(), weightChunkSize, biasChunkSize, matrixInfoList.front().m_layout, matrixInfoList.front().m_alignment, matrixInfoList.front().m_vectorStrideAlignment, vectorInfoList.front().m_alignment, options.m_useSoftwareLinalg, hasBias, inputEncoding, options.m_positionalFrequencies, gridResolution, gridBufferSize, optBeta1, optBeta2, optEpsilon, optWeightDecay);

    std::shared_ptr program = ex::createGfxProgram(*context, "03_texture_compression_with_input_encoding", shaderDir, includeDirList);
    const ex::OptionString backwardKernelName = ex::createOptionString("trainingF{}Kernel", 8 * sizeof(Type));
    std::shared_ptr backwardKernel = ex::createGfxComputeKernel(*context, *program, backwardKernelName.data(), definitions);

    const ex::OptionString optimizationKernelName = ex::createOptionString("{}StepF{}Kernel", options.m_optimizer, 8 * sizeof(Type));
    std::shared_ptr<GfxKernel> optimizationKernel;

    const size_t numSamples = uvData.size() / 2;
    size_t timestep = 0;

    float totalTrainingKernelTimeMs = 0.0f;
    float totalOptimizerKernelTimeMs = 0.0f;

    // Persistent staging buffer for loss readback — allocated once, reused every epoch
    std::shared_ptr lossStaging = ex::createGfxBuffer<float>(*context, 1, kGfxCpuAccess_Read);

    std::cout << "Backend: GPU\n";
    std::cout << "Starting training...\n";
    for (size_t epoch = 0; epoch < options.m_epochs; ++epoch) {
      size_t numBatches = 0;

      // Clear loss once per epoch so it accumulates across all batches
      gfxCommandClearBuffer(*context, *lossBuffer);
      gfxFinish(*context);

      for (size_t batchStart = 0; batchStart < numSamples; batchStart += options.m_batchSize) {
        const size_t batchEnd = std::min(batchStart + options.m_batchSize, numSamples);
        const size_t currentBatchSize = batchEnd - batchStart;

        // Zero weight/bias/grid gradients (not lossBuffer — it accumulates for the whole epoch)
        gfxCommandClearBuffer(*context, *weightGradBuffer);
        gfxCommandClearBuffer(*context, *biasGradBuffer);
        if (inputEncoding == InputEncoding::GRID)
          gfxCommandClearBuffer(*context, *gridGradBuffer);
        gfxFinish(*context);

        const size_t batchIndex = batchStart / options.m_batchSize;
        {
          float kernelTimeMs = 0.0f;
          const size_t threadGroupSize = ex::align(currentBatchSize, numThreadsX) / numThreadsX;
          std::vector<ex::BufferBindingDataT> trainingBuffers = {
            ex::bind(*uvBuffer, "UvBuffer"),
            ex::bind(*targetBuffer, "TargetBuffer"),
            ex::bind(*weightBuffer, "WeightBuffer"),
            ex::bind(*biasBuffer, "BiasBuffer"),
            ex::bind(*weightGradBuffer, "WeightGradBuffer"),
            ex::bind(*biasGradBuffer, "BiasGradBuffer"),
            ex::bind(*logitsCacheBuffer, "LogitsCacheBuffer"),
            ex::bind(*lossBuffer, "LossBuffer"),
          };
          if (inputEncoding == InputEncoding::GRID) {
            trainingBuffers.push_back(ex::bind(*gridBuffer, "GridBuffer"));
            trainingBuffers.push_back(ex::bind(*gridGradBuffer, "GridGradBuffer"));
          }
          ex::runKernel(*context, *program, *backwardKernel, threadGroupSize,
              std::span<const ex::BufferBindingDataT>{trainingBuffers},
              {
                ex::bind(static_cast<std::int32_t>(matrixInfoList.front().m_dataSize), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
                ex::bind(static_cast<std::int32_t>((matrixInfoList.size() > 1) ? matrixInfoList.at(1).m_dataSize : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
                ex::bind(static_cast<std::int32_t>(currentBatchSize), "TEST_CURRENT_BATCH_SIZE"),
                ex::bind(static_cast<std::int32_t>(batchIndex), "TEST_BATCH_INDEX"),
                ex::bind(static_cast<std::int32_t>(biasElements * sizeof(Type)), "TEST_BIAS_STRIDE"),
              },
              kernelTimeMs);
          totalTrainingKernelTimeMs += kernelTimeMs;
        }
        numBatches++;

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
            if (inputEncoding == InputEncoding::GRID) {
              optBuffersVec.push_back(ex::bind(*gridBuffer, "RWGridBuffer"));
              optBuffersVec.push_back(ex::bind(*gridGradBuffer, "GridGradBuffer"));
              optBuffersVec.push_back(ex::bind(*gridFirstMomentBuffer, "GridFirstMoment"));
              optBuffersVec.push_back(ex::bind(*gridSecondMomentBuffer, "GridSecondMoment"));
            }
          } else if (optimizerType == ex::OptimizerType::LION) {
            optBuffersVec = {
              ex::bind(*weightBuffer, "RWWeightBuffer"),
              ex::bind(*biasBuffer, "RWBiasBuffer"),
              ex::bind(*weightGradBuffer, "WeightGradBuffer"),
              ex::bind(*biasGradBuffer, "BiasGradBuffer"),
              ex::bind(*weightFirstMomentBuffer, "WeightFirstMoment"),
              ex::bind(*biasFirstMomentBuffer, "BiasFirstMoment"),
            };
            if (inputEncoding == InputEncoding::GRID) {
              optBuffersVec.push_back(ex::bind(*gridBuffer, "RWGridBuffer"));
              optBuffersVec.push_back(ex::bind(*gridGradBuffer, "GridGradBuffer"));
              optBuffersVec.push_back(ex::bind(*gridFirstMomentBuffer, "GridFirstMoment"));
            }
          } else {
            optBuffersVec = {
              ex::bind(*weightBuffer, "RWWeightBuffer"),
              ex::bind(*biasBuffer, "RWBiasBuffer"),
              ex::bind(*weightGradBuffer, "WeightGradBuffer"),
              ex::bind(*biasGradBuffer, "BiasGradBuffer"),
            };
            if (inputEncoding == InputEncoding::GRID) {
              optBuffersVec.push_back(ex::bind(*gridBuffer, "RWGridBuffer"));
              optBuffersVec.push_back(ex::bind(*gridGradBuffer, "GridGradBuffer"));
            }
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

      // Read back accumulated epoch loss once (one GPU stall per epoch instead of per batch)
      ex::copyBuffer(*context, *lossBuffer, *lossStaging);
      const std::span epochLossSpan = ex::mapToCpu<float>(*context, *lossStaging);
      const size_t totalSamples = std::min(numSamples, numBatches * options.m_batchSize);
      const float avgLoss = epochLossSpan[0] / static_cast<float>(totalSamples);
      std::cout << std::format("Epoch [{}/{}], Loss: {:.6f}\n", epoch + 1, options.m_epochs, avgLoss);
    }
    std::cout << "Training completed!\n";
    std::cout << std::format("Training kernel time: {:.3f} ms\n", totalTrainingKernelTimeMs);
    std::cout << std::format("Optimizer kernel time: {:.3f} ms\n", totalOptimizerKernelTimeMs);
  }

  // --- Reconstruct texture using the trained MLP ---
  std::cout << "Reconstructing texture...\n";
  ex::PixmapRgb texture{options.m_textureWidth, options.m_textureHeight};
  const size_t numPixels = static_cast<size_t>(options.m_textureWidth) * static_cast<size_t>(options.m_textureHeight);
  const std::vector reconstructUv = ex::createUvData<Type>(texture.width(), texture.height());
  std::shared_ptr reconstructUvBuffer = ex::createGfxBuffer<Type>(*context, reconstructUv);
  std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(*context, numPixels * 4);

  {
    const std::vector definitions = buildKernelDefinitions(mlpData, options.m_batchSize, static_cast<float>(options.m_learningRate), weightBuffer->getSize(), biasBuffer->getSize(), weightChunkSize, biasChunkSize, matrixInfoList.front().m_layout, matrixInfoList.front().m_alignment, matrixInfoList.front().m_vectorStrideAlignment, vectorInfoList.front().m_alignment, options.m_useSoftwareLinalg, hasBias, inputEncoding, options.m_positionalFrequencies, gridResolution, gridBufferSize);
    std::shared_ptr program = ex::createGfxProgram(*context, "03_texture_compression_with_input_encoding", shaderDir, includeDirList);
    const ex::OptionString inferenceKernelName = ex::createOptionString("inferenceF{}Kernel", 8 * sizeof(Type));
    std::shared_ptr inferenceKernel = ex::createGfxComputeKernel(*context, *program, inferenceKernelName.data(), definitions);

    const size_t threadGroupSize = ex::align(numPixels, numThreadsX) / numThreadsX;
    float inferenceKernelTimeMs = 0.0f;
    std::vector<ex::BufferBindingDataT> inferenceBuffers = {
      ex::bind(*reconstructUvBuffer, "UvBuffer"),
      ex::bind(*outputBuffer, "OutputBuffer"),
      ex::bind(*weightBuffer, "WeightBuffer"),
      ex::bind(*biasBuffer, "BiasBuffer"),
    };
    if (inputEncoding == InputEncoding::GRID) {
      inferenceBuffers.push_back(ex::bind(*gridBuffer, "GridBuffer"));
    }
    ex::runKernel(*context, *program, *inferenceKernel, threadGroupSize,
        std::span<const ex::BufferBindingDataT>{inferenceBuffers},
        {
          ex::bind(static_cast<std::int32_t>(matrixInfoList.front().m_dataSize), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
          ex::bind(static_cast<std::int32_t>((matrixInfoList.size() > 1) ? matrixInfoList.at(1).m_dataSize : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
          ex::bind(static_cast<std::int32_t>(numPixels), "TEST_NUM_INFERENCE_TASKS"),
        },
        inferenceKernelTimeMs);
    std::cout << std::format("Reconstruction time: {:.3f} ms\n", inferenceKernelTimeMs);
  }

  // Read back results
  std::shared_ptr stagingOutput = ex::createGfxBuffer<Type>(*context, numPixels * 4, kGfxCpuAccess_Read);
  ex::copyBuffer(*context, *outputBuffer, *stagingOutput);
  const std::span outputData = ex::mapToCpu<Type>(*context, *stagingOutput);
  mapToLdr<Type>(outputData, texture);

  return texture;
}
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

// ============================================================================
// C++ fallback training and inference paths
// ============================================================================

#include "cpp_fallback_path.hpp"

// ============================================================================
// Unified training dispatcher
// ============================================================================

/*!
  \brief Train the MLP and reconstruct the texture.

  Dispatches to GPU or C++ fallback implementation based on build and CLI flags.
*/
template <ex::Arithmetic DataT>
auto trainAndReconstructTexture(
    std::span<ex::MlpLayer<DataT, DataT, DataT, DataT>> mlpData,
    const std::vector<DataT>& uvData,
    const std::vector<DataT>& texelData,
    const bool hasBias,
    const CliOptions& options) -> ex::PixmapRgb
{
  if (ex::isCppFallbackForced || options.m_useCppFallback) {
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

  // Step 1: Create or load ground-truth texture
  ex::Texture3Ch texture = [&]() -> ex::Texture3Ch {
    if (!options.m_inputImage.empty()) {
      auto loaded = ex::loadTextureFromPng(options.m_inputImage);
      options.m_textureWidth = loaded.width();
      options.m_textureHeight = loaded.height();
      return loaded;
    }
    std::cout << std::format("Creating {} texture ({}x{})...\n",
        options.m_texturePattern, options.m_textureWidth, options.m_textureHeight);
    const auto texturePattern = ex::getTexturePatternFromString(options.m_texturePattern);
    return ex::createTexture(texturePattern, options.m_textureWidth, options.m_textureHeight);
  }();

  // Step 2: Initialize RNG and create MLP with He/Kaiming initialization
  //   The RNG seed order matters: weight init first, then training data generation,
  //   matching the Python reference implementation.
  ex::Xoshiro128Plus rng{options.m_seed};
  std::cout << std::format("Initializing MLP: backbone={}, hidden={}, activation={}, bias={}\n",
      options.m_numBackboneLayers, options.m_hiddenLayerDim, options.m_activation,
      options.m_hasBias ? "true" : "false");
  MlpConfig<DataT> mlpConfig = initializeMlp<DataT>(options, rng);

  // Step 3: Generate training samples by randomly sampling from the texture
  std::cout << std::format("Generating {} training samples...\n",
      options.m_numSamples);
  auto [uvData, texelData] = generateTrainingData<DataT>(
      texture, options.m_numSamples, rng);

  // Step 3b: Shuffle training data if requested
  if (options.m_shuffle) {
    std::cout << "Shuffling training data...\n";
    shuffleTrainingData<DataT>(uvData, texelData, rng, 2, 4);
  }

  // Step 4: Train the MLP and reconstruct the texture
  if (inputEncodingFromString(options.m_inputEncoding) == InputEncoding::GRID) {
    std::cout << std::format("Training: epochs={}, batch={}, lr={}, optimizer={}, input-encoding={}, grid-resolution={}, grid-feature-dim={}\n",
        options.m_epochs, options.m_batchSize, options.m_learningRate, options.m_optimizer,
        options.m_inputEncoding, options.m_gridResolution, options.m_gridFeatureDim);
  } else {
    std::cout << std::format("Training: epochs={}, batch={}, lr={}, optimizer={}, input-encoding={}\n",
        options.m_epochs, options.m_batchSize, options.m_learningRate, options.m_optimizer,
        options.m_inputEncoding);
  }
  const ex::PixmapRgb outputTexture = trainAndReconstructTexture<DataT>(
      mlpConfig.m_layers, uvData, texelData, mlpConfig.m_hasBias, options);

  // Step 5: Save the reconstructed texture as a PNG image
  if (!ex::writeAsPng(outputTexture, options.m_outputImagePath)) {
    std::cerr << std::format("[Error] Failed to write output file: {}\n", options.m_outputImagePath);
    return 1;
  }
  std::cout << std::format("Output image saved to: {}\n", options.m_outputImagePath);

  return 0;
}
