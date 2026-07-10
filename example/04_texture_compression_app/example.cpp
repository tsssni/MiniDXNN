/*!
  \file example.cpp
  \author Sho Ikeda
  \brief Windowed texture MLP training application
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  Real-time windowed application that trains an MLP to reconstruct a 2D texture.
  Each frame runs a configurable number of training epochs, then reconstructs
  the texture via inference and displays the result as a fullscreen quad.

  Usage:
    04-texture-compression-app
        [--backbone-layers N] [--hidden-dim N] [--activation TYPE]
        [--epochs N] [--batch-size N] [--learning-rate F] [--optimizer TYPE]
        [--texture-width N] [--texture-height N] [--texture-pattern TYPE]
        [--input-encoding TYPE] [--positional-frequencies N]
        [--input-image FILE]
        [--epochs-per-frame N] [--window-width N] [--window-height N]
        [--software-linalg] [--debug] [--seed N]
*/

#include <algorithm>
#include <array>
#include <chrono>
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
#include <utility>
#include <vector>
// Half
#include "half.hpp"
// CLI
#include "CLI/CLI.hpp"
// GFX
#include "gfx.h"
#include "gfx_window.h"
#ifdef GFX_ENABLE_GUI
#include "gfx_imgui.h"
#endif
// Example
#include "hlsl_include_dirs.hpp"
#include "common/activation.hpp"
#include "common/gfx_utility.hpp"
#include "common/image.hpp"
#include "common/loss.hpp"
#include "common/matrix.hpp"
#include "common/mlp_layer.hpp"
#include "common/optimizer.hpp"
#include "common/pixmap.hpp"
#include "common/texture.hpp"
#include "common/utility.hpp"
#include "common/xoshiro128plus.hpp"

namespace {

// ============================================================================
// Input encoding
// ============================================================================

enum class InputEncoding { NONE = 0, POSITIONAL = 1 };

constexpr size_t encodedInputDim(const InputEncoding enc, const size_t positionalFrequencies = 4) noexcept {
  switch (enc) {
    case InputEncoding::POSITIONAL: return 4u * positionalFrequencies;
    case InputEncoding::NONE: [[fallthrough]];
    default: return 2u;
  }
}

InputEncoding inputEncodingFromString(const std::string& s) noexcept {
  if (s == "positional") return InputEncoding::POSITIONAL;
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
  size_t m_batchSize = 200000;
  size_t m_epochs = 30;
  double m_learningRate = 0.005;
  std::string m_optimizer = "adam";
  size_t m_textureWidth = 2048;
  size_t m_textureHeight = 2048;
  std::string m_texturePattern = "checkerboard";
  std::string m_inputImage;
  std::string m_inputEncoding = "positional";
  size_t m_positionalFrequencies = 4;
  double m_adamBeta1 = 0.9;
  double m_adamBeta2 = 0.999;
  double m_adamEpsilon = 1e-6;
  double m_lionBeta1 = 0.9;
  double m_lionBeta2 = 0.99;
  double m_lionWeightDecay = 0.3;
  double m_lossScale = 512.0;
  bool m_useSoftwareLinalg = false;
  bool m_enableDebugMode = false;
  bool m_shuffle = true;

  size_t m_epochsPerFrame = 1;
  size_t m_windowWidth = 1024;
  size_t m_windowHeight = 1024;
};

auto createCommandLineParser(CliOptions& options) -> std::unique_ptr<CLI::App>
{
  auto parser = std::make_unique<CLI::App>(
      "Texture compression app - Real-time windowed MLP texture training");

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

  parser->add_option("--seed", options.m_seed,
      "Random seed for xoshiro128+ (default: 987654321)")
      ->default_val(options.m_seed);
  parser->add_option("--samples", options.m_numSamples,
      "Number of training samples (default: 200000)")
      ->default_val(options.m_numSamples)
      ->check(CLI::PositiveNumber);
  parser->add_option("--batch-size", options.m_batchSize,
      "Batch size for training (default: 200000)")
      ->default_val(options.m_batchSize)
      ->check(CLI::PositiveNumber);
  parser->add_option("--epochs", options.m_epochs,
      "Total number of training epochs (default: 30)")
      ->default_val(options.m_epochs)
      ->check(CLI::PositiveNumber);
  parser->add_option("--learning-rate", options.m_learningRate,
      "Learning rate for optimizer (default: 0.005)")
      ->default_val(options.m_learningRate)
      ->check(CLI::PositiveNumber);
  parser->add_option("--optimizer", options.m_optimizer,
      "Optimizer type: sgd, adam, lion (default: adam)")
      ->default_val(options.m_optimizer)
      ->check(CLI::IsMember({"sgd", "adam", "lion"}));

  parser->add_option("--adam-beta1", options.m_adamBeta1,
      "Adam first moment decay rate (default: 0.9)")
      ->default_val(options.m_adamBeta1);
  parser->add_option("--adam-beta2", options.m_adamBeta2,
      "Adam second moment decay rate (default: 0.999)")
      ->default_val(options.m_adamBeta2);
  parser->add_option("--adam-epsilon", options.m_adamEpsilon,
      "Adam epsilon for numerical stability (default: 1e-6)")
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
      "Loss scale factor for FP16 gradient stability (default: 512)")
      ->default_val(options.m_lossScale)
      ->check(CLI::PositiveNumber);

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

  parser->add_option("--input-image", options.m_inputImage,
      "Input PNG image to use as ground truth (overrides --texture-pattern)")
      ->check(CLI::ExistingFile);

  parser->add_option("--input-encoding", options.m_inputEncoding,
      "Input encoding applied to UV coordinates before the MLP (none, positional)")
      ->default_val(options.m_inputEncoding)
      ->check(CLI::IsMember({"none", "positional"}));
  parser->add_option("--positional-frequencies", options.m_positionalFrequencies,
      "Number of frequency bands for positional encoding (default: 4)")
      ->default_val(options.m_positionalFrequencies)
      ->check(CLI::Range(static_cast<size_t>(1), static_cast<size_t>(16)));

  parser->add_flag("--software-linalg", options.m_useSoftwareLinalg,
      "Use software-implementation linear algebra functions on HLSL")
      ->default_val(options.m_useSoftwareLinalg);
  parser->add_flag("--debug", options.m_enableDebugMode,
      "Enable debug mode for detailed output")
      ->default_val(options.m_enableDebugMode);
  parser->add_flag("--shuffle,!--no-shuffle", options.m_shuffle,
      "Shuffle training data before training (default: true)")
      ->default_val(options.m_shuffle);

  parser->add_option("--epochs-per-frame", options.m_epochsPerFrame,
      "Number of training epochs to run per frame (default: 1)")
      ->default_val(options.m_epochsPerFrame)
      ->check(CLI::PositiveNumber);
  parser->add_option("--window-width", options.m_windowWidth,
      "Window width (default: 1024)")
      ->default_val(options.m_windowWidth)
      ->check(CLI::PositiveNumber);
  parser->add_option("--window-height", options.m_windowHeight,
      "Window height (default: 1024)")
      ->default_val(options.m_windowHeight)
      ->check(CLI::PositiveNumber);

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

template <ex::Arithmetic DataT>
auto initializeMlp(const CliOptions& options, ex::Xoshiro128Plus& rng)
    -> MlpConfig<DataT>
{
  const auto activationType = ex::getActivationTypeFromString(options.m_activation);

  std::vector<ex::LayerConfiguration> configs;
  const size_t inputDim = encodedInputDim(inputEncodingFromString(options.m_inputEncoding), options.m_positionalFrequencies);
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
  config.m_layers = ex::createMlp<DataT, DataT, DataT, DataT>(configs, false, rng);

  return config;
}

// ============================================================================
// Training data generation
// ============================================================================

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
    texelData[i * 4 + 3] = static_cast<DataT>(0);
  }

  return {std::move(uvData), std::move(texelData)};
}

template <ex::Arithmetic DataT>
auto shuffleTrainingData(std::vector<DataT>& uvData,
                         std::vector<DataT>& texelData,
                         ex::Xoshiro128Plus& rng,
                         const size_t uvStride = 2,
                         const size_t texelStride = 4) -> void
{
  const size_t numSamples = uvData.size() / uvStride;
  for (size_t i = numSamples - 1; i > 0; --i) {
    const size_t j = static_cast<size_t>(rng.draw() * static_cast<float>(i + 1));
    for (size_t k = 0; k < uvStride; ++k)
      std::swap(uvData[i * uvStride + k], uvData[j * uvStride + k]);
    for (size_t k = 0; k < texelStride; ++k)
      std::swap(texelData[i * texelStride + k], texelData[j * texelStride + k]);
  }
}

// ============================================================================
// Shader kernel definitions
// ============================================================================

template <ex::Arithmetic Type>
auto buildKernelDefinitions(std::span<ex::MlpLayer<Type, Type, Type, Type>> mlpData,
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
                            const float optimizerBeta1 = 0.0f,
                            const float optimizerBeta2 = 0.0f,
                            const float optimizerEpsilon = 0.0f,
                            const float optimizerWeightDecay = 0.0f,
                            const float lossScale = 1.0f,
                            const bool useWaveReducedVectorAcc = false) -> std::vector<ex::OptionString>
{
  const size_t inputDim = mlpData.front().inputDimension();
  const size_t outputDim = mlpData.back().outputDimension();
  const size_t numLayers = mlpData.size();
  const size_t hiddenLayerDim = mlpData.front().outputDimension();
  const ex::ActivationType activationHidden = mlpData.front().configuration().m_activation;
  const ex::ActivationType activationLast = mlpData.back().configuration().m_activation;
  constexpr size_t numThreadsX = 32;

  std::vector<ex::OptionString> defs;
  defs.reserve(28);

  defs.push_back(ex::createOptionString("MINIDXNN_INPUT_DIMENSION={}", inputDim));
  defs.push_back(ex::createOptionString("MINIDXNN_OUTPUT_DIMENSION={}", outputDim));
  defs.push_back(ex::createOptionString("MINIDXNN_NUM_LAYERS={}", numLayers));
  defs.push_back(ex::createOptionString("MINIDXNN_HIDDEN_LAYER_DIMENSIONS={}", hiddenLayerDim));
  defs.push_back(ex::createOptionString("MINIDXNN_HAS_BIAS={}", hasBias ? 1 : 0));
  defs.push_back(ex::createOptionString("MINIDXNN_INPUT_ENCODING={}", static_cast<int>(inputEncoding)));
  if (inputEncoding == InputEncoding::POSITIONAL) {
    defs.push_back(ex::createOptionString("MINIDXNN_POSITIONAL_ENCODING_NUM_FREQUENCIES={}", positionalFrequencies));
  }
  defs.push_back(ex::createOptionString("MINIDXNN_LEARNING_RATE={}", learningRate));

  defs.push_back(ex::createOptionString("MINIDXNN_ACTIVATION_HIDDEN_TYPE={}", ex::getActivationTypeString(activationHidden)));
  defs.push_back(ex::createOptionString("MINIDXNN_ACTIVATION_LAST_TYPE={}", ex::getActivationTypeString(activationLast)));

  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", ex::toHlslMatrixLayout(weightMatrixLayout)));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_ALIGNMENT={}", matrixAlignment));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", vectorStrideAlignment));
  defs.push_back(ex::createOptionString("MINIDXNN_BIAS_VECTOR_ALIGNMENT={}", biasAlignment));

  defs.push_back(ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", numThreadsX));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_BUFFER_SIZE={}", weightBufferSize));
  defs.push_back(ex::createOptionString("MINIDXNN_BIAS_BUFFER_SIZE={}", biasBufferSize));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_CHUNK_SIZE={}", weightChunkSize));
  defs.push_back(ex::createOptionString("MINIDXNN_BIAS_CHUNK_SIZE={}", biasChunkSize));
  defs.push_back(ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", useSoftwareLinalg ? 1 : 0));
  defs.push_back(ex::createOptionString("MINIDXNN_USE_WAVE_REDUCED_VECTOR_ACC={}", useWaveReducedVectorAcc ? 1 : 0));

  defs.push_back(ex::createOptionString("MINIDXNN_OPTIMIZER_BETA1={:.10f}f", optimizerBeta1));
  defs.push_back(ex::createOptionString("MINIDXNN_OPTIMIZER_BETA2={:.10f}f", optimizerBeta2));
  defs.push_back(ex::createOptionString("MINIDXNN_OPTIMIZER_EPSILON={:.10e}f", optimizerEpsilon));
  defs.push_back(ex::createOptionString("MINIDXNN_OPTIMIZER_WEIGHT_DECAY={:.10f}f", optimizerWeightDecay));
  defs.push_back(ex::createOptionString("MINIDXNN_LOSS_SCALE={:.10f}f", lossScale));

  return defs;
}

// ============================================================================
// Application
// ============================================================================

template <ex::Arithmetic Type>
auto runApp(const ex::Texture3Ch& texture,
            CliOptions options) -> void
{
  const bool useSoftwareLinalg = options.m_useSoftwareLinalg;
  constexpr size_t weightChunkSize = ex::MATRIX_ALIGNMENT;
  constexpr size_t biasChunkSize = ex::VECTOR_ALIGNMENT;
  const auto optimizerType = ex::getOptimizerTypeFromString(options.m_optimizer);
  InputEncoding inputEncoding = inputEncodingFromString(options.m_inputEncoding);
  const bool hasBias = options.m_hasBias;
  bool useWaveReducedVectorAcc = true;

  // ---- Generate training data ----
  ex::Xoshiro128Plus rng{options.m_seed};
  auto [uvData, texelData] = generateTrainingData<Type>(texture, options.m_numSamples, rng);
  if (options.m_shuffle)
    shuffleTrainingData<Type>(uvData, texelData, rng, 2, 4);

  // ---- Create window and GFX context ----
  GfxWindow window = gfxCreateWindow(
      static_cast<std::uint32_t>(options.m_windowWidth),
      static_cast<std::uint32_t>(options.m_windowHeight),
      "MiniDXNN - Texture Compression");
  GfxCreateContextFlags contextFlags = kGfxCreateContextFlag_EnableExperimentalShaders;
  if (options.m_enableDebugMode) {
    contextFlags |= kGfxCreateContextFlag_EnableShaderDebugging;
  }
  GfxContext gfx = gfxCreateContext(window, contextFlags);

#ifdef GFX_ENABLE_GUI
  gfxImGuiInitialize(gfx);
#endif

  const std::filesystem::path shaderDir = ex::getComputeShaderDir();
  const std::array includeDirList = ex::getHlslIncludeDirList();

  constexpr size_t numThreadsX = 32;

  // Optimizer hyperparameters
  const float adamBeta1 = static_cast<float>(options.m_adamBeta1);
  const float adamBeta2 = static_cast<float>(options.m_adamBeta2);
  const float adamEpsilon = static_cast<float>(options.m_adamEpsilon);
  const float lionBeta1 = static_cast<float>(options.m_lionBeta1);
  const float lionBeta2 = static_cast<float>(options.m_lionBeta2);
  const float lionWeightDecay = static_cast<float>(options.m_lionWeightDecay);

  // ---- Training session state (rebuilt on parameter change) ----
  MlpConfig<Type> mlpConfig;
  std::vector<ex::D3D12MatrixInfo<Type>> matrixInfoList;
  std::vector<ex::D3D12VectorInfo<Type>> vectorInfoList;
  ex::MatrixLayout weightMatrixLayout = useSoftwareLinalg ? ex::MatrixLayout::ROW_MAJOR : ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL;

  std::shared_ptr<GfxBuffer> uvBuffer;
  std::shared_ptr<GfxBuffer> targetBuffer;
  std::shared_ptr<GfxBuffer> weightBuffer;
  std::shared_ptr<GfxBuffer> weightGradBuffer;
  std::shared_ptr<GfxBuffer> biasBuffer;
  std::shared_ptr<GfxBuffer> biasGradBuffer;
  std::shared_ptr<GfxBuffer> logitsCacheBuffer;
  std::shared_ptr<GfxBuffer> lossGradBuffer;
  std::shared_ptr<GfxBuffer> lossBuffer;
  std::shared_ptr<GfxBuffer> lossStaging;
  std::shared_ptr<GfxBuffer> weightFirstMomentBuffer;
  std::shared_ptr<GfxBuffer> weightSecondMomentBuffer;
  std::shared_ptr<GfxBuffer> biasFirstMomentBuffer;
  std::shared_ptr<GfxBuffer> biasSecondMomentBuffer;
  std::shared_ptr<GfxBuffer> outputBuffer;
  std::shared_ptr<GfxBuffer> reconstructUvBuffer;

  std::shared_ptr<GfxProgram> trainingProgram;
  std::shared_ptr<GfxKernel> forwardKernel;
  std::shared_ptr<GfxKernel> backwardKernel;
  std::shared_ptr<GfxKernel> optimizationKernel;
  std::shared_ptr<GfxProgram> inferenceProgram;
  std::shared_ptr<GfxKernel> inferenceKernel;

  size_t weightElements = 0;
  size_t biasElements = 0;
  size_t numOptElements = 0;

  const size_t inferenceWidth = options.m_windowWidth;
  const size_t inferenceHeight = options.m_windowHeight;
  const size_t numPixels = inferenceWidth * inferenceHeight;

  auto initTrainingSession = [&]() {
    gfxFinish(gfx);

    ex::Xoshiro128Plus initRng{options.m_seed};
    mlpConfig = initializeMlp<Type>(options, initRng);
    auto mlpData = std::span{mlpConfig.m_layers};

    weightMatrixLayout = useSoftwareLinalg ? ex::MatrixLayout::ROW_MAJOR : ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL;

    matrixInfoList.clear();
    matrixInfoList.reserve(mlpData.size());
    for (const auto& layer : mlpData) {
      ex::D3D12MatrixInfo<Type> info;
      info.m_srcData = layer.weightData();
      info.m_rowSize = layer.outputDimension();
      info.m_columnSize = layer.inputDimension();
      info.m_layout = weightMatrixLayout;
      matrixInfoList.push_back(info);
    }
    vectorInfoList.clear();
    vectorInfoList.reserve(mlpData.size());
    for (const auto& layer : mlpData) {
      ex::D3D12VectorInfo<Type> info;
      info.m_srcData = layer.biasData();
      vectorInfoList.push_back(info);
    }

    uvBuffer = ex::createGfxBuffer<Type>(gfx, uvData);
    targetBuffer = ex::createGfxBuffer<Type>(gfx, texelData);
    weightBuffer = ex::packAsD3D12MatrixBuffer<Type>(gfx, matrixInfoList, true);
    weightMatrixLayout = matrixInfoList.front().m_layout;
    weightGradBuffer = ex::createGfxBuffer<Type>(gfx, weightBuffer->getSize() / sizeof(Type));
    biasBuffer = ex::packAsD3D12VectorBuffer<Type>(gfx, vectorInfoList);
    biasGradBuffer = ex::createGfxBuffer<Type>(gfx, biasBuffer->getSize() / sizeof(Type));
    logitsCacheBuffer = ex::createGfxBuffer<Type>(gfx, options.m_batchSize * biasBuffer->getSize() / sizeof(Type));
    lossGradBuffer = ex::createGfxBuffer<Type>(gfx, options.m_batchSize * 4);
    lossBuffer = ex::createGfxBuffer<float>(gfx, 1);
    lossStaging = ex::createGfxBuffer<float>(gfx, 1, kGfxCpuAccess_Read);

    weightElements = weightBuffer->getSize() / sizeof(Type);
    biasElements = biasBuffer->getSize() / sizeof(Type);

    weightFirstMomentBuffer.reset();
    weightSecondMomentBuffer.reset();
    biasFirstMomentBuffer.reset();
    biasSecondMomentBuffer.reset();

    if (optimizerType == ex::OptimizerType::ADAM) {
      weightFirstMomentBuffer = ex::createGfxBuffer<float>(gfx, weightElements);
      weightSecondMomentBuffer = ex::createGfxBuffer<float>(gfx, weightElements);
      biasFirstMomentBuffer = ex::createGfxBuffer<float>(gfx, biasElements);
      biasSecondMomentBuffer = ex::createGfxBuffer<float>(gfx, biasElements);
      gfxCommandClearBuffer(gfx, *weightFirstMomentBuffer);
      gfxCommandClearBuffer(gfx, *weightSecondMomentBuffer);
      gfxCommandClearBuffer(gfx, *biasFirstMomentBuffer);
      gfxCommandClearBuffer(gfx, *biasSecondMomentBuffer);
      gfxFinish(gfx);
    } else if (optimizerType == ex::OptimizerType::LION) {
      weightFirstMomentBuffer = ex::createGfxBuffer<float>(gfx, weightElements);
      biasFirstMomentBuffer = ex::createGfxBuffer<float>(gfx, biasElements);
      gfxCommandClearBuffer(gfx, *weightFirstMomentBuffer);
      gfxCommandClearBuffer(gfx, *biasFirstMomentBuffer);
      gfxFinish(gfx);
    }

    numOptElements = (std::max)(weightElements, biasElements);

    const auto [optBeta1, optBeta2, optEpsilon, optWeightDecay] = [&]() -> std::tuple<float, float, float, float> {
      if (optimizerType == ex::OptimizerType::ADAM)
        return {adamBeta1, adamBeta2, adamEpsilon, 0.0f};
      else if (optimizerType == ex::OptimizerType::LION)
        return {lionBeta1, lionBeta2, 0.0f, lionWeightDecay};
      else
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }();
    const float lossScale = static_cast<float>(options.m_lossScale);
    const std::vector trainingDefinitions = buildKernelDefinitions(mlpData, static_cast<float>(options.m_learningRate), weightBuffer->getSize(), biasBuffer->getSize(), weightChunkSize, biasChunkSize, matrixInfoList.front().m_layout, matrixInfoList.front().m_alignment, matrixInfoList.front().m_vectorStrideAlignment, vectorInfoList.front().m_alignment, options.m_useSoftwareLinalg, hasBias, inputEncoding, options.m_positionalFrequencies, optBeta1, optBeta2, optEpsilon, optWeightDecay, lossScale, useWaveReducedVectorAcc);

    trainingProgram = ex::createGfxProgram(gfx, "04_texture_compression_app", shaderDir, includeDirList);
    const ex::OptionString forwardKernelName = ex::createOptionString("trainingForwardF{}Kernel", 8 * sizeof(Type));
    forwardKernel = ex::createGfxComputeKernel(gfx, *trainingProgram, forwardKernelName.data(), trainingDefinitions);
    const ex::OptionString backwardKernelName = ex::createOptionString("trainingBackwardF{}Kernel", 8 * sizeof(Type));
    backwardKernel = ex::createGfxComputeKernel(gfx, *trainingProgram, backwardKernelName.data(), trainingDefinitions);

    const ex::OptionString optimizationKernelName = ex::createOptionString("{}StepF{}Kernel", options.m_optimizer, 8 * sizeof(Type));
    optimizationKernel = ex::createGfxComputeKernel(gfx, *trainingProgram, optimizationKernelName.data(), trainingDefinitions);

    const std::vector reconstructUv = ex::createUvData<Type>(inferenceWidth, inferenceHeight);
    reconstructUvBuffer = ex::createGfxBuffer<Type>(gfx, reconstructUv);
    outputBuffer = ex::createGfxBuffer<Type>(gfx, numPixels * 4);

    const std::vector inferenceDefinitions = buildKernelDefinitions(mlpData, static_cast<float>(options.m_learningRate), weightBuffer->getSize(), biasBuffer->getSize(), weightChunkSize, biasChunkSize, matrixInfoList.front().m_layout, matrixInfoList.front().m_alignment, matrixInfoList.front().m_vectorStrideAlignment, vectorInfoList.front().m_alignment, options.m_useSoftwareLinalg, hasBias, inputEncoding, options.m_positionalFrequencies);
    inferenceProgram = ex::createGfxProgram(gfx, "04_texture_compression_app", shaderDir, includeDirList);
    const ex::OptionString inferenceKernelName = ex::createOptionString("inferenceF{}Kernel", 8 * sizeof(Type));
    inferenceKernel = ex::createGfxComputeKernel(gfx, *inferenceProgram, inferenceKernelName.data(), inferenceDefinitions);
  };

  initTrainingSession();

  // ---- Create display resources ----
  GfxTexture displayTexture = gfxCreateTexture2D(gfx,
      static_cast<std::uint32_t>(inferenceWidth),
      static_cast<std::uint32_t>(inferenceHeight),
      DXGI_FORMAT_R16G16B16A16_FLOAT);

  // Color buffer for rendering the scene (auto-resizes with back buffer)
  GfxTexture colorBuffer = gfxCreateTexture2D(gfx, DXGI_FORMAT_R8G8B8A8_UNORM);
#ifdef GFX_ENABLE_GUI
  GfxTexture imguiBuffer = gfxCreateTexture2D(gfx, DXGI_FORMAT_R8G8B8A8_UNORM);
#endif

  GfxProgram displayProgram = gfxCreateProgram(gfx, "display", shaderDir.string().c_str());
  GfxDrawState displayDrawState;
  gfxDrawStateSetColorTarget(displayDrawState, 0, colorBuffer.getFormat());
  GfxKernel displayKernel = gfxCreateGraphicsKernel(gfx, displayProgram, displayDrawState);
  GfxSamplerState displaySampler = gfxCreateSamplerState(gfx, D3D12_FILTER_MIN_MAG_MIP_LINEAR);

#ifdef GFX_ENABLE_GUI
  GfxProgram compositeProgram = gfxCreateProgram(gfx, "composite", shaderDir.string().c_str());
  GfxKernel compositeKernel = gfxCreateGraphicsKernel(gfx, compositeProgram);
#endif

  // ---- Training state ----
  const size_t numSamples = uvData.size() / 2;
  size_t currentEpoch = 0;
  size_t timestep = 0;
  float lastEpochLoss = 0.0f;
  bool trainingEnabled = true;
  bool hasTrainedAtLeastOnce = false;
  std::vector<float> lossHistory;

  // UI-editable parameters (separate from active values)
  int uiEncoding = static_cast<int>(inputEncoding);
  int uiPositionalFrequencies = static_cast<int>(options.m_positionalFrequencies);
  int uiBatchSize = static_cast<int>(options.m_batchSize);
  int uiEpochsPerFrame = static_cast<int>(options.m_epochsPerFrame);
  InputEncoding activeEncoding = inputEncoding;
  size_t activePositionalFrequencies = options.m_positionalFrequencies;

  auto frameStart = std::chrono::high_resolution_clock::now();
  float fps = 0.0f;
  float trainingTimeMs = 0.0f;
  float inferenceTimeMs = 0.0f;
  float forwardTimeMs = 0.0f;
  float backwardTimeMs = 0.0f;
  bool captureKernelTimes = options.m_enableDebugMode;

  std::cout << "Starting windowed training application...\n";
  std::cout << std::format("Window: {}x{}, Texture: {}x{}, Epochs: {}, Epochs/frame: {}\n",
      options.m_windowWidth, options.m_windowHeight,
      options.m_textureWidth, options.m_textureHeight,
      options.m_epochs, options.m_epochsPerFrame);

  // ---- Frame loop ----
  while (!gfxWindowIsCloseRequested(window)) {
    gfxWindowPumpEvents(window);

    if (gfxWindowIsMinimized(window))
      continue;

    const auto frameBegin = std::chrono::high_resolution_clock::now();

    // --- Resize logits cache and loss-gradient buffer if batch size changed ---
    const size_t requiredLogitsSize = options.m_batchSize * biasBuffer->getSize() / sizeof(Type);
    if (logitsCacheBuffer->getSize() / sizeof(Type) < requiredLogitsSize) {
      logitsCacheBuffer = ex::createGfxBuffer<Type>(gfx, requiredLogitsSize);
    }
    const size_t requiredLossGradSize = options.m_batchSize * 4;
    if (lossGradBuffer->getSize() / sizeof(Type) < requiredLossGradSize) {
      lossGradBuffer = ex::createGfxBuffer<Type>(gfx, requiredLossGradSize);
    }

    // --- Training step ---
    if (trainingEnabled) {
      hasTrainedAtLeastOnce = true;
      const auto trainBegin = std::chrono::high_resolution_clock::now();
      forwardTimeMs = 0.0f;
      backwardTimeMs = 0.0f;

      for (size_t ep = 0; ep < options.m_epochsPerFrame; ++ep) {
        size_t numBatches = 0;

        gfxCommandClearBuffer(gfx, *lossBuffer);

        for (size_t batchStart = 0; batchStart < numSamples; batchStart += options.m_batchSize) {
          const size_t batchEnd = std::min(batchStart + options.m_batchSize, numSamples);
          const size_t currentBatchSize = batchEnd - batchStart;

          gfxCommandClearBuffer(gfx, *weightGradBuffer);
          gfxCommandClearBuffer(gfx, *biasGradBuffer);

          const size_t batchIndex = batchStart / options.m_batchSize;
          const size_t threadGroupSize = ex::align(currentBatchSize, numThreadsX) / numThreadsX;

          // Forward pass: populate logits cache, accumulate loss, write loss gradient
          {
            const std::array forwardBuffers = {
              ex::bind(*uvBuffer, "UvBuffer"),
              ex::bind(*targetBuffer, "TargetBuffer"),
              ex::bind(*weightBuffer, "WeightBuffer"),
              ex::bind(*biasBuffer, "BiasBuffer"),
              ex::bind(*logitsCacheBuffer, "LogitsCacheBuffer"),
              ex::bind(*lossBuffer, "LossBuffer"),
              ex::bind(*lossGradBuffer, "LossGradBuffer"),
            };
            float dispatchMs = 0.0f;
            ex::runKernel(gfx, *trainingProgram, *forwardKernel, threadGroupSize,
                std::span<const ex::BufferBindingDataT>{forwardBuffers},
                {
                  ex::bind(static_cast<std::int32_t>(matrixInfoList.front().m_dataSize), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
                  ex::bind(static_cast<std::int32_t>((matrixInfoList.size() > 1) ? matrixInfoList.at(1).m_dataSize : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
                  ex::bind(static_cast<std::int32_t>(currentBatchSize), "TEST_CURRENT_BATCH_SIZE"),
                  ex::bind(static_cast<std::int32_t>(batchIndex), "TEST_BATCH_INDEX"),
                  ex::bind(static_cast<std::int32_t>(options.m_batchSize), "TEST_BATCH_SIZE"),
                  ex::bind(static_cast<std::int32_t>(biasElements * sizeof(Type)), "TEST_BIAS_STRIDE"),
                },
                captureKernelTimes ? ex::OptionalRef<float>{dispatchMs} : std::nullopt);
            if (captureKernelTimes) forwardTimeMs += dispatchMs;
          }

          // Backward pass: consume logits cache and loss gradient, accumulate weight/bias gradients
          {
            const std::array backwardBuffers = {
              ex::bind(*uvBuffer, "UvBuffer"),
              ex::bind(*weightBuffer, "WeightBuffer"),
              ex::bind(*biasBuffer, "BiasBuffer"),
              ex::bind(*weightGradBuffer, "WeightGradBuffer"),
              ex::bind(*biasGradBuffer, "BiasGradBuffer"),
              ex::bind(*logitsCacheBuffer, "LogitsCacheBuffer"),
              ex::bind(*lossGradBuffer, "LossGradBuffer"),
            };
            float dispatchMs = 0.0f;
            ex::runKernel(gfx, *trainingProgram, *backwardKernel, threadGroupSize,
                std::span<const ex::BufferBindingDataT>{backwardBuffers},
                {
                  ex::bind(static_cast<std::int32_t>(matrixInfoList.front().m_dataSize), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
                  ex::bind(static_cast<std::int32_t>((matrixInfoList.size() > 1) ? matrixInfoList.at(1).m_dataSize : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
                  ex::bind(static_cast<std::int32_t>(currentBatchSize), "TEST_CURRENT_BATCH_SIZE"),
                  ex::bind(static_cast<std::int32_t>(batchIndex), "TEST_BATCH_INDEX"),
                  ex::bind(static_cast<std::int32_t>(options.m_batchSize), "TEST_BATCH_SIZE"),
                  ex::bind(static_cast<std::int32_t>(biasElements * sizeof(Type)), "TEST_BIAS_STRIDE"),
                },
                captureKernelTimes ? ex::OptionalRef<float>{dispatchMs} : std::nullopt);
            if (captureKernelTimes) backwardTimeMs += dispatchMs;
          }
          numBatches++;

          // Optimizer step
          {
            ++timestep;
            const size_t optThreadGroupSize = ex::align(numOptElements, numThreadsX) / numThreadsX;

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

            ex::runKernel(gfx, *trainingProgram, *optimizationKernel, optThreadGroupSize,
                std::span<const ex::BufferBindingDataT>{optBuffersVec},
                { ex::bind(static_cast<std::int32_t>(timestep), "OptimizerTimestep") });
          }
        }

        currentEpoch++;

        // Read back loss only on the last epoch of this frame
        if (ep + 1 == options.m_epochsPerFrame) {
          ex::copyBuffer(gfx, *lossBuffer, *lossStaging);
          const std::span epochLossSpan = ex::mapToCpu<float>(gfx, *lossStaging);
          const size_t totalSamples = std::min(numSamples, numBatches * options.m_batchSize);
          lastEpochLoss = epochLossSpan[0] / static_cast<float>(totalSamples);
          lossHistory.push_back(std::log10(std::max(lastEpochLoss, 1e-10f)));
          std::cout << std::format("Epoch [{}], Loss: {:.6f}\n", currentEpoch, lastEpochLoss);
        }
      }

      const auto trainEnd = std::chrono::high_resolution_clock::now();
      trainingTimeMs = static_cast<float>(std::chrono::duration<double, std::milli>(trainEnd - trainBegin).count());
    }

    // --- Inference and display ---
    if (hasTrainedAtLeastOnce) {
      // Reconstruct texture from trained MLP
      const size_t threadGroupSize = ex::align(numPixels, numThreadsX) / numThreadsX;
      std::vector<ex::BufferBindingDataT> inferenceBuffers = {
        ex::bind(*reconstructUvBuffer, "UvBuffer"),
        ex::bind(*outputBuffer, "OutputBuffer"),
        ex::bind(*weightBuffer, "WeightBuffer"),
        ex::bind(*biasBuffer, "BiasBuffer"),
      };
      ex::runKernel(gfx, *inferenceProgram, *inferenceKernel, threadGroupSize,
          std::span<const ex::BufferBindingDataT>{inferenceBuffers},
          {
            ex::bind(static_cast<std::int32_t>(matrixInfoList.front().m_dataSize), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
            ex::bind(static_cast<std::int32_t>((matrixInfoList.size() > 1) ? matrixInfoList.at(1).m_dataSize : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
            ex::bind(static_cast<std::int32_t>(numPixels), "TEST_NUM_INFERENCE_TASKS"),
          },
          inferenceTimeMs);

      gfxCommandCopyBufferToTexture(gfx, displayTexture, *outputBuffer);

      const float texelSize[2] = {
        1.0f / static_cast<float>(gfxGetBackBufferWidth(gfx)),
        1.0f / static_cast<float>(gfxGetBackBufferHeight(gfx))
      };
      gfxProgramSetParameter(gfx, displayProgram, "g_Texture", displayTexture);
      gfxProgramSetParameter(gfx, displayProgram, "g_Sampler", displaySampler);
      gfxProgramSetParameter(gfx, displayProgram, "g_TexelSize", texelSize);

      gfxCommandBindColorTarget(gfx, 0, colorBuffer);
      gfxCommandBindKernel(gfx, displayKernel);
      gfxCommandDraw(gfx, 3);
    } else {
      gfxCommandClearTexture(gfx, colorBuffer);
    }

    // --- ImGui overlay ---
#ifdef GFX_ENABLE_GUI
    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.7f);
    ImGui::Begin("Training Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings);
    const InputEncoding uiEncodingEnum = static_cast<InputEncoding>(uiEncoding);
    const bool encodingParamsChanged =
        uiEncodingEnum != activeEncoding ||
        (uiEncodingEnum == InputEncoding::POSITIONAL &&
         static_cast<size_t>(uiPositionalFrequencies) != activePositionalFrequencies);

    if (trainingEnabled) {
      if (ImGui::Button("Pause Training"))
        trainingEnabled = false;
    } else {
      if (ImGui::Button("Start Training")) {
        if (encodingParamsChanged) {
          inputEncoding = uiEncodingEnum;
          options.m_inputEncoding = (inputEncoding == InputEncoding::POSITIONAL) ? "positional" : "none";
          options.m_positionalFrequencies = static_cast<size_t>(uiPositionalFrequencies);
          activeEncoding = inputEncoding;
          activePositionalFrequencies = options.m_positionalFrequencies;
          initTrainingSession();
          currentEpoch = 0;
          timestep = 0;
          lastEpochLoss = 0.0f;
          hasTrainedAtLeastOnce = false;
          lossHistory.clear();
        }
        trainingEnabled = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Reset")) {
        initTrainingSession();
        currentEpoch = 0;
        timestep = 0;
        lastEpochLoss = 0.0f;
        hasTrainedAtLeastOnce = false;
        lossHistory.clear();
      }
    }
    ImGui::Separator();
    ImGui::BeginDisabled(trainingEnabled);
    const char* encodingNames[] = {"None", "Positional"};
    ImGui::Combo("Encoding", &uiEncoding, encodingNames, 2);
    if (uiEncodingEnum == InputEncoding::POSITIONAL) {
      ImGui::InputInt("Frequencies", &uiPositionalFrequencies);
      uiPositionalFrequencies = std::clamp(uiPositionalFrequencies, 1, 16);
    }
    if (ImGui::Checkbox("Wave-reduced bias accum", &useWaveReducedVectorAcc)) {
      initTrainingSession();
      currentEpoch = 0;
      timestep = 0;
      lastEpochLoss = 0.0f;
      hasTrainedAtLeastOnce = false;
      lossHistory.clear();
    }
    ImGui::EndDisabled();
    ImGui::InputInt("Batch Size", &uiBatchSize);
    uiBatchSize = std::clamp(uiBatchSize, 1, static_cast<int>(numSamples));
    options.m_batchSize = static_cast<size_t>(uiBatchSize);
    ImGui::InputInt("Epochs/Frame", &uiEpochsPerFrame);
    uiEpochsPerFrame = std::clamp(uiEpochsPerFrame, 1, 1000);
    options.m_epochsPerFrame = static_cast<size_t>(uiEpochsPerFrame);
    if (encodingParamsChanged) {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Encoding changed - will reinitialize on Start");
    }
    ImGui::Separator();
    ImGui::Text("Epoch: %zu", currentEpoch);
    ImGui::Text("Loss: %.6f", lastEpochLoss);
    if (!lossHistory.empty()) {
      const auto [minIt, maxIt] = std::minmax_element(lossHistory.begin(), lossHistory.end());
      const float plotMin = *minIt - 0.1f;
      const float plotMax = *maxIt + 0.1f;
      ImGui::PlotLines("##LossGraph", lossHistory.data(),
          static_cast<int>(lossHistory.size()), 0,
          "Loss (log10)", plotMin, plotMax, ImVec2(0.0f, 80.0f));
    }
    ImGui::Text("Training: %.1f ms/frame", trainingTimeMs);
    ImGui::Text("Inference: %.1f ms", inferenceTimeMs);
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Checkbox("Capture kernel times (GPU)", &captureKernelTimes);
    if (captureKernelTimes) {
      ImGui::Text("  Forward:  %.3f ms/frame", forwardTimeMs);
      ImGui::Text("  Backward: %.3f ms/frame", backwardTimeMs);
    }
    ImGui::Separator();
    ImGui::Text("Texture: %zux%zu", options.m_textureWidth, options.m_textureHeight);
    ImGui::Text("Optimizer: %s", options.m_optimizer.c_str());
    ImGui::End();

    gfxImGuiRender(imguiBuffer);
    const float compositeRes[2] = {
      static_cast<float>(gfxGetBackBufferWidth(gfx)),
      static_cast<float>(gfxGetBackBufferHeight(gfx))
    };
    gfxProgramSetParameter(gfx, compositeProgram, "g_SceneBuffer", colorBuffer);
    gfxProgramSetParameter(gfx, compositeProgram, "g_ImGuiBuffer", imguiBuffer);
    gfxProgramSetParameter(gfx, compositeProgram, "g_Resolution", compositeRes);
    gfxCommandBindKernel(gfx, compositeKernel);
    gfxCommandDraw(gfx, 3);
#else
    gfxCommandCopyTextureToBackBuffer(gfx, colorBuffer);
#endif

    // --- Present ---
    gfxFrame(gfx);

    // --- Measure FPS ---
    const auto frameEnd = std::chrono::high_resolution_clock::now();
    const float frameDurationMs = static_cast<float>(std::chrono::duration<double, std::milli>(frameEnd - frameBegin).count());
    fps = (frameDurationMs > 0.0f) ? (1000.0f / frameDurationMs) : 0.0f;
  }

  // ---- Cleanup ----
  gfxDestroyTexture(gfx, displayTexture);
  gfxDestroyTexture(gfx, colorBuffer);
  gfxDestroySamplerState(gfx, displaySampler);
  gfxDestroyKernel(gfx, displayKernel);
  gfxDestroyProgram(gfx, displayProgram);

#ifdef GFX_ENABLE_GUI
  gfxDestroyKernel(gfx, compositeKernel);
  gfxDestroyProgram(gfx, compositeProgram);
  gfxDestroyTexture(gfx, imguiBuffer);
  gfxImGuiTerminate();
#endif
  gfxDestroyContext(gfx);
  gfxDestroyWindow(window);
}

} // namespace

// ============================================================================
// Entry point
// ============================================================================

auto main(const int argc, const char** argv) -> int
{
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
      options.m_numSamples = options.m_textureWidth * options.m_textureHeight;
      options.m_batchSize = options.m_numSamples;
      return loaded;
    }
    std::cout << std::format("Creating {} texture ({}x{})...\n",
        options.m_texturePattern, options.m_textureWidth, options.m_textureHeight);
    const auto texturePattern = ex::getTexturePatternFromString(options.m_texturePattern);
    return ex::createTexture(texturePattern, options.m_textureWidth, options.m_textureHeight);
  }();

  // Step 2: Run windowed application
  std::cout << std::format("MLP: backbone={}, hidden={}, activation={}, bias={}\n",
      options.m_numBackboneLayers, options.m_hiddenLayerDim, options.m_activation,
      options.m_hasBias ? "true" : "false");
  std::cout << std::format("Training: batch={}, lr={}, optimizer={}, encoding={}\n",
      options.m_batchSize, options.m_learningRate, options.m_optimizer, options.m_inputEncoding);

  runApp<DataT>(texture, options);

  return 0;
}
