/*!
  \file example.cpp
  \author Sho Ikeda
  \brief Texture MLP inference example
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This example demonstrates how to use MiniDXNN to run MLP inference on the GPU.

  Overview:
    1. Load a pre-trained MLP from a binary file (produced by 02_texture_training)
    2. Generate normalized (u,v) texture coordinates for every pixel
    3. Run MLP inference (GPU compute shader or CPU fallback) to predict pixel values
    4. Save the reconstructed texture as a PPM image

  The MLP encodes a 2D texture as a compact neural network: given (u,v) coordinates
  as input, it outputs the corresponding pixel intensity. This is a form of neural
  texture compression.

  Usage:
    01-texture-inference <mlp-binary> [output.ppm]
        [--texture-width N] [--texture-height N]
        [--cpu] [--software-linalg] [--debug]
*/

// Standard C++ library
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <format>
#include <fstream>
#include <istream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
// Half
#include "half.hpp"
// CLI
#include "CLI/CLI.hpp"
// Example
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
#include "hlsl_include_dirs.hpp"
#include "common/gfx_utility.hpp"
#endif
#include "common/image.hpp"
#include "common/pixmap.hpp"
// C++ fallback infrastructure (includes hlsl_compat.hpp, mlp.hlsl, utility, mlp_layer)
#include "common/cpp_fallback.hpp"
#include "kernel/texture_inference_common.hlsl"

namespace {

// ============================================================================
// Command-line options
// ============================================================================

struct CliOptions
{
  std::string m_mlpBinPath = "texture-mlp-data.bin";
  std::string m_outputPath = "mlp-inference-output.ppm";
  size_t m_textureWidth = 4096;
  size_t m_textureHeight = 4096;
  bool m_useReferenceMlpOperations = false;
  bool m_useCppFallback = false;
  bool m_useSoftwareLinalg = false;
  bool m_enableDebugMode = false;
};

auto createCommandLineParser(CliOptions& options) -> std::unique_ptr<CLI::App>
{
  auto parser = std::make_unique<CLI::App>(
      "Texture MLP inference - Reconstruct texture from a trained MLP");

  parser->add_option("mlp-binary", options.m_mlpBinPath, "Path to MLP binary file")
      ->required()
      ->check(CLI::ExistingFile);
  parser->add_option("output", options.m_outputPath, "Output image path (PPM format)")
      ->default_val(options.m_outputPath);
  parser->add_option("--texture-width", options.m_textureWidth, "Texture width in pixels")
      ->default_val(options.m_textureWidth)
      ->check(CLI::PositiveNumber);
  parser->add_option("--texture-height", options.m_textureHeight, "Texture height in pixels")
      ->default_val(options.m_textureHeight)
      ->check(CLI::PositiveNumber);
  parser->add_flag("--cpu", options.m_useReferenceMlpOperations,
      "Use CPU reference ML operations instead of GPU")
      ->default_val(options.m_useReferenceMlpOperations);
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
  std::vector<ex::MlpLayer<Type, Type>> m_layers;
};

// ============================================================================
// MLP loading
// ============================================================================

/*!
  \brief Load a pre-trained MLP from a binary stream.

  Binary format (produced by 02_texture_training):
    Header (12 bytes):
      uint32  numBackboneLayers   Number of hidden layers
      uint32  hiddenLayerDim      Neurons per hidden layer
      int32   activationType      Activation function (see ex::ActivationType)

    Per-layer data (numBackboneLayers + 1 layers):
      float32[outputDim * inputDim]   Weight matrix (row-major)
      float32[outputDim]              Bias vector

  Layer architecture:
    Layer 0:       input(2) -> hidden(hiddenLayerDim)   [user-specified activation]
    Layer 1..N-1:  hidden   -> hidden                   [user-specified activation]
    Layer N:       hidden   -> output(2)                [Sigmoid — maps output to [0,1]]

  Weights are stored as float32 in the binary file and converted to the target
  Type (typically half-precision) for efficient GPU inference.

  After loading, m_hasBias is automatically set to false if all bias values
  across all layers are zero.
*/
template <ex::Arithmetic Type>
auto loadMlp(std::istream& bin) -> MlpConfig<Type>
{
  // Read the MLP header
  std::uint32_t numBackboneLayers = 0;
  std::uint32_t hiddenLayerDim = 0;
  std::int32_t activationInt = 0;
  ex::read(&numBackboneLayers, bin);
  ex::read(&hiddenLayerDim, bin);
  ex::read(&activationInt, bin);

  if (!bin.good()) {
    std::cerr << "[Error] Failed to read header." << std::endl;
    std::abort();
  }

  MlpConfig<Type> config;
  config.m_numBackboneLayers = numBackboneLayers;
  config.m_hiddenLayerDim = hiddenLayerDim;
  config.m_activation = static_cast<ex::ActivationType>(activationInt);

  std::cout << std::format("MLP Configuration:\n"
      "  numBackboneLayers: {}\n"
      "  hiddenLayerDim: {}\n"
      "  activation: {}\n",
      config.m_numBackboneLayers, config.m_hiddenLayerDim, activationInt);

  // Read each layer's weight matrix and bias vector, converting float32 -> Type
  using MlpLayerT = ex::MlpLayer<Type, Type>;
  config.m_layers.reserve(numBackboneLayers + 1);
  const auto addLayer = [&bin, &config](const ex::LayerConfiguration& layerConfig)
  {
    MlpLayerT& layer = config.m_layers.emplace_back(layerConfig);
    std::vector<float> buffer;
    // Weights: float32[outputDim * inputDim] -> Type
    {
      const size_t size = layerConfig.m_inputDim * layerConfig.m_outputDim;
      buffer.resize(size);
      ex::read(buffer.data(), bin, static_cast<std::streamsize>(sizeof(float) * size));
      std::ranges::transform(buffer, layer.weightData().begin(), [](const float v) -> Type {
        return static_cast<Type>(v);
      });
    }
    // Bias: float32[outputDim] -> Type
    {
      const size_t size = layerConfig.m_outputDim;
      buffer.resize(size);
      ex::read(buffer.data(), bin, static_cast<std::streamsize>(sizeof(float) * size));
      std::ranges::transform(buffer, layer.biasData().begin(), [](const float v) -> Type {
        return static_cast<Type>(v);
      });
    }
  };

  // Build the MLP: input layer -> backbone hidden layers -> output layer
  addLayer(ex::LayerConfiguration{2, hiddenLayerDim, config.m_activation});
  for (size_t i = 0; i < numBackboneLayers - 1; ++i) {
    addLayer(ex::LayerConfiguration{hiddenLayerDim, hiddenLayerDim, config.m_activation});
  }
  addLayer(ex::LayerConfiguration{hiddenLayerDim, 2, ex::ActivationType::SIGMOID});

  if (!bin.good()) {
    std::cerr << "[Error] Failed to read data" << std::endl;
    std::abort();
  }

  // Auto-detect bias: if all bias values across all layers are zero, disable bias
  config.m_hasBias = std::ranges::any_of(config.m_layers, [](const MlpLayerT& layer) {
    return std::ranges::any_of(layer.biasData(), [](const auto& b) {
      return static_cast<float>(b) != 0.0f;
    });
  });
  std::cout << std::format("  hasBias: {}\n", config.m_hasBias ? "true" : "false");

  return config;
}

// ============================================================================
// UV coordinate generation
// ============================================================================

/*!
  \brief Generate normalized UV coordinates for every pixel in the texture.

  Creates a flat array of interleaved (u, v) pairs where u,v in [0,1], ordered
  row-by-row from top-left to bottom-right. These serve as MLP input — each
  (u,v) pair asks the network "what is the pixel value at this position?"

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
// GPU inference
// ============================================================================

//! Number of threads per workgroup for the compute shader dispatch
[[maybe_unused]] constexpr size_t kNumThreadsX = 32;

/*!
  \brief Build compile-time shader definitions for the MLP inference kernel.

  These definitions are passed to the HLSL compiler and configure the MLP
  architecture, memory layout, and dispatch parameters at compile time. This
  allows the GPU kernel to be fully specialized for the loaded model.
*/
template <ex::Arithmetic Type>
auto buildKernelDefinitions(const std::span<const ex::MlpLayer<Type, Type>> mlpData,
                            const size_t numTasks,
                            const ex::MatrixLayout weightMatrixLayout,
                            const bool useSoftwareLinalg,
                            const bool hasBias) -> std::vector<ex::OptionString>
{
  const size_t inputDim = mlpData.front().inputDimension();
  const size_t outputDim = mlpData.back().outputDimension();
  const size_t numLayers = mlpData.size();
  const size_t hiddenLayerDim = mlpData.front().outputDimension();
  const auto activationHidden = mlpData.front().configuration().m_activation;
  const auto activationLast = mlpData.back().configuration().m_activation;

  std::vector<ex::OptionString> defs;
  defs.reserve(14);

  // MLP architecture
  defs.push_back(ex::createOptionString("MINIDXNN_INPUT_DIMENSION={}", inputDim));
  defs.push_back(ex::createOptionString("MINIDXNN_OUTPUT_DIMENSION={}", outputDim));
  defs.push_back(ex::createOptionString("MINIDXNN_NUM_LAYERS={}", numLayers));
  defs.push_back(ex::createOptionString("MINIDXNN_HIDDEN_LAYER_DIMENSIONS={}", hiddenLayerDim));
  defs.push_back(ex::createOptionString("MINIDXNN_HAS_BIAS={}", hasBias ? 1 : 0));

  // Activation functions
  defs.push_back(ex::createOptionString("MINIDXNN_ACTIVATION_HIDDEN_TYPE={}", ex::getActivationTypeString(activationHidden)));
  defs.push_back(ex::createOptionString("MINIDXNN_ACTIVATION_LAST_TYPE={}", ex::getActivationTypeString(activationLast)));

  // Weight matrix memory layout and alignment
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", static_cast<int>(weightMatrixLayout)));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_ALIGNMENT={}", ex::MATRIX_ALIGNMENT));
  defs.push_back(ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", ex::MATRIX_VECTOR_STRIDE_ALIGNMENT));
  defs.push_back(ex::createOptionString("MINIDXNN_BIAS_VECTOR_ALIGNMENT={}", ex::VECTOR_ALIGNMENT));

  // Dispatch configuration
  defs.push_back(ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", kNumThreadsX));
  defs.push_back(ex::createOptionString("MINIDXNN_NUM_TASKS={}", numTasks));

  // Use software linear algebra instead of hardware cooperative matrix intrinsics
  defs.push_back(ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", useSoftwareLinalg ? 1 : 0));

  return defs;
}

/*!
  \brief Run MLP inference on the GPU using a DirectX compute shader.

  This function orchestrates the full GPU inference pipeline:
    1. Initialize the GFX context and compile the compute shader
    2. Upload MLP weights/biases and UV coordinates to GPU buffers
    3. Dispatch the inference kernel
    4. Read back results and convert to 8-bit grayscale

  The compute shader (01_texture_inference.comp) processes all pixels in parallel,
  with each thread evaluating the MLP for one (u,v) coordinate.
*/
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
template <ex::Arithmetic Type>
auto runGpuInference(const MlpConfig<Type>& mlpConfig,
                     const std::vector<Type>& uvData,
                     const CliOptions& options,
                     ex::PixmapU8& texture) -> void
{
  const std::span mlpData = std::span{mlpConfig.m_layers};
  const bool hasBias = mlpConfig.m_hasBias;
  const size_t numTasks = texture.width() * texture.height();
  const ex::MatrixLayout weightMatrixLayout = ex::MatrixLayout::ROW_MAJOR;

  // Step 1: Create GFX context and compile the compute shader program.
  //   The shader is compiled at runtime with model-specific definitions,
  //   allowing the kernel to be specialized for the loaded MLP architecture.
  std::shared_ptr context = ex::createGfxContext(options.m_enableDebugMode);
  const std::filesystem::path shaderDir = ex::getComputeShaderDir();
  const std::array includeDirList = ex::getHlslIncludeDirList();
  std::shared_ptr program = ex::createGfxProgram(*context, "01_texture_inference", shaderDir, includeDirList);

  // Step 2: Upload data to GPU buffers.
  //   Weight matrices and bias vectors are packed with proper alignment required
  //   by dx::linalg cooperative matrix operations on the GPU.
  std::vector<size_t> matrixSizeList(mlpData.size());
  std::shared_ptr uvBuffer = ex::createGfxBuffer<Type>(*context, uvData);
  std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(*context, 2 * numTasks);
  std::shared_ptr weightBuffer = ex::convertToMatrixBuffer<Type>(*context, mlpData, weightMatrixLayout, matrixSizeList, ex::MATRIX_ALIGNMENT, ex::MATRIX_VECTOR_STRIDE_ALIGNMENT);
  std::shared_ptr biasBuffer = ex::convertToVectorBuffer<Type>(*context, mlpData, ex::VECTOR_ALIGNMENT);

  // Step 3: Configure and dispatch the inference compute kernel.
  const std::vector definitions = buildKernelDefinitions<Type>(mlpData, numTasks, weightMatrixLayout, options.m_useSoftwareLinalg, hasBias);
  const ex::OptionString kernelName = ex::createOptionString("inferenceF{}Kernel", 8 * sizeof(Type));
  std::shared_ptr kernel = ex::createGfxComputeKernel(*context, *program, kernelName.data(), definitions);

  const size_t threadGroupSize = numTasks / kNumThreadsX;
  gfxFinish(*context);

  float kernelTimeInMs = 0.0f;
  ex::runKernel(*context, *program, *kernel, threadGroupSize,
      {
        ex::bind(*uvBuffer, "UvBuffer"),
        ex::bind(*outputBuffer, "OutputBuffer"),
        ex::bind(*weightBuffer, "WeightBuffer"),
        ex::bind(*biasBuffer, "BiasBuffer"),
      },
      {
        ex::bind(static_cast<std::int32_t>(matrixSizeList.front()), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
        ex::bind(static_cast<std::int32_t>((matrixSizeList.size() > 1) ? matrixSizeList.at(1) : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
      },
      kernelTimeInMs);
  std::cout << std::format("Inference (shader) time: {:.3f} ms", kernelTimeInMs)
            << std::endl;

  // Step 4: Read back GPU results to CPU and convert to 8-bit grayscale.
  std::shared_ptr staging = ex::createGfxBuffer<Type>(*context, 2 * numTasks, kGfxCpuAccess_Read);
  ex::copyBuffer(*context, *outputBuffer, *staging);
  const std::span output = ex::mapToCpu<Type>(*context, *staging);
  mapToLdr<Type>(output, texture);
}
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

// ============================================================================
// C++ fallback inference (mlp.hlsl compiled as C++)
// ============================================================================

// Templated forward kernel: delegates to texkernel::inferenceStep from shared HLSL.
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          typename ActivationHiddenT, typename ActivationLastT>
void cppFallbackForwardKernel(const ex::PackedMlpBuffers<Type>& packed,
                              const std::vector<Type>& uvData,
                              std::vector<Type>& output,
                              size_t numTasks)
{
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
            mininn::impl::TypeTraits<Type>::COMPONENT_TYPE,
            dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
            ActivationHiddenT, ActivationLastT,
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

// Dispatch hidden-layer activation type at runtime.
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM>
bool dispatchActivation(ex::ActivationType hiddenAct,
                        const ex::PackedMlpBuffers<Type>& packed,
                        const std::vector<Type>& uvData,
                        std::vector<Type>& output,
                        size_t numTasks)
{
  // Last activation is always Sigmoid for this example.
  using Sigmoid = mininn::SigmoidActivation;
  switch (hiddenAct) {
    case ex::ActivationType::RELU:
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, mininn::ReluActivation, Sigmoid>(packed, uvData, output, numTasks);
      return true;
    case ex::ActivationType::IDENTITY:
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, mininn::IdentityActivation, Sigmoid>(packed, uvData, output, numTasks);
      return true;
    case ex::ActivationType::SIGMOID:
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, mininn::SigmoidActivation, Sigmoid>(packed, uvData, output, numTasks);
      return true;
    case ex::ActivationType::LEAKY_RELU:
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, mininn::LeakyReluActivation, Sigmoid>(packed, uvData, output, numTasks);
      return true;
    case ex::ActivationType::TANH:
    default:
      return false;
  }
}

// Dispatch NUM_LAYERS and HIDDEN_DIM at runtime.
// Supports common MLP configurations used in texture inference.
template <ex::Arithmetic Type>
bool dispatchForward(size_t numLayers, size_t hiddenDim,
                     ex::ActivationType hiddenAct,
                     const ex::PackedMlpBuffers<Type>& packed,
                     const std::vector<Type>& uvData,
                     std::vector<Type>& output,
                     size_t numTasks)
{
  // Macro to reduce boilerplate for each (NUM_LAYERS, HIDDEN_DIM) pair
  #define DISPATCH_CASE(NL, HD) \
    if (numLayers == (NL) && hiddenDim == (HD)) \
      return dispatchActivation<Type, (NL), (HD)>(hiddenAct, packed, uvData, output, numTasks);

  // 2 layers (1 backbone): common small models
  DISPATCH_CASE(2, 8)   DISPATCH_CASE(2, 16)
  DISPATCH_CASE(2, 32)  DISPATCH_CASE(2, 64)
  // 3 layers (2 backbone)
  DISPATCH_CASE(3, 8)   DISPATCH_CASE(3, 16)
  DISPATCH_CASE(3, 32)  DISPATCH_CASE(3, 64)
  // 4 layers (3 backbone)
  DISPATCH_CASE(4, 16)  DISPATCH_CASE(4, 32)
  DISPATCH_CASE(4, 64)
  // 5 layers (4 backbone)
  DISPATCH_CASE(5, 16)  DISPATCH_CASE(5, 32)
  DISPATCH_CASE(5, 64)

  #undef DISPATCH_CASE
  return false;
}

/*!
  \brief Run inference using the C++ fallback path (mlp.hlsl compiled as C++).

  Creates ByteAddressBuffers from packed MLP layer data and calls mininn::forward.
*/
template <ex::Arithmetic Type>
auto runCppFallbackInference(const MlpConfig<Type>& mlpConfig,
                             const std::vector<Type>& uvData,
                             ex::PixmapU8& texture) -> void
{
  const std::span mlpData = std::span{mlpConfig.m_layers};
  const bool hasBias = mlpConfig.m_hasBias;
  const size_t numTasks = texture.width() * texture.height();
  const size_t numLayers = mlpData.size();
  const size_t hiddenDim = mlpData.front().outputDimension();
  const auto hiddenAct = mlpData.front().configuration().m_activation;

  ex::PackedMlpBuffers<Type> packed;
  packed.pack(mlpData, hasBias);

  std::vector<Type> output(numTasks * 2);

  const auto startTime = std::chrono::high_resolution_clock::now();
  if (!dispatchForward<Type>(numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks)) {
    std::cerr << std::format("[Error] C++ fallback: unsupported MLP config (layers={}, hiddenDim={})\n",
        numLayers, hiddenDim);
    std::abort();
  }
  const auto endTime = std::chrono::high_resolution_clock::now();
  const auto elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  std::cout << std::format("Reconstruction time: {:.3f} ms", elapsedMs) << std::endl;

  mapToLdr<Type>(output, texture);
}

// ============================================================================

/*!
  \brief Reconstruct a texture by running MLP inference over all pixel coordinates.

  Supports two execution paths:
    - GPU (default): Uses a DirectX compute shader for high-throughput inference
    - CPU (--cpu flag): Reference implementation using C++ matrix operations
*/
template <ex::Arithmetic Type>
auto reconstructTexture(const MlpConfig<Type>& mlpConfig, const CliOptions& options) -> ex::PixmapU8
{
  ex::PixmapU8 texture{options.m_textureWidth, options.m_textureHeight};
  const std::vector uvData = createUvData<Type>(texture.width(), texture.height());

  if (options.m_useReferenceMlpOperations) {
    std::cout << "Backend: CPU (reference)" << std::endl;
    const auto startTime = std::chrono::high_resolution_clock::now();
    const std::vector output = ex::forwardBatch<Type, Type, Type, Type, Type>(uvData, mlpConfig.m_layers);
    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    std::cout << std::format("Reconstruction time: {:.3f} ms", elapsedMs) << std::endl;
    mapToLdr<Type>(output, texture);
  }
  else if (ex::isCppFallbackForced || options.m_useCppFallback) {
    std::cout << "Backend: C++ fallback" << std::endl;
    runCppFallbackInference<Type>(mlpConfig, uvData, texture);
  }
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  else {
    std::cout << "Backend: GPU" << std::endl;
    runGpuInference<Type>(mlpConfig, uvData, options, texture);
  }
#endif

  return texture;
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

  // Use half-precision (float16) for efficient GPU inference
  using DataT = half_float::half;

  // Step 1: Load the pre-trained MLP from a binary file
  MlpConfig<DataT> mlpConfig;
  {
    const std::filesystem::path mlpBinPathFs{options.m_mlpBinPath};
    std::ifstream mlpBin{mlpBinPathFs, std::ios::binary};
    if (!mlpBin.is_open()) {
      std::cerr << std::format("[Error] Failed to open binary file: {}", mlpBinPathFs.string()) << std::endl;
      std::abort();
    }
    mlpConfig = loadMlp<DataT>(mlpBin);
    mlpBin.close();
  }

  // Step 2: Run inference to reconstruct the texture
  const ex::PixmapU8 texture = reconstructTexture<DataT>(mlpConfig, options);

  // Step 3: Save the reconstructed texture as a PPM image
  {
    std::ofstream textureOut{options.m_outputPath, std::ios::binary};
    if (!textureOut.is_open()) {
      std::cerr << std::format("[Error] Failed to open output file: {}", options.m_outputPath) << std::endl;
      std::abort();
    }
    ex::writeAsPpm(texture, textureOut);
    std::cout << std::format("Output image saved to: {}", options.m_outputPath) << std::endl;
  }

  std::cout << std::flush;

  return 0;
}
