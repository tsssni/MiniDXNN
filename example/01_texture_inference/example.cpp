/*!
  \file example.cpp
  \author Sho Ikeda
  \brief Texture MLP demo - Load binary MLP data and generate 512x512 texture
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

// Standard C++ library
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <istream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
// GFX
#include "gfx.h"
// Half
#include "half.hpp"
// CLI
#include "CLI/CLI.hpp"
// Example
#include "hlsl_include_dirs.hpp"
#include "common/gfx_utility.hpp"
#include "common/image.hpp"
#include "common/mlp_layer.hpp"
#include "common/pixmap.hpp"
#include "common/utility.hpp"

namespace {

template <ex::Arithmetic Type>
auto loadMlp(std::istream& bin) -> std::vector<ex::MlpLayer<Type, Type>>
{
  // Read header
  std::uint32_t numHiddenLayers = 0;
  std::uint32_t hiddenLayerDim = 0;
  std::int32_t activationInt = 0;
  ex::read(&numHiddenLayers, bin);
  ex::read(&hiddenLayerDim, bin);
  ex::read(&activationInt, bin);

  if (!bin.good()) {
    std::cerr << "[Error] Failed to read header." << std::endl;
    std::abort();
  }

  std::cout << "MLP Configuration:\n";
  std::cout << "  numHiddenLayers: " << numHiddenLayers << "\n";
  std::cout << "  hiddenLayerDim: " << hiddenLayerDim << "\n";
  std::cout << "  activation: " << activationInt << "\n";

  using MlpLayerT = ex::MlpLayer<Type, Type>;
  std::vector<MlpLayerT> mlpLayers;
  mlpLayers.reserve(numHiddenLayers + 1);
  const auto addLayer = [&bin, &mlpLayers](const ex::LayerConfiguration& config)
  {
    MlpLayerT& layer = mlpLayers.emplace_back(config);
    std::vector<float> memory;
    {
      const size_t size = config.m_inputDim * config.m_outputDim;
      memory.resize(size);
      ex::read(memory.data(), bin, sizeof(float) * size); 
      std::ranges::transform(memory, layer.weightData().begin(), [](const float input) -> Type 
      {
        return static_cast<Type>(input);
      });
    }
    {
      const size_t size = config.m_outputDim;
      memory.resize(size);
      ex::read(memory.data(), bin, sizeof(float) * size); 
      std::ranges::transform(memory, layer.biasData().begin(), [](const float input) -> Type
      {
        return static_cast<Type>(input);
      });
    }
  };
  // Input layer
  const auto activation = static_cast<ex::ActivationType>(activationInt);
  addLayer(ex::LayerConfiguration{2, hiddenLayerDim, activation});
  for (size_t i = 0; i < numHiddenLayers - 1; ++i) {
    addLayer(ex::LayerConfiguration{hiddenLayerDim, hiddenLayerDim, activation});
  }
  // Output layer
  addLayer(ex::LayerConfiguration{hiddenLayerDim, 2, ex::ActivationType::SIGMOID});

  if (!bin.good()) {
    std::cerr << "[Error] Failed to read data" << std::endl;
    std::abort();
  }

  return mlpLayers;
}

template <ex::Arithmetic Type>
auto createUvData(const size_t width, const size_t height) -> std::vector<Type>
{
  const size_t numPixels = width * height;
  std::vector<Type> uvData;
  uvData.reserve(numPixels * 2);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      const float u = static_cast<float>(j) / (width - 1);
      const float v = static_cast<float>(i) / (height - 1);
      uvData.push_back(static_cast<Type>(u));
      uvData.push_back(static_cast<Type>(v));
    }
  }

  return uvData;
}

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
    out[i] = static_cast<std::uint8_t>(x);
  }
}

struct CliOptions
{
  std::string m_mlpBinPath = "texture-mlp-data.bin";
  std::string m_outputPath = "mlp-inference-output.ppm";
  size_t m_textureWidth = 4096;
  size_t m_textureHeight = 4096;
  bool m_useCpuMlpOperations = false;
  bool m_useSoftwareLinalg = false;
  bool m_enableDebugMode = false;
};

auto createCommandLineParser(CliOptions& options) -> std::unique_ptr<CLI::App>
{
  // Setup CLI11
  const std::string appDesc = "Texture MLP demo - Load binary MLP data and generate 512x512 texture";
  std::unique_ptr parser = std::make_unique<CLI::App>(appDesc);

  {
    const std::string desc = "Path to MLP binary file";
    parser->add_option("mlp-binary", options.m_mlpBinPath, desc)
        ->required()
        ->check(CLI::ExistingFile);
  }
  {
    const std::string desc = "Output image path (PPM format)";
    parser->add_option("output", options.m_outputPath, desc)
        ->default_val(options.m_outputPath);
  }
  {
    const std::string desc = "Texture width (in pixels)";
    parser->add_option("--texture-width", options.m_textureWidth, desc)
        ->default_val(options.m_textureWidth)
        ->check(CLI::PositiveNumber);
  }
  {
    const std::string desc = "Texture height (in pixels)";
    parser->add_option("--texture-height", options.m_textureHeight, desc)
        ->default_val(options.m_textureHeight)
        ->check(CLI::PositiveNumber);
  }
  {
    const std::string desc = "Use CPU ML operations intead of GPU";
    parser->add_flag("--cpu", options.m_useCpuMlpOperations, desc)
        ->default_val(options.m_useCpuMlpOperations);
  }
  {
    const std::string desc = "Use software-implementation linear algebra functions on HLSL";
    parser->add_flag("--software-linalg", options.m_useSoftwareLinalg, desc)
        ->default_val(options.m_useSoftwareLinalg);
  }
  {
    const std::string desc = "Enable debug mode for detailed output";
    parser->add_flag("--debug", options.m_enableDebugMode, desc)
        ->default_val(options.m_enableDebugMode);
  }

  return parser;
}

template <ex::Arithmetic Type>
auto reconstructTexture(const std::span<const ex::MlpLayer<Type, Type>> mlpData,
                        const CliOptions& options) -> ex::PixmapU8
{
  ex::PixmapU8 texture{options.m_textureWidth, options.m_textureHeight, 1};
  const std::vector uvData = createUvData<Type>(texture.width(), texture.height());
  if (options.m_useCpuMlpOperations) {
    const std::vector output = ex::forward<Type, Type, Type, Type>(mlpData, uvData);
    mapToLdr<Type>(output, texture);
  }
  else {
    std::shared_ptr context = ex::createGfxContext(options.m_enableDebugMode);
    {
      // Initialize GFX context
      const std::filesystem::path shaderDir = ex::getComputeShaderDir();
      const std::array includeDirList = ex::getHlslIncludeDirList();
      std::shared_ptr program = ex::createGfxProgram(*context, "01_texture_inference", shaderDir, includeDirList);

      const size_t inputDim = mlpData.front().inputDimension();
      const size_t outputDim = mlpData.back().outputDimension();
      const size_t numHiddenLayers = mlpData.size() - 1;
      const size_t hiddenLayerDim = mlpData.front().outputDimension();
      const ex::ActivationType activationHidden = mlpData.front().configuration().m_activation;
      const ex::ActivationType activationLast = mlpData.back().configuration().m_activation;
      const ex::MatrixLayout weightMatrixLayout = ex::MatrixLayout::ROW_MAJOR;
      constexpr size_t numThreadsX = 32;
      const size_t numTasks = texture.width() * texture.height();

      // Create buffers
      std::shared_ptr uvBuffer = ex::createGfxBuffer<Type>(*context, uvData);
      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(*context, 2 * numTasks);
      std::shared_ptr weightBuffer = ex::convertToMatrixBuffer<Type>(*context, mlpData, weightMatrixLayout, ex::MATRIX_ALIGNMENT, ex::MATRIX_STRIDE_ALIGNMENT);
      std::shared_ptr biasBuffer = ex::convertToVectorBuffer<Type>(*context, mlpData, ex::VECTOR_ALIGNMENT);

      // Create and run the example kernel
      {
        const ex::OptionString kernelName = ex::createOptionString("inferenceF{}Kernel", 8 * sizeof(Type));
        const std::array kernelDefinitions = std::to_array<ex::OptionString>({
            ex::createOptionString("MINIDXNN_INPUT_DIMENSION={}", inputDim),
            ex::createOptionString("MINIDXNN_OUTPUT_DIMENSION={}", outputDim),
            ex::createOptionString("MINIDXNN_NUM_HIDDEN_LAYERS={}", numHiddenLayers),
            ex::createOptionString("MINIDXNN_HIDDEN_LAYER_DIMENSIONS={}", hiddenLayerDim),
            ex::createOptionString("MINIDXNN_HAS_BIAS={}", 1),
            ex::createOptionString("MINIDXNN_ACTIVATION_HIDDEN_TYPE={}", ex::getActivationTypeString(activationHidden)),
            ex::createOptionString("MINIDXNN_ACTIVATION_LAST_TYPE={}", ex::getActivationTypeString(activationLast)),
            ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", static_cast<int>(weightMatrixLayout)),
            ex::createOptionString("MINIDXNN_IS_WEIGHT_MATRIX_TRANSPOSED={}", 0),
            ex::createOptionString("MINIDXNN_WEIGHT_ALIGNMENT={}", ex::MATRIX_ALIGNMENT),
            ex::createOptionString("MINIDXNN_WEIGHT_STRIDE_ALIGNMENT={}", ex::MATRIX_STRIDE_ALIGNMENT),
            ex::createOptionString("MINIDXNN_BIAS_ALIGNMENT={}", ex::VECTOR_ALIGNMENT),
            ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", numThreadsX),
            ex::createOptionString("MINIDXNN_NUM_TASKS={}", numTasks),
            ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", options.m_useSoftwareLinalg ? 1 : 0),
        });
        std::shared_ptr kernel = ex::createGfxComputeKernel(*context, *program, kernelName.data(), kernelDefinitions);
        const size_t threadGroupSize = numTasks / numThreadsX;
        gfxFinish(*context);
        float kernelTimeInMs = 0.0f;
        ex::runKernel(*context, *program, *kernel, threadGroupSize,
        {
          ex::bind(*uvBuffer, "UvBuffer"),
          ex::bind(*outputBuffer, "OutputBuffer"),
          ex::bind(*weightBuffer, "WeightBuffer"),
          ex::bind(*biasBuffer, "BiasBuffer"),
        }, {}, kernelTimeInMs);
        std::cout << std::format("inference (shader) time: {:.3f} ms", kernelTimeInMs) << std::endl;
      }
      {
        std::shared_ptr staging = ex::createGfxBuffer<Type>(*context, 2 * numTasks, kGfxCpuAccess_Read);
        ex::copyBuffer(*context, *outputBuffer, *staging);
        const std::span output = ex::mapToCpu<Type>(*context, *staging);
        mapToLdr<Type>(output, texture);
      }
    }
  }
  return texture;
}

} // namespace

auto main(const int argc, const char** argv) -> int
{
  ::CliOptions options{};
  std::unique_ptr cliParser = ::createCommandLineParser(options);
  CLI11_PARSE(*cliParser, argc, argv);

  using DataT = half_float::half;

  std::vector<ex::MlpLayer<DataT, DataT>> mlpData;
  // Load MLP binary
  {
    const std::filesystem::path mlpBinPathFs{options.m_mlpBinPath};
    std::ifstream mlpBin{mlpBinPathFs, std::ios::binary};
    if (!mlpBin.is_open()) {
      std::cerr << "[Error] Failed to open binary file: " << mlpBinPathFs << std::endl;
      std::abort();
    }
    mlpData = ::loadMlp<DataT>(mlpBin);
    mlpBin.close();
  }

  // Reconstruct a texture using the input MLP
  const ex::PixmapU8 texture = reconstructTexture<DataT>(mlpData, options);

  // Save the texture
  {
    std::ofstream textureOut{options.m_outputPath, std::ios::binary};
    if (!textureOut.is_open()) {
      std::cerr << "[Error] Failed to open output file: " << options.m_outputPath << std::endl;
      std::abort();
    }
    ex::writeAsPpm(texture, textureOut);
  }

  std::cout << std::flush;

  return 0;
}
