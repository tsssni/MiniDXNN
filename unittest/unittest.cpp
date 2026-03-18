/*!
  \file unittest.cpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

// Standard C++ library
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string_view>
#include <vector>
// GoogleTest
#include "gtest/gtest.h"
// Half
#include "common/activation.hpp"
#include "half.hpp"
// CLI
#include "CLI/CLI.hpp"
// Test
#include "hlsl_include_dirs.hpp"
#include "test.hpp"
#include "common/gfx_utility.hpp"
#include "common/matrix.hpp"
#include "common/mlp_layer.hpp"
#include "common/utility.hpp"
#include "common/xoshiro128plus.hpp"

static_assert(sizeof(half_float::half) == 2);
using test::CoopVecTest;

namespace {

test::TestParameters g_testParams;

template <ex::Arithmetic Type>
auto createRandomInputs(const size_t size, ex::Xoshiro128Plus& rng) -> std::vector<Type>
{
  std::normal_distribution<float> sampler{1.0f, 0.5f};
  std::vector<Type> result;
  result.resize(size);
  std::ranges::for_each(result, [&rng, &sampler](Type& v)
  {
    v = static_cast<Type>(sampler(rng));
    ex::validateValue(v);
  });
  return result;
}

template <ex::Arithmetic Type>
auto createRandomMlp(const size_t inputDim,
                     const size_t outputDim,
                     const size_t hiddenLayerDim,
                     const size_t numHiddenLayers,
                     const bool hasBias,
                     const ex::ActivationType activationHidden,
                     const ex::ActivationType activationLast,
                     ex::Xoshiro128Plus rng) -> std::vector<ex::MlpLayer<Type, Type>>
{
  // Create a random MLP
  std::vector<ex::LayerConfiguration> mlpConfiguration;
  if (numHiddenLayers == 0) {
    mlpConfiguration.emplace_back(inputDim, outputDim, activationLast);
  }
  else {
    mlpConfiguration.emplace_back(inputDim, hiddenLayerDim, activationHidden);
    for (size_t depth = 1; depth < numHiddenLayers; ++depth)
      mlpConfiguration.emplace_back(hiddenLayerDim, hiddenLayerDim, activationHidden);
    mlpConfiguration.emplace_back(hiddenLayerDim, outputDim, activationLast);
  }
  std::vector mlpData = ex::createMlp<Type, Type>(mlpConfiguration, hasBias, rng);
  return mlpData;
}

// Calculate the similarity of the given two values from 0 to 1
template <ex::Arithmetic Type>
auto calcSimilarity(const Type lhs, const Type rhs) noexcept -> double
{
  const auto l = static_cast<double>(lhs);
  const auto r = static_cast<double>(rhs);
  const double diff = std::abs(l - r);
  const double norm = std::max((std::abs(l) + std::abs(r)) / 2.0, std::numeric_limits<double>::epsilon());
  const double similarity = 1.0 - std::clamp(diff / norm, 0.0, 1.0);
  return similarity;
}

auto calcThreadGroupSize(const size_t numTasks, const size_t numThreads) noexcept -> size_t
{
  assert(std::has_single_bit(numThreads));
  const size_t n = (numTasks + (numThreads - 1)) / numThreads;
  return n;
}

template <ex::Arithmetic Type>
::testing::AssertionResult assertSimilarity(const char* expectedLabel,
                                            const char* valueLabel,
                                            const Type expected,
                                            const Type value,
                                            const double similarityThreshold)
{
  const auto e = static_cast<double>(expected);
  const auto v = static_cast<double>(value);
  const double similarity = calcSimilarity(e, v);
  if (similarity < similarityThreshold) {
    return ::testing::AssertionFailure()
        << "\n"
        << std::setprecision(8)
        << "expected: " << e << " (" << expectedLabel << ") and\n"
        << "value   : " << v << " (" << valueLabel << " )\n"
        << std::setprecision(4)
        << "  the similarity is less than the threshold. (similarity=" << similarity << " < " << similarityThreshold << ").";
  }
  return ::testing::AssertionSuccess();
}

template <ex::Arithmetic Type>
auto assertSimilarityBatch(const std::span<const Type> expected,
                           const std::span<const Type> values,
                           const double similarityThreshold,
                           const std::string_view testLabel,
                           const bool isDebugMode = false) -> ::testing::AssertionResult
{
  assert(expected.size() == values.size());
  const size_t numElements = expected.size();

  std::vector<double> similarities;
  similarities.reserve(numElements);

  for (size_t i = 0; i < numElements; ++i) {
    const double similarity = calcSimilarity(expected[i], values[i]);
    similarities.push_back(similarity);
  }

  const double averageSimilarity = std::accumulate(similarities.begin(), similarities.end(), 0.0) / numElements;
  const double maxSimilarity = *std::max_element(similarities.begin(), similarities.end());
  const double minSimilarity = *std::min_element(similarities.begin(), similarities.end());

  std::cout << std::format("{}: Similarity stats - avg: {:.6f}, max: {:.6f}, min: {:.6f}\n", 
                           testLabel, averageSimilarity, maxSimilarity, minSimilarity);

  if (isDebugMode and (minSimilarity < similarityThreshold)) {
    std::stringstream errorMessage;
    errorMessage << std::format("\n{}: [Warning] Minimum similarity ({:.6f}) is below threshold ({:.6f})\n", testLabel, minSimilarity, similarityThreshold);
    errorMessage << "Elements below threshold:\n";
    for (size_t i = 0; i < numElements; ++i) {
      if (similarities[i] < similarityThreshold) {
        errorMessage << std::format("  [{}]: expected={:.8f}, value={:.8f}, similarity={:.6f}\n", i, static_cast<double>(expected[i]), static_cast<double>(values[i]), similarities[i]);
      }
    }
    std::cerr << errorMessage.str() << std::endl;
  }

  if (averageSimilarity < similarityThreshold) {
    std::stringstream errorMessage;
    errorMessage << std::format("\n{}: [Error] Average similarity ({:.6f}) is below threshold ({:.6f})\n", testLabel, averageSimilarity, similarityThreshold);
    return ::testing::AssertionFailure() << errorMessage.str();
  }

  return ::testing::AssertionSuccess();
}

template <ex::Arithmetic Type>
auto testLinearAlgebraMulFloat(test::CoopVecTest& test,
                               const size_t rowSize, //! \note odd number with half may not work due to alignment issue
                               const size_t columnSize, //! \note odd number with half may not work due to alignment issue
                               const bool hasBias = false,
                               const bool useSoftwareLinAlgImpl = false,
                               const size_t numOfTests = 5) -> void
{
  const test::TestParameters& testParams = test.params();
  ex::Xoshiro128Plus rng{testParams.m_seed};

  for (size_t trial = 0; trial < numOfTests; ++trial) {
    // Create random weight matrix (rowSize x columnSize)
    std::vector<Type> matrixData = ::createRandomInputs<Type>(rowSize * columnSize, rng);
    const ex::MatrixRef<Type> matrix{rowSize, columnSize, std::span<Type>{matrixData}};
    // Create random bias (rowSize)
    std::vector<Type> bias;
    if (hasBias) {
      bias = ::createRandomInputs<Type>(rowSize, rng);
    }

    // Create random vector (columnSize)
    const std::vector<Type> vector = ::createRandomInputs<Type>(columnSize, rng);

    // Compute matrix-vector product
    const std::vector references = hasBias
        ? ex::mulAdd<Type, Type, Type, Type>(matrix, vector, bias)
        : ex::mul<Type, Type, Type>(matrix, vector);
    ASSERT_EQ(references.size(), rowSize);

    // Initialize GFX context
    const std::filesystem::path shaderDir = ex::getComputeShaderDir();
    const std::array includeDirList = ex::getHlslIncludeDirList();
    std::shared_ptr program = ex::createGfxProgram(test.context(), "linear_algebra_test", shaderDir, includeDirList);

    // Create buffers
    std::shared_ptr inputBuffer = ex::createGfxBuffer<Type>(test.context(), vector);
    const size_t outputSize = testParams.m_numThreadsX * rowSize;
    std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(test.context(), outputSize);
    std::shared_ptr weightBuffer = ex::convertToMatrixBuffer<Type>(test.context(), columnSize, rowSize, matrixData, testParams.m_weightMatrixLayout, ex::MATRIX_ALIGNMENT, ex::MATRIX_STRIDE_ALIGNMENT);
    std::shared_ptr biasBuffer = ex::convertToVectorBuffer<Type>(test.context(), bias, ex::VECTOR_ALIGNMENT);

    // Create and run the test kernel
    {
      const ex::OptionString kernelName = ex::createOptionString("testLinearAlgebraMulF{}Kernel", 8 * sizeof(Type));
      // Create the test kernel
      const std::array kernelDefinitions = std::to_array<ex::OptionString>({
          ex::createOptionString("MINIDXNN_HAS_BIAS={}", hasBias ? 1 : 0),
          ex::createOptionString("MINIDXNN_ROW_SIZE={}", rowSize),
          ex::createOptionString("MINIDXNN_COLUMN_SIZE={}", columnSize),
          ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", static_cast<int>(testParams.m_weightMatrixLayout)),
          ex::createOptionString("MINIDXNN_IS_WEIGHT_MATRIX_TRANSPOSED={}", 0),
          ex::createOptionString("MINIDXNN_MATRIX_STRIDE_ALIGNMENT={}", ex::MATRIX_STRIDE_ALIGNMENT),
          ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", testParams.m_numThreadsX),
          ex::createOptionString("MINIDXNN_NUM_TASKS={}", testParams.m_numThreadsX),
          ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", useSoftwareLinAlgImpl ? 1 : 0),
      });

      std::shared_ptr kernel = ex::createGfxComputeKernel(test.context(), *program, kernelName.data(), kernelDefinitions);
      constexpr size_t threadGroupSize = 1;
      ex::runKernel(test.context(), *program, *kernel, threadGroupSize,
          {
            ex::bind(*inputBuffer, "InputBuffer"),
            ex::bind(*outputBuffer, "OutputBuffer"),
            ex::bind(*weightBuffer, "WeightBuffer"),
            ex::bind(*biasBuffer, "BiasBuffer"),
          });
    }

    std::vector<Type> result;
    {
      std::shared_ptr staging = ex::createGfxBuffer<Type>(test.context(), outputSize, kGfxCpuAccess_Read);
      ex::copyBuffer(test.context(), *outputBuffer, *staging);
      const std::span output = ex::mapToCpu<Type>(test.context(), *staging);
      result.resize(outputSize);
      std::ranges::copy(output, result.begin());
    }

    // Manually verify the computation
    {
      const std::span<const Type> values{result.data(), rowSize};
      const std::string testLabel = std::format("Mul{}", trial + 1);
      ASSERT_TRUE(assertSimilarityBatch<Type>(references, values, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
    }
  }
}

template <ex::Arithmetic Type>
auto testSimpleMlpForwardFloat(test::CoopVecTest& test,
                               const size_t inputDim = 2,
                               const size_t outputDim = 2,
                               const size_t hiddenLayerDim = 8,
                               const size_t numHiddenLayers = 1,
                               const bool hasBias = true,
                               const bool useSoftwareLinAlgImpl = false,
                               const size_t numOfTests = 5) -> void
{
  const test::TestParameters& testParams = test.params();
  ex::Xoshiro128Plus rng{testParams.m_seed};

  for (size_t trial = 0; trial < numOfTests; ++trial) {
    // Create a random MLP
    constexpr ex::ActivationType activationHidden = ex::ActivationType::IDENTITY;
    constexpr ex::ActivationType activationLast = ex::ActivationType::IDENTITY;
    std::vector mlpData = ::createRandomMlp<Type>(inputDim, outputDim, hiddenLayerDim, numHiddenLayers, hasBias, activationHidden, activationLast, rng);

    // Load the test program
    const std::filesystem::path shaderDir = ex::getComputeShaderDir();
    const std::array includeDirList = ex::getHlslIncludeDirList();
    std::shared_ptr program = ex::createGfxProgram(test.context(), "simple_mlp_inference_test", shaderDir, includeDirList);

    const size_t numTasks = testParams.m_numTasks;
    // Create inputs and references
    const std::vector inputs = ::createRandomInputs<Type>(inputDim * numTasks, rng);
    const std::vector references = ex::forward<Type, Type, Type, Type>(mlpData, inputs);

    // Create buffers
    std::shared_ptr inputBuffer = ex::createGfxBuffer<Type>(test.context(), inputs);
    std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(test.context(), references.size());
    std::shared_ptr weightBuffer = ex::convertToMatrixBuffer<Type>(test.context(), mlpData, testParams.m_weightMatrixLayout, ex::MATRIX_ALIGNMENT, ex::MATRIX_STRIDE_ALIGNMENT);
    std::shared_ptr biasBuffer = ex::convertToVectorBuffer<Type>(test.context(), mlpData, ex::VECTOR_ALIGNMENT);

    // Create and run the test kernel
    {
      const ex::OptionString kernelName = ex::createOptionString("testMlpInferenceF{}Kernel", 8 * sizeof(Type));
      const std::array kernelDefinitions = std::to_array<ex::OptionString>({
          ex::createOptionString("MINIDXNN_HAS_BIAS={}", hasBias ? 1 : 0),
          ex::createOptionString("MINIDXNN_INPUT_DIMENSION={}", inputDim),
          ex::createOptionString("MINIDXNN_OUTPUT_DIMENSION={}", outputDim),
          ex::createOptionString("MINIDXNN_HIDDEN_LAYER_DIMENSIONS={}", hiddenLayerDim),
          ex::createOptionString("MINIDXNN_NUM_HIDDEN_LAYERS={}", numHiddenLayers),
          ex::createOptionString("MINIDXNN_ACTIVATION_HIDDEN_TYPE={}", ex::getActivationTypeString(activationHidden)),
          ex::createOptionString("MINIDXNN_ACTIVATION_LAST_TYPE={}", ex::getActivationTypeString(activationLast)),
          ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", static_cast<int>(testParams.m_weightMatrixLayout)),
          ex::createOptionString("MINIDXNN_IS_WEIGHT_MATRIX_TRANSPOSED={}", 0),
          ex::createOptionString("MINIDXNN_WEIGHT_ALIGNMENT={}", ex::MATRIX_ALIGNMENT),
          ex::createOptionString("MINIDXNN_WEIGHT_STRIDE_ALIGNMENT={}", ex::MATRIX_STRIDE_ALIGNMENT),
          ex::createOptionString("MINIDXNN_BIAS_ALIGNMENT={}", ex::VECTOR_ALIGNMENT),
          ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", testParams.m_numThreadsX),
          ex::createOptionString("MINIDXNN_NUM_TASKS={}", numTasks),
          ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", useSoftwareLinAlgImpl ? 1 : 0),
      });
      std::shared_ptr kernel = ex::createGfxComputeKernel(test.context(), *program, kernelName.data(), kernelDefinitions);
      const size_t threadGroupSize = calcThreadGroupSize(numTasks, testParams.m_numThreadsX);
      ex::runKernel(test.context(), *program, *kernel, threadGroupSize,
      {
        ex::bind(*inputBuffer, "InputBuffer"),
        ex::bind(*outputBuffer, "OutputBuffer"),
        ex::bind(*weightBuffer, "WeightBuffer"),
        ex::bind(*biasBuffer, "BiasBuffer"),
      });
    }

    {
      std::shared_ptr staging = ex::createGfxBuffer<Type>(test.context(), references.size(), kGfxCpuAccess_Read);
      ex::copyBuffer(test.context(), *outputBuffer, *staging);
      const std::span outputs = ex::mapToCpu<Type>(test.context(), *staging);

      const std::string testLabel = std::format("MLP{}", trial + 1);
      ASSERT_TRUE(assertSimilarityBatch<Type>(references, outputs, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
    }
  }
}

}

#define ADD_MATRIX_MUL_TEST(typeName, type, inputDim, outputDim) \
  TEST_P(CoopVecTest, MatrixMul_ ## typeName ## inputDim ## x ## outputDim ) \
  { \
    ::testLinearAlgebraMulFloat<type>(*this, inputDim, outputDim, false, false); \
  } \
  TEST_P(CoopVecTest, MatrixMulAdd_ ## typeName ## inputDim ## x ## outputDim ) \
  { \
    ::testLinearAlgebraMulFloat<type>(*this, inputDim, outputDim, true, false); \
  } \
  TEST_P(CoopVecTest, MatrixMul_ ## typeName ## inputDim ## x ## outputDim ## _Software) \
  { \
    ::testLinearAlgebraMulFloat<type>(*this, inputDim, outputDim, false, true); \
  } \
  TEST_P(CoopVecTest, MatrixMulAdd_ ## typeName ## inputDim ## x ## outputDim ## _Software) \
  { \
    ::testLinearAlgebraMulFloat<type>(*this, inputDim, outputDim, true, true); \
  }

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
ADD_MATRIX_MUL_TEST(F32, float, 2, 2);
ADD_MATRIX_MUL_TEST(F32, float, 16, 4);
ADD_MATRIX_MUL_TEST(F32, float, 64, 64);
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

ADD_MATRIX_MUL_TEST(F16, half_float::half, 2, 2);
ADD_MATRIX_MUL_TEST(F16, half_float::half, 16, 4);
ADD_MATRIX_MUL_TEST(F16, half_float::half, 64, 64);

#define ADD_SIMPL_MLP_TEST(typeName, type, layerLabel, inputDim, outputDim, hiddenDim, numHidden) \
  TEST_P(CoopVecTest, SimplMlpForward_ ## typeName ## _ ## layerLabel ##_NoBias ) \
  { \
    ::testSimpleMlpForwardFloat<type>(*this, inputDim, outputDim, hiddenDim, numHidden, false, false); \
  } \
  TEST_P(CoopVecTest, SimplMlpForward_ ## typeName ## _ ## layerLabel ) \
  { \
    ::testSimpleMlpForwardFloat<type>(*this, inputDim, outputDim, hiddenDim, numHidden, true, false); \
  } \
  TEST_P(CoopVecTest, SimplMlpForward_ ## typeName ## _ ## layerLabel ##_NoBias_Software ) \
  { \
    ::testSimpleMlpForwardFloat<type>(*this, inputDim, outputDim, hiddenDim, numHidden, false, true); \
  } \
  TEST_P(CoopVecTest, SimplMlpForward_ ## typeName ## _ ## layerLabel ##_Software ) \
  { \
    ::testSimpleMlpForwardFloat<type>(*this, inputDim, outputDim, hiddenDim, numHidden, true, true); \
  }

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
ADD_SIMPL_MLP_TEST(F32, float, 2x2, 2, 2, 1, 0);
ADD_SIMPL_MLP_TEST(F32, float, 16x4, 16, 4, 1, 0);
ADD_SIMPL_MLP_TEST(F32, float, 2x2x2, 2, 2, 2, 1);
ADD_SIMPL_MLP_TEST(F32, float, 8x16x16x4, 8, 4, 16, 2);
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

ADD_SIMPL_MLP_TEST(F16, half_float::half, 2x2, 2, 2, 1, 0);
ADD_SIMPL_MLP_TEST(F16, half_float::half, 16x4, 16, 4, 1, 0);
ADD_SIMPL_MLP_TEST(F16, half_float::half, 2x2x2, 2, 2, 2, 1);
ADD_SIMPL_MLP_TEST(F16, half_float::half, 8x16x16x4, 8, 4, 16, 2);

INSTANTIATE_TEST_SUITE_P(CoopVecTest, CoopVecTest, testing::Values(::g_testParams));

namespace {

auto createCommandLineParser(test::TestParameters& params) -> std::unique_ptr<CLI::App>
{
  // Setup CLI11
  const std::string appDesc = "MiniDxNn Unit Tests - Run compute shader tests with configurable parameters";
  std::unique_ptr parser = std::make_unique<CLI::App>(appDesc);

  // Allow extras for GoogleTest arguments
  parser->allow_extras();

  {
    const std::string desc = "Similarity threshold for test validation (0.0 to 1.0)";
    parser->add_option("--similarity-threshold", params.m_similarityThreshold, desc)
        ->default_val(params.m_similarityThreshold)
        ->check(CLI::Range(0.0, 1.0));
  }
  {
    const std::string desc = "Random seed for test data generation";
    parser->add_option("--seed", params.m_seed, desc)
        ->default_val(params.m_seed);
  }
  {
    const std::string desc = "Number of threads in X dimension (must be power of 2)";
    parser->add_option("--num-threads-x", params.m_numThreadsX, desc)
        ->default_val(params.m_numThreadsX)
        ->check(CLI::PositiveNumber);
  }
  {
    const std::string desc = "Number of tasks to execute";
    parser->add_option("--num-tasks", params.m_numTasks, desc)
        ->default_val(params.m_numTasks)
        ->check(CLI::PositiveNumber);
  }
  {
    const std::string desc = "Enable debug mode for detailed output";
    parser->add_flag("--debug", params.m_enableDebugMode, desc)
        ->default_val(params.m_enableDebugMode);
  }
  {
    const std::string desc = "Weight matrix layout: 0=ROW_MAJOR, 1=COLUMN_MAJOR";
    parser->add_option("--weight-matrix-layout", params.m_weightMatrixLayout, desc)
        ->default_val(ex::MatrixLayout::ROW_MAJOR)
        ->transform(CLI::CheckedTransformer(std::map<std::string, ex::MatrixLayout>{
            {"row-major", ex::MatrixLayout::ROW_MAJOR},
            {"column-major", ex::MatrixLayout::COLUMN_MAJOR},
            {"0", ex::MatrixLayout::ROW_MAJOR},
            {"1", ex::MatrixLayout::COLUMN_MAJOR}
        }, CLI::ignore_case));
  }

  return parser;
}

} // namespace

auto main(int argc, char** argv) -> int
{
  // Parse command line arguments with CLI11
  std::unique_ptr cliParser = ::createCommandLineParser(::g_testParams);
  try {
    cliParser->parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return cliParser->exit(e);
  }

  // Get remaining arguments for GoogleTest
  const std::vector remainingArgs = cliParser->remaining_for_passthrough();
  std::vector<const char*> gtestArgs;
  gtestArgs.resize(remainingArgs.size() + 1);
  gtestArgs.emplace_back(argv[0]);
  std::ranges::for_each(remainingArgs, [&gtestArgs](const std::string& arg)
  {
    gtestArgs.emplace_back(arg.c_str());
  });
  int gtestArgc = static_cast<int>(gtestArgs.size());

  ::testing::InitGoogleTest(&gtestArgc, const_cast<char**>(gtestArgs.data()));
  return RUN_ALL_TESTS();
}
