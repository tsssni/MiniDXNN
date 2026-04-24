/*!
  \file unittest.cpp
  \author Sho Ikeda
  \brief Unit test implementations for linear algebra, MLP inference/training, and atomic operations
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
#include <functional>
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
#include "half.hpp"
// Example
#include "common/activation.hpp"
// CLI
#include "CLI/CLI.hpp"
// Test
#include "test.hpp"
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
#include "hlsl_include_dirs.hpp"
#include "common/gfx_utility.hpp"
#endif // MINIDXNN_CPP_FALLBACK_ONLY
#include "common/loss.hpp"
#include "common/matrix.hpp"
#include "common/optimizer.hpp"
#include "common/xoshiro128plus.hpp"
// C++ fallback infrastructure (includes hlsl_compat.hpp, mlp.hlsl, utility, mlp_layer)
#include "common/cpp_fallback.hpp"
#include "kernel/mlp_test_common.hlsl"

static_assert(sizeof(half_float::half) == 2);

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
using test::CoopVecTest;
#endif
using test::CppFallbackTest;

namespace {

test::TestParameters g_testParams;

} // namespace

namespace {

template <ex::Arithmetic Type>
[[nodiscard]]
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
[[nodiscard]]
auto createRandomMlp(const size_t inputDim,
                     const size_t outputDim,
                     const size_t hiddenLayerDim,
                     const size_t numBackboneLayers,
                     const bool hasBias,
                     const ex::ActivationType activationHidden,
                     const ex::ActivationType activationLast,
                     ex::Xoshiro128Plus rng) -> std::vector<ex::MlpLayer<Type, Type>>
{
  std::vector<ex::LayerConfiguration> mlpConfiguration;
  if (numBackboneLayers == 0) {
    mlpConfiguration.emplace_back(inputDim, outputDim, activationLast);
  }
  else {
    mlpConfiguration.emplace_back(inputDim, hiddenLayerDim, activationHidden);
    for (size_t depth = 1; depth < numBackboneLayers; ++depth)
      mlpConfiguration.emplace_back(hiddenLayerDim, hiddenLayerDim, activationHidden);
    mlpConfiguration.emplace_back(hiddenLayerDim, outputDim, activationLast);
  }
  std::vector mlpData = ex::createMlp<Type, Type>(mlpConfiguration, hasBias, rng);
  return mlpData;
}

// Calculate the similarity of the given two values from 0 to 1
template <ex::Arithmetic Type>
[[nodiscard]]
auto calcSimilarity(const Type lhs, const Type rhs) noexcept -> double
{
  const auto l = static_cast<double>(lhs);
  const auto r = static_cast<double>(rhs);
  const double diff = std::abs(l - r);
  const double norm = std::max((std::abs(l) + std::abs(r)) / 2.0, std::numeric_limits<double>::epsilon());
  const double similarity = 1.0 - std::clamp(diff / norm, 0.0, 1.0);
  return similarity;
}

[[nodiscard, maybe_unused]]
auto calcThreadGroupSize(const size_t numTasks, const size_t numThreads) noexcept -> size_t
{
  assert(std::has_single_bit(numThreads));
  return (numTasks + (numThreads - 1)) / numThreads;
}

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
/*!
  \brief Build common MLP kernel definitions for test shaders.

  Creates the shared compile-time definitions array used by both inference and training
  test kernels. Additional test-specific definitions (e.g., MINIDXNN_NUM_TASKS,
  MINIDXNN_BATCH_SIZE) should be appended by the caller.
*/
[[nodiscard]]
auto buildMlpTestDefinitions(const test::TestParameters& testParams,
                             const size_t inputDim,
                             const size_t outputDim,
                             const size_t hiddenLayerDim,
                             const size_t numLayers,
                             const ex::ActivationType activationHidden,
                             const ex::ActivationType activationLast,
                             const bool hasBias,
                             const bool useSoftwareLinAlgImpl) -> std::vector<ex::OptionString>
{
  return {
      ex::createOptionString("MINIDXNN_HAS_BIAS={}", hasBias ? 1 : 0),
      ex::createOptionString("MINIDXNN_INPUT_DIMENSION={}", inputDim),
      ex::createOptionString("MINIDXNN_OUTPUT_DIMENSION={}", outputDim),
      ex::createOptionString("MINIDXNN_HIDDEN_LAYER_DIMENSIONS={}", hiddenLayerDim),
      ex::createOptionString("MINIDXNN_NUM_LAYERS={}", numLayers),
      ex::createOptionString("MINIDXNN_ACTIVATION_HIDDEN_TYPE={}", ex::getActivationTypeString(activationHidden)),
      ex::createOptionString("MINIDXNN_ACTIVATION_LAST_TYPE={}", ex::getActivationTypeString(activationLast)),
      ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", static_cast<int>(testParams.m_weightMatrixLayout)),
      ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_ALIGNMENT={}", ex::MATRIX_ALIGNMENT),
      ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", ex::MATRIX_VECTOR_STRIDE_ALIGNMENT),
      ex::createOptionString("MINIDXNN_BIAS_VECTOR_ALIGNMENT={}", ex::VECTOR_ALIGNMENT),
      ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", testParams.m_numThreadsX),
      ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", useSoftwareLinAlgImpl ? 1 : 0),
  };
}
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

template <ex::Arithmetic Type>
[[nodiscard]]
auto assertSimilarity(const char* expectedLabel,
                      const char* valueLabel,
                      const Type expected,
                      const Type value,
                      const double similarityThreshold) -> ::testing::AssertionResult
{
  const auto e = static_cast<double>(expected);
  const auto v = static_cast<double>(value);
  const double similarity = calcSimilarity(e, v);
  if (similarity < similarityThreshold) {
    return ::testing::AssertionFailure()
        << "\n"
        << std::format("expected: {:.8g} ('{}') and\n", e, expectedLabel)
        << std::format("value   : {:.8g} ('{}')\n", v, valueLabel)
        << "  the similarity is less than the threshold. "
        << std::format("(similarity={:.5g} < {:.5g}).", similarity, similarityThreshold);
  }
  return ::testing::AssertionSuccess();
}

template <ex::Arithmetic Type>
[[nodiscard]]
auto assertSimilarityBatch(const std::span<const Type> expected,
                           const std::span<const Type> values,
                           const double similarityThreshold,
                           const std::string_view testLabel,
                           const bool isDebugMode = false) -> ::testing::AssertionResult
{
  assert(expected.size() == values.size());
  const size_t numElements = expected.size();

  std::vector<double> similarities(numElements);
  std::ranges::transform(expected, values, similarities.begin(), [](const Type e, const Type v)
  {
    return calcSimilarity(e, v);
  });

  const auto [minIt, maxIt] = std::ranges::minmax_element(similarities);
  const double averageSimilarity = std::accumulate(similarities.begin(), similarities.end(), 0.0) / static_cast<double>(numElements);
  const double maxSimilarity = *maxIt;
  const double minSimilarity = *minIt;

  std::cout << std::format("{}: Similarity stats - avg: {:.6f}, max: {:.6f}, min: {:.6f}\n", testLabel, averageSimilarity, maxSimilarity, minSimilarity);

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
    return ::testing::AssertionFailure()
        << std::format("\n{}: [Error] Average similarity ({:.6f}) is below threshold ({:.6f})\n", testLabel, averageSimilarity, similarityThreshold);
  }

  return ::testing::AssertionSuccess();
}

} // namespace


// ============================================================================
// C++ fallback infrastructure (always available)
// Packs MLP layer data into flat byte buffers matching mlp.hlsl's layout,
// then dispatches forward/backward through compile-time template instantiations
// ============================================================================

namespace {

// ----------------------------------------------------------------------------
// C++ fallback helper functions
// These implement the kernel-equivalent operations using the mlp.hlsl C++ path.
// ----------------------------------------------------------------------------

template <ex::Arithmetic Type, int ROW_SIZE, int COLUMN_SIZE, bool IS_TRANSPOSED, bool HAS_BIAS>
void cppFallbackLinearAlgebraMul(const std::vector<std::uint8_t>& weightBuf,
                                 const size_t vectorStride,
                                 const std::vector<std::uint8_t>& biasBuf,
                                 const std::vector<Type>& inputVec,
                                 std::vector<Type>& outputVec)
{
  constexpr auto DT = ex::DxLinalgDataTypeOf<Type>::value;
  using Resolver = mininn::TransposeResolver<dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, IS_TRANSPOSED>;
  using MatrixRefT = dx::linalg::MatrixRefImpl<ByteAddressBuffer, DT, ROW_SIZE, COLUMN_SIZE, Resolver::EFFECTIVE_LAYOUT, Resolver::EFFECTIVE_TRANSPOSED>;

  ByteAddressBuffer wBuf{weightBuf};
  MatrixRefT matrix = {wBuf, 0, static_cast<uint>(vectorStride)};

  ::vector<Type, COLUMN_SIZE> input{};
  for (size_t d = 0; d < static_cast<size_t>(COLUMN_SIZE); ++d)
    input[d] = inputVec[d];

  auto interpreted = dx::linalg::MakeInterpretedVector<DT>(input);

  ::vector<Type, ROW_SIZE> result{};

  if constexpr (HAS_BIAS) {
    ByteAddressBuffer bBuf{biasBuf};
    dx::linalg::VectorRefImpl<ByteAddressBuffer, DT> biasRef = {bBuf, 0};
    result = mininn::impl::LinearAlgebra::mulAdd<Type>(matrix, interpreted, biasRef);
  } else {
    result = mininn::impl::LinearAlgebra::mul<Type>(matrix, interpreted);
  }

  outputVec.resize(ROW_SIZE);
  for (size_t d = 0; d < static_cast<size_t>(ROW_SIZE); ++d)
    outputVec[d] = result[d];
}

template <ex::Arithmetic Type, int SIZE>
void cppFallbackVectorAcc(RWByteAddressBuffer& outputBuf, const size_t numTasks)
{
  constexpr auto DT = mininn::impl::TypeTraits<Type>::COMPONENT_TYPE;

  for (size_t task = 0; task < numTasks; ++task) {
    ::vector<Type, SIZE> input{};
    for (size_t i = 0; i < static_cast<size_t>(SIZE); ++i)
      input[i] = static_cast<Type>(static_cast<float>(1u << static_cast<unsigned>(i)));

    dx::linalg::RWVectorRef<DT> output = {outputBuf, 0};
    mininn::impl::LinearAlgebra::vectorAcc(input, output);
  }
}

template <ex::Arithmetic Type>
void cppFallbackAtomicFetchAdd(RWByteAddressBuffer& outputBuf, const size_t numTasks)
{
  for (size_t task = 0; task < numTasks; ++task) {
    mininn::impl::atomicFetchAdd(outputBuf, 0 * sizeof(Type), static_cast<Type>(1));
    mininn::impl::atomicFetchAdd(outputBuf, 1 * sizeof(Type), static_cast<Type>(2));
    mininn::impl::atomicFetchAdd(outputBuf, 2 * sizeof(Type), static_cast<Type>(4));
    mininn::impl::atomicFetchAdd(outputBuf, 3 * sizeof(Type), static_cast<Type>(8));
  }
}

// ----------------------------------------------------------------------------
// Unified linear algebra, vector accumulation, and atomic test functions
// Shared between CoopVecTest (GPU) and CppFallbackTest (C++ fallback).
// GPU-specific code is guarded by #ifndef MINIDXNN_CPP_FALLBACK_ONLY.
// ----------------------------------------------------------------------------

template <ex::Arithmetic Type, int ROW_SIZE, int COLUMN_SIZE, bool IS_TRANSPOSED, bool HAS_BIAS>
auto testLinearAlgebraMul(const test::TestParameters& testParams,
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
                          GfxContext& gfxContext,
                          const bool useSoftwareLinAlgImpl,
                          const bool useCppFallback,
#endif
                          const size_t numOfTests = 5) -> void
{
  constexpr size_t rowSize = static_cast<size_t>(ROW_SIZE);
  constexpr size_t columnSize = static_cast<size_t>(COLUMN_SIZE);
  ex::Xoshiro128Plus rng{testParams.m_seed};

  for (size_t trial = 0; trial < numOfTests; ++trial) {
    std::vector<Type> matrixData = ::createRandomInputs<Type>(rowSize * columnSize, rng);
    const std::unique_ptr matrix = IS_TRANSPOSED
        ? ex::makeTransposedMatrix<Type>(columnSize, rowSize, std::span<Type>{matrixData})
        : ex::makeMatrix<Type>(rowSize, columnSize, std::span<Type>{matrixData});

    std::vector<Type> bias;
    if constexpr (HAS_BIAS)
      bias = ::createRandomInputs<Type>(rowSize, rng);

    const std::vector<Type> inputVec = ::createRandomInputs<Type>(columnSize, rng);

    const std::vector references = HAS_BIAS
        ? ex::mulAdd<Type, Type, Type, Type>(*matrix, inputVec, bias)
        : ex::mul<Type, Type, Type>(*matrix, inputVec);
    ASSERT_EQ(references.size(), rowSize);

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
    if (!useCppFallback) {
      const std::filesystem::path shaderDir = ex::getComputeShaderDir();
      const std::array includeDirList = ex::getHlslIncludeDirList();
      std::shared_ptr program = ex::createGfxProgram(gfxContext, "linear_algebra_test", shaderDir, includeDirList);

      std::array<size_t, 1> matrixSizeList;
      std::shared_ptr inputBuffer = ex::createGfxBuffer<Type>(gfxContext, inputVec);
      const size_t outputSize = testParams.m_numThreadsX * rowSize;
      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(gfxContext, outputSize);
      std::shared_ptr weightBuffer = IS_TRANSPOSED
          ? ex::convertToMatrixBuffer<Type>(gfxContext, rowSize, columnSize, matrixData, testParams.m_weightMatrixLayout, matrixSizeList, ex::MATRIX_ALIGNMENT, ex::MATRIX_VECTOR_STRIDE_ALIGNMENT)
          : ex::convertToMatrixBuffer<Type>(gfxContext, columnSize, rowSize, matrixData, testParams.m_weightMatrixLayout, matrixSizeList, ex::MATRIX_ALIGNMENT, ex::MATRIX_VECTOR_STRIDE_ALIGNMENT);
      std::shared_ptr biasBuffer = ex::convertToVectorBuffer<Type>(gfxContext, bias, ex::VECTOR_ALIGNMENT);

      {
        const ex::OptionString kernelName = ex::createOptionString("testLinearAlgebraMulF{}Kernel", 8 * sizeof(Type));
        const std::array kernelDefinitions = std::to_array<ex::OptionString>({
            ex::createOptionString("MINIDXNN_HAS_BIAS={}", HAS_BIAS ? 1 : 0),
            ex::createOptionString("MINIDXNN_ROW_SIZE={}", rowSize),
            ex::createOptionString("MINIDXNN_COLUMN_SIZE={}", columnSize),
            ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", static_cast<int>(testParams.m_weightMatrixLayout)),
            ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_IS_TRANSPOSED={}", IS_TRANSPOSED ? 1 : 0),
            ex::createOptionString("MINIDXNN_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", ex::MATRIX_VECTOR_STRIDE_ALIGNMENT),
            ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", testParams.m_numThreadsX),
            ex::createOptionString("MINIDXNN_NUM_TASKS={}", testParams.m_numThreadsX),
            ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", useSoftwareLinAlgImpl ? 1 : 0),
        });
        std::shared_ptr kernel = ex::createGfxComputeKernel(gfxContext, *program, kernelName.data(), kernelDefinitions);
        constexpr size_t threadGroupSize = 1;
        ex::runKernel(gfxContext, *program, *kernel, threadGroupSize,
            {
              ex::bind(*inputBuffer, "InputBuffer"),
              ex::bind(*outputBuffer, "OutputBuffer"),
              ex::bind(*weightBuffer, "WeightBuffer"),
              ex::bind(*biasBuffer, "BiasBuffer"),
            });
      }

      std::vector<Type> result;
      {
        std::shared_ptr staging = ex::createGfxBuffer<Type>(gfxContext, outputSize, kGfxCpuAccess_Read);
        ex::copyBuffer(gfxContext, *outputBuffer, *staging);
        const std::span output = ex::mapToCpu<Type>(gfxContext, *staging);
        result.resize(outputSize);
        std::ranges::copy(output, result.begin());
      }

      const std::span<const Type> values{result.data(), rowSize};
      const std::string testLabel = std::format("Mul{}", trial + 1);
      ASSERT_TRUE(assertSimilarityBatch<Type>(references, values, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
    } else
#endif // !MINIDXNN_CPP_FALLBACK_ONLY
    {
      auto [weightBuf, vectorStride] = ex::packSingleMatrix<Type>(matrixData, rowSize, columnSize, IS_TRANSPOSED);
      std::vector<std::uint8_t> biasBuf = HAS_BIAS ? ex::packSingleBias<Type>(bias, rowSize)
                                                   : std::vector<std::uint8_t>{};

      std::vector<Type> result;
      cppFallbackLinearAlgebraMul<Type, ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED, HAS_BIAS>(weightBuf, vectorStride, biasBuf, inputVec, result);

      const std::string testLabel = std::format("Mul{}", trial + 1);
      ASSERT_TRUE(assertSimilarityBatch<Type>(references, result, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
    }
  }
}

template <ex::Arithmetic Type>
auto testVectorAcc(const test::TestParameters& testParams,
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
                   GfxContext& gfxContext,
                   const bool useSoftwareLinAlgImpl,
                   const bool useCppFallback,
#endif
                   const size_t numOfTests = 5) -> void
{
  constexpr size_t size = 4;
  constexpr size_t numTasks = 1ull << (std::min)(std::numeric_limits<Type>::digits, 18);

  for (size_t trial = 0; trial < numOfTests; ++trial) {
    std::array<Type, size> result{};
    result.fill(static_cast<Type>(0));

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
    if (!useCppFallback) {
      const std::filesystem::path shaderDir = ex::getComputeShaderDir();
      const std::array includeDirList = ex::getHlslIncludeDirList();
      std::shared_ptr program = ex::createGfxProgram(gfxContext, "linear_algebra_test", shaderDir, includeDirList);

      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(gfxContext, result);

      {
        const ex::OptionString kernelName = ex::createOptionString("testLinearAlgebraVectorAccF{}Kernel", 8 * sizeof(Type));
        const std::array kernelDefinitions = std::to_array<ex::OptionString>({
            ex::createOptionString("MINIDXNN_HAS_BIAS={}", 0),
            ex::createOptionString("MINIDXNN_ROW_SIZE={}", 1),
            ex::createOptionString("MINIDXNN_COLUMN_SIZE={}", size),
            ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", 0),
            ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_IS_TRANSPOSED={}", 0),
            ex::createOptionString("MINIDXNN_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", ex::MATRIX_VECTOR_STRIDE_ALIGNMENT),
            ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", testParams.m_numThreadsX),
            ex::createOptionString("MINIDXNN_NUM_TASKS={}", numTasks),
            ex::createOptionString("MINIDXNN_USE_SOFTWARE_LINALG_IMPL={}", useSoftwareLinAlgImpl ? 1 : 0),
        });

        std::shared_ptr kernel = ex::createGfxComputeKernel(gfxContext, *program, kernelName.data(), kernelDefinitions);
        const size_t threadGroupSize = calcThreadGroupSize(numTasks, testParams.m_numThreadsX);
        ex::runKernel(gfxContext, *program, *kernel, threadGroupSize,
            {
              ex::bind(*outputBuffer, "OutputBuffer"),
            });
      }

      {
        std::shared_ptr staging = ex::createGfxBuffer<Type>(gfxContext, size, kGfxCpuAccess_Read);
        ex::copyBuffer(gfxContext, *outputBuffer, *staging);
        const std::span output = ex::mapToCpu<Type>(gfxContext, *staging);
        for (size_t i = 0; i < size; ++i)
          result[i] = output[i];
      }
    } else
#endif // !MINIDXNN_CPP_FALLBACK_ONLY
    {
      std::vector<std::uint8_t> outputData(size * sizeof(Type), 0);
      RWByteAddressBuffer outputBuf{outputData};

      cppFallbackVectorAcc<Type, static_cast<int>(size)>(outputBuf, numTasks);

      for (size_t i = 0; i < size; ++i)
        std::memcpy(&result[i], outputData.data() + i * sizeof(Type), sizeof(Type));
    }

    const auto assertTest = [&testParams](const char* expectedLabel, const char* valueLabel, const Type expected, const Type value)
    {
      return assertSimilarity(expectedLabel, valueLabel, expected, value, testParams.m_similarityThreshold);
    };

    for (size_t i = 0; i < size; ++i) {
      const auto expected = static_cast<Type>(static_cast<float>(numTasks << i));
      EXPECT_PRED_FORMAT2(assertTest, expected, result[i]);
    }
  }
}

template <ex::Arithmetic Type>
auto testAtomicFetchAdd(const test::TestParameters& testParams,
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
                        GfxContext& gfxContext,
                        const bool useCppFallback,
#endif
                        const size_t numOfTests = 5) -> void
{
  constexpr size_t size = 4;
  constexpr size_t numTasks = 1ull << (std::min)(std::numeric_limits<Type>::digits, 17);

  for (size_t trial = 0; trial < numOfTests; ++trial) {
    std::array<Type, size> result{};
    result.fill(static_cast<Type>(0));

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
    if (!useCppFallback) {
      const std::filesystem::path shaderDir = ex::getComputeShaderDir();
      const std::array includeDirList = ex::getHlslIncludeDirList();
      std::shared_ptr program = ex::createGfxProgram(gfxContext, "atomic_test", shaderDir, includeDirList);

      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(gfxContext, result);

      {
        const ex::OptionString kernelName = ex::createOptionString("testAtomicFetchAddF{}Kernel", 8 * sizeof(Type));
        const std::array kernelDefinitions = std::to_array<ex::OptionString>({
            ex::createOptionString("MINIDXNN_NUM_THREADS_X={}", testParams.m_numThreadsX),
            ex::createOptionString("MINIDXNN_NUM_TASKS={}", numTasks),
        });

        std::shared_ptr kernel = ex::createGfxComputeKernel(gfxContext, *program, kernelName.data(), kernelDefinitions);
        const size_t threadGroupSize = calcThreadGroupSize(numTasks, testParams.m_numThreadsX);
        ex::runKernel(gfxContext, *program, *kernel, threadGroupSize,
            {
              ex::bind(*outputBuffer, "OutputBuffer"),
            });
      }

      {
        std::shared_ptr staging = ex::createGfxBuffer<Type>(gfxContext, size, kGfxCpuAccess_Read);
        ex::copyBuffer(gfxContext, *outputBuffer, *staging);
        const std::span output = ex::mapToCpu<Type>(gfxContext, *staging);
        for (size_t i = 0; i < size; ++i)
          result[i] = output[i];
      }
    } else
#endif // !MINIDXNN_CPP_FALLBACK_ONLY
    {
      std::vector<std::uint8_t> outputData(size * sizeof(Type), 0);
      RWByteAddressBuffer outputBuf{outputData};

      cppFallbackAtomicFetchAdd<Type>(outputBuf, numTasks);

      for (size_t i = 0; i < size; ++i)
        std::memcpy(&result[i], outputData.data() + i * sizeof(Type), sizeof(Type));
    }

    const auto assertTest = [&testParams](const char* expectedLabel, const char* valueLabel, const Type expected, const Type value)
    {
      return assertSimilarity(expectedLabel, valueLabel, expected, value, testParams.m_similarityThreshold);
    };

    for (size_t i = 0; i < size; ++i) {
      const auto expected = static_cast<Type>(static_cast<float>(numTasks << i));
      EXPECT_PRED_FORMAT2(assertTest, expected, result[i]);
    }
  }
}

// ----------------------------------------------------------------------------
// C++ fallback kernel dispatch (MLP forward/backward)
// ----------------------------------------------------------------------------

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM,
          typename ActivationHiddenT, typename ActivationLastT, bool HAS_BIAS>
void cppFallbackForwardKernel(ex::PackedMlpBuffers<Type>& packed,
                              const std::vector<Type>& inputs,
                              std::vector<Type>& outputs,
                              size_t numTasks)
{
  constexpr auto DT = ex::DxLinalgDataTypeOf<Type>::value;

  ByteAddressBuffer inputBuf{inputs};
  RWByteAddressBuffer outputBuf{outputs};

  for (uint task = 0; task < static_cast<uint>(numTasks); ++task) {
    testkernel::inferenceStep<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM,
        DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
        128, 16, 64, HAS_BIAS>(
        task, inputBuf, outputBuf, packed.weightBAB(), packed.biasBAB(),
        packed.matrixSizes, static_cast<uint>(numTasks));
  }
}

// Dispatch activation types at runtime
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM, bool HAS_BIAS>
bool dispatchForwardActivation(ex::ActivationType hiddenAct,
                               ex::ActivationType lastAct,
                               ex::PackedMlpBuffers<Type>& packed,
                               const std::vector<Type>& inputs,
                               std::vector<Type>& outputs,
                               size_t numTasks)
{
  #define DISPATCH_ACT(HiddenT, LastT, hiddenE, lastE) \
    if (hiddenAct == (hiddenE) && lastAct == (lastE)) { \
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM, HiddenT, LastT, HAS_BIAS>(packed, inputs, outputs, numTasks); \
      return true; \
    }

  using namespace mininn;
  DISPATCH_ACT(IdentityActivation,  IdentityActivation,  ex::ActivationType::IDENTITY, ex::ActivationType::IDENTITY)
  DISPATCH_ACT(IdentityActivation,  SigmoidActivation,   ex::ActivationType::IDENTITY, ex::ActivationType::SIGMOID)
  DISPATCH_ACT(ReluActivation,      IdentityActivation,  ex::ActivationType::RELU,     ex::ActivationType::IDENTITY)
  DISPATCH_ACT(ReluActivation,      SigmoidActivation,   ex::ActivationType::RELU,     ex::ActivationType::SIGMOID)
  DISPATCH_ACT(SigmoidActivation,   SigmoidActivation,   ex::ActivationType::SIGMOID,  ex::ActivationType::SIGMOID)
  DISPATCH_ACT(LeakyReluActivation, SigmoidActivation,   ex::ActivationType::LEAKY_RELU, ex::ActivationType::SIGMOID)
  DISPATCH_ACT(LeakyReluActivation, IdentityActivation,  ex::ActivationType::LEAKY_RELU, ex::ActivationType::IDENTITY)

  #undef DISPATCH_ACT
  return false;
}

// Dispatch NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM at runtime
template <ex::Arithmetic Type>
bool dispatchForward(size_t numLayers, size_t hiddenDim, size_t inputDim, size_t outputDim,
                     ex::ActivationType hiddenAct, ex::ActivationType lastAct,
                     ex::PackedMlpBuffers<Type>& packed,
                     const std::vector<Type>& inputs,
                     std::vector<Type>& outputs,
                     size_t numTasks,
                     bool hasBias = true)
{
  #define DISPATCH_FWD(NL, HD, ID, OD) \
    if (numLayers == (NL) && hiddenDim == (HD) && inputDim == (ID) && outputDim == (OD)) { \
      if (hasBias) \
        return dispatchForwardActivation<Type, (NL), (HD), (ID), (OD), true>(hiddenAct, lastAct, packed, inputs, outputs, numTasks); \
      else \
        return dispatchForwardActivation<Type, (NL), (HD), (ID), (OD), false>(hiddenAct, lastAct, packed, inputs, outputs, numTasks); \
    }

  // Single layer (numBackboneLayers=0)
  DISPATCH_FWD(1, 2, 2, 2)
  DISPATCH_FWD(1, 4, 2, 4)
  DISPATCH_FWD(1, 16, 16, 4)
  DISPATCH_FWD(1, 16, 16, 16)
  // 2 layers (1 backbone)
  DISPATCH_FWD(2, 2, 2, 2)
  DISPATCH_FWD(2, 8, 2, 4)
  // 3 layers (2 backbone)
  DISPATCH_FWD(3, 8, 2, 4)
  DISPATCH_FWD(3, 16, 8, 4)
  // 4 layers (3 backbone)
  DISPATCH_FWD(4, 6, 2, 4)
  DISPATCH_FWD(4, 8, 2, 4)

  #undef DISPATCH_FWD
  return false;
}

// ----------------------------------------------------------------------------
// Backward test — dispatched on NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM, activations
// ----------------------------------------------------------------------------

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM,
          typename ActivationHiddenT, typename ActivationLastT, bool HAS_BIAS>
void cppFallbackBackwardKernel(ex::PackedMlpBuffers<Type>& packed,
                               const std::vector<Type>& inputs,
                               const std::vector<Type>& targets,
                               std::vector<Type>& outputs,
                               std::vector<std::uint8_t>& weightGradBuf,
                               std::vector<std::uint8_t>& biasGradBuf,
                               size_t batchSize)
{
  constexpr auto DT = ex::DxLinalgDataTypeOf<Type>::value;

  const size_t logitsStride = ex::alignBytes(static_cast<size_t>(HIDDEN_DIM) * sizeof(Type), ex::VECTOR_ALIGNMENT);
  const size_t logitsPerSample = logitsStride * NUM_LAYERS;
  const size_t logitsBufSize = logitsPerSample * batchSize;

  weightGradBuf.assign(packed.weightBuf.size(), 0);
  biasGradBuf.assign(packed.biasBuf.size(), 0);
  std::vector<std::uint8_t> logitsBuf(logitsBufSize, 0);

  ByteAddressBuffer inputBuf{inputs};
  ByteAddressBuffer targetBuf{targets};
  RWByteAddressBuffer outputBuf{outputs};

  RWByteAddressBuffer wGradBAB{weightGradBuf};
  RWByteAddressBuffer bGradBAB{biasGradBuf};
  RWByteAddressBuffer logitsBAB{logitsBuf};

  // Forward pass for all samples
  for (uint s = 0; s < static_cast<uint>(batchSize); ++s) {
    testkernel::trainingForwardStep<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM,
        DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
        128, 16, 64, HAS_BIAS>(
        s, inputBuf, outputBuf, packed.weightBAB(), packed.biasBAB(),
        logitsBAB, packed.matrixSizes,
        static_cast<uint>(batchSize), static_cast<uint>(logitsPerSample));
  }

  // Backward pass for all samples
  RWByteAddressBuffer outputReadBuf{outputs};

  for (uint s = 0; s < static_cast<uint>(batchSize); ++s) {
    testkernel::trainingBackwardStep<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM,
        DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
        128, 16, 64, HAS_BIAS>(
        s, inputBuf, targetBuf, outputReadBuf, packed.weightBAB(), packed.biasBAB(),
        wGradBAB, bGradBAB, logitsBAB, packed.matrixSizes,
        static_cast<uint>(batchSize), static_cast<uint>(logitsPerSample));
  }
}

// Dispatch activation types for backward
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM, bool HAS_BIAS>
bool dispatchBackwardActivation(ex::ActivationType hiddenAct,
                                ex::ActivationType lastAct,
                                ex::PackedMlpBuffers<Type>& packed,
                                const std::vector<Type>& inputs,
                                const std::vector<Type>& targets,
                                std::vector<Type>& outputs,
                                std::vector<std::uint8_t>& weightGradBuf,
                                std::vector<std::uint8_t>& biasGradBuf,
                                size_t batchSize)
{
  #define DISPATCH_BACK_ACT(HiddenT, LastT, hiddenE, lastE) \
    if (hiddenAct == (hiddenE) && lastAct == (lastE)) { \
      cppFallbackBackwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM, HiddenT, LastT, HAS_BIAS>( \
          packed, inputs, targets, outputs, weightGradBuf, biasGradBuf, batchSize); \
      return true; \
    }

  using namespace mininn;
  DISPATCH_BACK_ACT(LeakyReluActivation, SigmoidActivation, ex::ActivationType::LEAKY_RELU, ex::ActivationType::SIGMOID)
  DISPATCH_BACK_ACT(IdentityActivation,  IdentityActivation, ex::ActivationType::IDENTITY,  ex::ActivationType::IDENTITY)
  DISPATCH_BACK_ACT(ReluActivation,      SigmoidActivation,  ex::ActivationType::RELU,      ex::ActivationType::SIGMOID)

  #undef DISPATCH_BACK_ACT
  return false;
}

// Dispatch NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM for backward
template <ex::Arithmetic Type>
bool dispatchBackward(size_t numLayers, size_t hiddenDim, size_t inputDim, size_t outputDim,
                      ex::ActivationType hiddenAct, ex::ActivationType lastAct,
                      ex::PackedMlpBuffers<Type>& packed,
                      const std::vector<Type>& inputs,
                      const std::vector<Type>& targets,
                      std::vector<Type>& outputs,
                      std::vector<std::uint8_t>& weightGradBuf,
                      std::vector<std::uint8_t>& biasGradBuf,
                      size_t batchSize,
                      bool hasBias = true)
{
  #define DISPATCH_BACK(NL, HD, ID, OD) \
    if (numLayers == (NL) && hiddenDim == (HD) && inputDim == (ID) && outputDim == (OD)) { \
      if (hasBias) \
        return dispatchBackwardActivation<Type, (NL), (HD), (ID), (OD), true>(hiddenAct, lastAct, packed, inputs, targets, outputs, weightGradBuf, biasGradBuf, batchSize); \
      else \
        return dispatchBackwardActivation<Type, (NL), (HD), (ID), (OD), false>(hiddenAct, lastAct, packed, inputs, targets, outputs, weightGradBuf, biasGradBuf, batchSize); \
    }

  // Single layer (numBackboneLayers=0): inputDim=2, outputDim=4
  DISPATCH_BACK(1, 4, 2, 4)
  // 2 layers (1 backbone)
  DISPATCH_BACK(2, 8, 2, 4)
  // 3 layers (2 backbone)
  DISPATCH_BACK(3, 8, 2, 4)
  // 4 layers (3 backbone)
  DISPATCH_BACK(4, 6, 2, 4)
  DISPATCH_BACK(4, 8, 2, 4)

  #undef DISPATCH_BACK
  return false;
}

// Smoke test: identity weights with sigmoid activation (verifies basic mlp.hlsl C++ path)
template <typename Type>
void testCppFallbackForwardSmoke(const CppFallbackTest& /* test */)
{
  constexpr unsigned int NUM_LAYERS = 1;
  constexpr int DIM = 2;

  std::vector<std::uint8_t> weightBuf(256, 0);
  std::vector<std::uint8_t> biasBuf(128, 0);

  {
    Type* w0 = reinterpret_cast<Type*>(weightBuf.data());
    w0[0] = Type(1.0f);
    w0[1] = Type(0.0f);
    Type* w1 = reinterpret_cast<Type*>(weightBuf.data() + 16);
    w1[0] = Type(0.0f);
    w1[1] = Type(1.0f);
  }
  {
    Type* b = reinterpret_cast<Type*>(biasBuf.data());
    b[0] = Type(0.1f);
    b[1] = Type(0.2f);
  }

  ByteAddressBuffer wBuf{weightBuf};
  ByteAddressBuffer bBuf{biasBuf};

  constexpr auto DT = ex::DxLinalgDataTypeOf<Type>::value;
  using LayerDataRefT = mininn::InferenceLayerDataRef<
    NUM_LAYERS, DIM,
    DT,
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    DT,
    DT,
    mininn::IdentityActivation,
    mininn::SigmoidActivation,
    DT,
    128, 16, 64>;

  LayerDataRefT layerData;
  layerData.setWeightData(wBuf, uint2{128u, 128u}, 0);
  layerData.setBiasData(bBuf, 0);

  vector<Type, DIM> input;
  input[0] = Type(0.5f);
  input[1] = Type(0.3f);
  vector<Type, DIM> output;
  mininn::forward(output, input, layerData);

  const float expected0 = 1.0f / (1.0f + std::exp(-0.6f));
  const float expected1 = 1.0f / (1.0f + std::exp(-0.5f));
  EXPECT_NEAR(static_cast<float>(output[0]), expected0, 0.01f);
  EXPECT_NEAR(static_cast<float>(output[1]), expected1, 0.01f);
}

} // namespace

// ============================================================================
// Unified MLP test functions — shared between CoopVecTest and CppFallbackTest
// ============================================================================

namespace {

template <ex::Arithmetic Type>
auto testSimpleMlpForwardFloat(const test::TestParameters& testParams,
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
                               GfxContext& gfxContext,
                               const bool useSoftwareLinAlgImpl,
                               const bool useCppFallback,
#endif
                               const size_t inputDim,
                               const size_t outputDim,
                               const size_t hiddenLayerDim,
                               const size_t numBackboneLayers,
                               const bool hasBias,
                               const size_t numOfTests = 5) -> void
{
  ex::Xoshiro128Plus rng{testParams.m_seed};

  for (size_t trial = 0; trial < numOfTests; ++trial) {
    // Create a random MLP
    constexpr ex::ActivationType activationHidden = ex::ActivationType::IDENTITY;
    constexpr ex::ActivationType activationLast = ex::ActivationType::IDENTITY;
    std::vector mlpData = ::createRandomMlp<Type>(inputDim, outputDim, hiddenLayerDim, numBackboneLayers, hasBias, activationHidden, activationLast, rng);

    const size_t numTasks = testParams.m_numTasks;
    // Create inputs and references
    const std::vector inputs = ::createRandomInputs<Type>(inputDim * numTasks, rng);
    const std::vector references = ex::forwardBatch<Type, Type, Type, Type, Type>(inputs, mlpData);

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
    if (!useCppFallback) {
      // GPU path
      // Load the test program
      const std::filesystem::path shaderDir = ex::getComputeShaderDir();
      const std::array includeDirList = ex::getHlslIncludeDirList();
      std::shared_ptr program = ex::createGfxProgram(gfxContext, "simple_mlp_inference_test", shaderDir, includeDirList);

      std::vector<size_t> matrixSizeList;
      matrixSizeList.resize(numBackboneLayers + 1);

      // Create buffers
      std::shared_ptr inputBuffer = ex::createGfxBuffer<Type>(gfxContext, inputs);
      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(gfxContext, references.size());
      std::shared_ptr weightBuffer = ex::convertToMatrixBuffer<Type>(gfxContext, mlpData, testParams.m_weightMatrixLayout, matrixSizeList, ex::MATRIX_ALIGNMENT, ex::MATRIX_VECTOR_STRIDE_ALIGNMENT);
      std::shared_ptr biasBuffer = ex::convertToVectorBuffer<Type>(gfxContext, mlpData, ex::VECTOR_ALIGNMENT);

      // Create and run the test kernel
      {
        const ex::OptionString kernelName = ex::createOptionString("testMlpInferenceF{}Kernel", 8 * sizeof(Type));
        std::vector kernelDefinitions = buildMlpTestDefinitions(testParams, inputDim, outputDim, hiddenLayerDim, numBackboneLayers + 1, activationHidden, activationLast, hasBias, useSoftwareLinAlgImpl);
        kernelDefinitions.push_back(ex::createOptionString("MINIDXNN_NUM_TASKS={}", numTasks));
        std::shared_ptr kernel = ex::createGfxComputeKernel(gfxContext, *program, kernelName.data(), kernelDefinitions);
        const size_t threadGroupSize = calcThreadGroupSize(numTasks, testParams.m_numThreadsX);
        ex::runKernel(gfxContext, *program, *kernel, threadGroupSize,
            {
              ex::bind(*inputBuffer, "InputBuffer"),
              ex::bind(*outputBuffer, "OutputBuffer"),
              ex::bind(*weightBuffer, "WeightBuffer"),
              ex::bind(*biasBuffer, "BiasBuffer"),
            },
            {
              ex::bind(static_cast<std::int32_t>(matrixSizeList.front()), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
              ex::bind(static_cast<std::int32_t>((matrixSizeList.size() > 1) ? matrixSizeList.at(1) : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
            });
      }

      {
        std::shared_ptr staging = ex::createGfxBuffer<Type>(gfxContext, references.size(), kGfxCpuAccess_Read);
        ex::copyBuffer(gfxContext, *outputBuffer, *staging);
        const std::span outputs = ex::mapToCpu<Type>(gfxContext, *staging);

        const std::string testLabel = std::format("MLP{}", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(references, outputs, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }
    } else
#endif // !MINIDXNN_CPP_FALLBACK_ONLY
    {
      // C++ fallback path
      const size_t effectiveHiddenDim = (numBackboneLayers == 0)
          ? std::max(inputDim, outputDim)
          : hiddenLayerDim;
      ex::PackedMlpBuffers<Type> packed;
      packed.pack(mlpData, hasBias);
      std::vector<Type> outputs(numTasks * outputDim);
      const bool dispatched = dispatchForward<Type>(
          mlpData.size(), effectiveHiddenDim, inputDim, outputDim,
          activationHidden, activationLast,
          packed, inputs, outputs, numTasks, hasBias);
      ASSERT_TRUE(dispatched)
          << "Unsupported MLP config: layers=" << mlpData.size()
          << " hiddenDim=" << effectiveHiddenDim
          << " inputDim=" << inputDim
          << " outputDim=" << outputDim;

      const std::string testLabel = std::format("CppFallback_Forward{}", trial + 1);
      ASSERT_TRUE(assertSimilarityBatch<Type>(references, outputs, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
    }
  }
}

template <ex::Arithmetic Type>
auto testSimpleMlpBackwardFloat(const test::TestParameters& testParams,
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
                                GfxContext& gfxContext,
                                const bool useSoftwareLinAlgImpl,
                                const bool useCppFallback,
#endif
                                const size_t hiddenLayerDim,
                                const size_t numBackboneLayers,
                                const size_t batchSize,
                                const bool hasBias,
                                const size_t numOfTests = 5) -> void
{
  constexpr size_t inputDim = 2;
  constexpr size_t outputDim = 4;
  constexpr ex::ActivationType activationHidden = ex::ActivationType::LEAKY_RELU;
  constexpr ex::ActivationType activationLast = ex::ActivationType::SIGMOID;

  ex::Xoshiro128Plus rng{testParams.m_seed};

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  // GPU-only: load test program and create kernels outside the trial loop
  std::shared_ptr<GfxProgram> gfxProgram;
  std::shared_ptr<GfxKernel> gfxFwdKernel;
  std::shared_ptr<GfxKernel> gfxBwdKernel;
  if (!useCppFallback) {
    const std::filesystem::path shaderDir = ex::getComputeShaderDir();
    const std::array includeDirList = ex::getHlslIncludeDirList();
    gfxProgram = ex::createGfxProgram(gfxContext, "simple_mlp_training_test", shaderDir, includeDirList);

    std::vector kernelDefinitions = buildMlpTestDefinitions(testParams, inputDim, outputDim, hiddenLayerDim, numBackboneLayers + 1, activationHidden, activationLast, hasBias, useSoftwareLinAlgImpl);
    kernelDefinitions.push_back(ex::createOptionString("MINIDXNN_BATCH_SIZE={}", batchSize));
    const ex::OptionString fwdName = ex::createOptionString("testMlpTrainingForwardF{}Kernel", 8 * sizeof(Type));
    gfxFwdKernel = ex::createGfxComputeKernel(gfxContext, *gfxProgram, fwdName.data(), kernelDefinitions);
    const ex::OptionString bwdName = ex::createOptionString("testMlpTrainingBackwardF{}Kernel", 8 * sizeof(Type));
    gfxBwdKernel = ex::createGfxComputeKernel(gfxContext, *gfxProgram, bwdName.data(), kernelDefinitions);
  }
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

  for (size_t trial = 0; trial < numOfTests; ++trial) {
    // Create a random MLP (UV 2D -> RGBA 4D)
    std::vector mlpData = ::createRandomMlp<Type>(inputDim, outputDim, hiddenLayerDim, numBackboneLayers, hasBias, activationHidden, activationLast, rng);
    using LayerT = typename decltype(mlpData)::value_type;

    // Create random inputs and targets for the batch
    std::vector<Type> inputs(batchSize * inputDim);
    std::vector<Type> targets(batchSize * outputDim);
    std::vector<Type> outputs(batchSize * outputDim);
    std::ranges::for_each(inputs, [&rng](Type& v) { v = static_cast<Type>(rng.draw()); });
    std::ranges::for_each(targets, [&rng](Type& v) { v = static_cast<Type>(rng.draw()); });

    // Reset gradients before processing the batch
    std::ranges::for_each(mlpData, [](LayerT& layer) { layer.resetGrads(); });

    // CPU reference: forward + backward
    const float batchScale = 1.0f / static_cast<float>(batchSize);
    const size_t cacheStride = (mlpData.size() - 1) * mlpData.back().inputDimension() + mlpData.back().outputDimension();
    std::vector<Type> logitsCache(batchSize * cacheStride);

    for (size_t s = 0; s < batchSize; ++s) {
      const std::span<const Type> inputSpan{inputs.data() + (s * inputDim), inputDim};
      std::span<Type> logitsSpan{logitsCache.data() + (s * cacheStride), cacheStride};
      std::span<Type> outputSpan{outputs.data() + (s * outputDim), outputDim};
      ex::forward<Type, Type, Type, Type, Type>(outputSpan, inputSpan, mlpData, logitsSpan);

      const std::span<const Type> targetSpan{targets.data() + (s * outputDim), outputDim};
      std::vector lossGradient = ex::mseLossGradient<Type>(outputSpan, targetSpan);
      std::ranges::for_each(lossGradient, [batchScale](Type& v) { v *= batchScale; });

      [[maybe_unused]] const std::vector upstreamGrad = ex::backward<Type, Type, Type, Type, Type>(lossGradient, inputSpan, mlpData, logitsSpan);
    }

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
    if (!useCppFallback) {
      // GPU path
      std::vector<size_t> matrixSizeList;
      matrixSizeList.resize(numBackboneLayers + 1);
      // Create buffers
      std::shared_ptr inputBuffer = ex::createGfxBuffer<Type>(gfxContext, inputs);
      std::shared_ptr targetBuffer = ex::createGfxBuffer<Type>(gfxContext, targets);
      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(gfxContext, outputs.size());
      std::shared_ptr weightBuffer = ex::convertToMatrixBuffer<Type>(gfxContext, mlpData, testParams.m_weightMatrixLayout, matrixSizeList, ex::MATRIX_ALIGNMENT, ex::MATRIX_VECTOR_STRIDE_ALIGNMENT);
      std::shared_ptr biasBuffer = ex::convertToVectorBuffer<Type>(gfxContext, mlpData, ex::VECTOR_ALIGNMENT);
      std::shared_ptr weightGradBuffer = ex::createGfxBuffer<Type>(gfxContext, weightBuffer->getSize() / sizeof(Type));
      const size_t biasStride = biasBuffer->getSize() / sizeof(Type);
      std::shared_ptr biasGradBuffer = ex::createGfxBuffer<Type>(gfxContext, biasStride);
      std::shared_ptr logitsCacheBuffer = ex::createGfxBuffer<Type>(gfxContext, batchSize * biasStride);
      const size_t threadGroupSize = calcThreadGroupSize(batchSize, testParams.m_numThreadsX);
      // Forward pass
      ex::runKernel(gfxContext, *gfxProgram, *gfxFwdKernel, threadGroupSize,
          {
            ex::bind(*inputBuffer, "InputBuffer"),
            ex::bind(*outputBuffer, "OutputBuffer"),
            ex::bind(*weightBuffer, "WeightBuffer"),
            ex::bind(*biasBuffer, "BiasBuffer"),
            ex::bind(*logitsCacheBuffer, "LogitsCacheBuffer"),
          },
          {
            ex::bind(static_cast<std::int32_t>(matrixSizeList.front()), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
            ex::bind(static_cast<std::int32_t>((matrixSizeList.size() > 1) ? matrixSizeList.at(1) : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
            ex::bind(static_cast<std::int32_t>(biasStride * sizeof(Type)), "TEST_BIAS_STRIDE"),
          });
      { // Test outputs
        std::shared_ptr staging = ex::createGfxBuffer<Type>(gfxContext, outputs.size(), kGfxCpuAccess_Read);
        ex::copyBuffer(gfxContext, *outputBuffer, *staging);
        const std::span data = ex::mapToCpu<Type>(gfxContext, *staging);

        const std::string testLabel = std::format("MLP{} forward: outputs", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(outputs, data, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }
      { // test logits
        std::shared_ptr staging = ex::createGfxBuffer<Type>(gfxContext, logitsCacheBuffer->getSize() / sizeof(Type), kGfxCpuAccess_Read);
        ex::copyBuffer(gfxContext, *logitsCacheBuffer, *staging);
        const std::span data = ex::mapToCpu<Type>(gfxContext, *staging);
        std::vector<Type> logitsData;
        logitsData.resize(logitsCache.size());
        for (size_t batchId = 0; batchId < batchSize; ++batchId) {
          for (size_t i = 0; i < mlpData.size(); ++i) {
            const size_t layerInDim = mlpData[i].inputDimension();
            const size_t layerOutDim = mlpData[i].outputDimension();
            const size_t srcIndex = batchId * biasStride + i * ex::alignN<Type>(layerInDim, ex::VECTOR_ALIGNMENT);
            const size_t dstIndex = batchId * cacheStride + i * layerInDim;
            const auto srcBegin = static_cast<std::ptrdiff_t>(srcIndex);
            const auto srcEnd = static_cast<std::ptrdiff_t>(srcIndex + layerOutDim);
            const auto dstBegin = static_cast<std::ptrdiff_t>(dstIndex);
            std::copy(data.begin() + srcBegin, data.begin() + srcEnd, logitsData.begin() + dstBegin);
          }
        }

        const std::string testLabel = std::format("MLP{} forward:  logits", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(logitsCache, logitsData, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }
      // Backward pass
      ex::runKernel(gfxContext, *gfxProgram, *gfxBwdKernel, threadGroupSize,
          {
            ex::bind(*inputBuffer, "InputBuffer"),
            ex::bind(*targetBuffer, "TargetBuffer"),
            ex::bind(*outputBuffer, "OutputBuffer"),
            ex::bind(*weightBuffer, "WeightBuffer"),
            ex::bind(*biasBuffer, "BiasBuffer"),
            ex::bind(*weightGradBuffer, "WeightGradBuffer"),
            ex::bind(*biasGradBuffer, "BiasGradBuffer"),
            ex::bind(*logitsCacheBuffer, "LogitsCacheBuffer"),
          },
          {
            ex::bind(static_cast<std::int32_t>(matrixSizeList.front()), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
            ex::bind(static_cast<std::int32_t>((matrixSizeList.size() > 1) ? matrixSizeList.at(1) : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
            ex::bind(static_cast<std::int32_t>(biasStride * sizeof(Type)), "TEST_BIAS_STRIDE"),
          });
      { // weight grad
        std::shared_ptr staging = ex::createGfxBuffer<Type>(gfxContext, weightGradBuffer->getSize() / sizeof(Type), kGfxCpuAccess_Read);
        ex::copyBuffer(gfxContext, *weightGradBuffer, *staging);
        const std::span data = ex::mapToCpu<Type>(gfxContext, *staging);

        const size_t n = std::transform_reduce(mlpData.begin(), mlpData.end(), static_cast<size_t>(0), std::plus{}, [](const LayerT& layer) -> size_t
        {
          return layer.weightGrads().size();
        });

        std::vector<Type> expectedGrads;
        std::vector<Type> actualGrads;
        expectedGrads.reserve(n);
        actualGrads.reserve(n);

        for (size_t layerIndex = 0, offset = 0; layerIndex < mlpData.size(); ++layerIndex) {
          const LayerT& layer = mlpData[layerIndex];
          const ex::MatrixRef expected = layer.weightGradMatrix();
          const size_t vectorStride = ex::alignN<Type>(expected.columnSize(), ex::MATRIX_VECTOR_STRIDE_ALIGNMENT);
          const size_t matrixStride = ex::alignN<Type>(expected.rowSize() * vectorStride, ex::MATRIX_ALIGNMENT);
          const std::span<Type> actualData{data.data() + offset, matrixStride};
          const ex::MatrixRef<const Type> actual(expected.rowSize(), expected.columnSize(), vectorStride, actualData);
          for (size_t row = 0; row < expected.rowSize(); ++row) {
            for (size_t column = 0; column < expected.columnSize(); ++column) {
              expectedGrads.emplace_back(expected(row, column));
              actualGrads.emplace_back(actual(row, column));
            }
          }
          offset += matrixStride;
        }

        const std::string testLabel = std::format("MLP{} backward: weight grads", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(expectedGrads, actualGrads, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }
      if (hasBias) { // bias grad
        std::shared_ptr staging = ex::createGfxBuffer<Type>(gfxContext, biasGradBuffer->getSize() / sizeof(Type), kGfxCpuAccess_Read);
        ex::copyBuffer(gfxContext, *biasGradBuffer, *staging);
        const std::span data = ex::mapToCpu<Type>(gfxContext, *staging);

        const size_t n = std::transform_reduce(mlpData.begin(), mlpData.end(), static_cast<size_t>(0), std::plus{}, [](const LayerT& layer) -> size_t
        {
          return layer.biasGrads().size();
        });

        std::vector<Type> expectedGrads;
        std::vector<Type> actualGrads;
        expectedGrads.reserve(n);
        actualGrads.reserve(n);

        for (size_t layerIndex = 0, offset = 0; layerIndex < mlpData.size(); ++layerIndex) {
          const LayerT& layer = mlpData[layerIndex];
          const std::span expected = layer.biasGrads();
          const std::span<Type> actual{data.data() + offset, expected.size()};
          expectedGrads.insert(expectedGrads.end(), expected.begin(), expected.end());
          actualGrads.insert(actualGrads.end(), actual.begin(), actual.end());
          const size_t stride = ex::alignN<Type>(expected.size(), ex::VECTOR_ALIGNMENT);
          offset += stride;
        }

        const std::string testLabel = std::format("MLP{} backward:   bias grads", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(expectedGrads, actualGrads, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }
    } else
#endif // !MINIDXNN_CPP_FALLBACK_ONLY
    {
      // C++ fallback path
      const size_t effectiveHiddenDim = (numBackboneLayers == 0)
          ? std::max(inputDim, outputDim)
          : hiddenLayerDim;
      ex::PackedMlpBuffers<Type> packed;
      packed.pack(mlpData, hasBias);

      std::vector<Type> fbOutputs(batchSize * outputDim);
      std::vector<std::uint8_t> weightGradBuf, biasGradBuf;
      const bool dispatched = dispatchBackward<Type>(
          mlpData.size(), effectiveHiddenDim, inputDim, outputDim,
          activationHidden, activationLast,
          packed, inputs, targets, fbOutputs,
          weightGradBuf, biasGradBuf, batchSize, hasBias);
      ASSERT_TRUE(dispatched)
          << "Unsupported backward config: layers=" << mlpData.size()
          << " hiddenDim=" << effectiveHiddenDim
          << " inputDim=" << inputDim
          << " outputDim=" << outputDim;

      // Compare forward outputs
      {
        const std::string testLabel = std::format("CppFallback_Backward{} forward", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(outputs, fbOutputs, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }

      // Compare weight gradients
      {
        const std::vector<Type> expectedGrads = ex::collectWeightGrads<Type>(mlpData);
        const std::vector<Type> actualGrads = ex::unpackWeightGrads<Type>(weightGradBuf, mlpData, packed.matrixSizes);

        const std::string testLabel = std::format("CppFallback_Backward{} weight grads", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(expectedGrads, actualGrads, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }

      // Compare bias gradients
      if (hasBias) {
        const std::vector<Type> expectedGrads = ex::collectBiasGrads<Type>(mlpData);
        const std::vector<Type> actualGrads = ex::unpackBiasGrads<Type>(biasGradBuf, mlpData, effectiveHiddenDim);

        const std::string testLabel = std::format("CppFallback_Backward{} bias grads", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(expectedGrads, actualGrads, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }
    }
  }
}

} // namespace

// ============================================================================
// GPU test macros (CoopVecTest)
// ============================================================================

#ifndef MINIDXNN_CPP_FALLBACK_ONLY

// --- Linear algebra test macros ---
#define ADD_MATRIX_MUL_TEST(typeName, type, inputDim, outputDim) \
  TEST_P(CoopVecTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), false, false); \
  } \
  TEST_P(CoopVecTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), false, false); \
  } \
  TEST_P(CoopVecTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), true, false); \
  } \
  TEST_P(CoopVecTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), true, false); \
  } \
  TEST_P(CoopVecTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), false, false); \
  } \
  TEST_P(CoopVecTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), false, false); \
  } \
  TEST_P(CoopVecTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), true, false); \
  } \
  TEST_P(CoopVecTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), true, false); \
  }

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
ADD_MATRIX_MUL_TEST(F32, float, 2, 2);
ADD_MATRIX_MUL_TEST(F32, float, 16, 4);
ADD_MATRIX_MUL_TEST(F32, float, 64, 64);
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

ADD_MATRIX_MUL_TEST(F16, half_float::half, 2, 2);
ADD_MATRIX_MUL_TEST(F16, half_float::half, 16, 4);
ADD_MATRIX_MUL_TEST(F16, half_float::half, 64, 64);

// --- VectorAcc test macros ---
#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
TEST_P(CoopVecTest, VectorAcc_F32_Software)
{
  ::testVectorAcc<float>(params(), context(), true, false);
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

TEST_P(CoopVecTest, VectorAcc_F16_Software)
{
  ::testVectorAcc<half_float::half>(params(), context(), true, false);
}

// --- AtomicFetchAdd test macros ---
#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
TEST_P(CoopVecTest, AtomicFetchAdd_F32)
{
  ::testAtomicFetchAdd<float>(params(), context(), false);
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

TEST_P(CoopVecTest, AtomicFetchAdd_F16)
{
  ::testAtomicFetchAdd<half_float::half>(params(), context(), false);
}

// --- MLP forward test macros ---

#define ADD_SIMPLE_MLP_FORWARD_TEST(typeName, type, layerLabel, inputDim, outputDim, hiddenDim, numBackboneLayers) \
  TEST_P(CoopVecTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_NoBias ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, false, inputDim, outputDim, hiddenDim, numBackboneLayers, false); \
  } \
  TEST_P(CoopVecTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, false, inputDim, outputDim, hiddenDim, numBackboneLayers, true); \
  } \
  TEST_P(CoopVecTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_NoBias_Software ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), true, false, inputDim, outputDim, hiddenDim, numBackboneLayers, false); \
  } \
  TEST_P(CoopVecTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_Software ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), true, false, inputDim, outputDim, hiddenDim, numBackboneLayers, true); \
  } \
  TEST_P(CoopVecTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_NoBias_CppFallback ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, true, inputDim, outputDim, hiddenDim, numBackboneLayers, false); \
  } \
  TEST_P(CoopVecTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_CppFallback ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, true, inputDim, outputDim, hiddenDim, numBackboneLayers, true); \
  }

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
ADD_SIMPLE_MLP_FORWARD_TEST(F32, float, 2x2, 2, 2, 1, 0);
ADD_SIMPLE_MLP_FORWARD_TEST(F32, float, 16x4, 16, 4, 1, 0);
ADD_SIMPLE_MLP_FORWARD_TEST(F32, float, 2x2x2, 2, 2, 2, 1);
ADD_SIMPLE_MLP_FORWARD_TEST(F32, float, 8x16x16x4, 8, 4, 16, 2);
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

ADD_SIMPLE_MLP_FORWARD_TEST(F16, half_float::half, 2x2, 2, 2, 1, 0);
ADD_SIMPLE_MLP_FORWARD_TEST(F16, half_float::half, 16x4, 16, 4, 1, 0);
ADD_SIMPLE_MLP_FORWARD_TEST(F16, half_float::half, 2x2x2, 2, 2, 2, 1);
ADD_SIMPLE_MLP_FORWARD_TEST(F16, half_float::half, 8x16x16x4, 8, 4, 16, 2);


#define ADD_SIMPLE_MLP_BACKWARD_TEST(typeName, type, layerLabel, hiddenDim, numBackboneLayers, batchSize) \
  TEST_P(CoopVecTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_NoBias ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, false, hiddenDim, numBackboneLayers, batchSize, false); \
  } \
  TEST_P(CoopVecTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, false, hiddenDim, numBackboneLayers, batchSize, true); \
  } \
  TEST_P(CoopVecTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_NoBias_Software ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), true, false, hiddenDim, numBackboneLayers, batchSize, false); \
  } \
  TEST_P(CoopVecTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_Software ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), true, false, hiddenDim, numBackboneLayers, batchSize, true); \
  } \
  TEST_P(CoopVecTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_NoBias_CppFallback ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, true, hiddenDim, numBackboneLayers, batchSize, false); \
  } \
  TEST_P(CoopVecTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_CppFallback ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, true, hiddenDim, numBackboneLayers, batchSize, true); \
  }

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
ADD_SIMPLE_MLP_BACKWARD_TEST(F32, float, 2x4, 8, 0, 1);
ADD_SIMPLE_MLP_BACKWARD_TEST(F32, float, 2x8x8x4, 8, 2, 1);
ADD_SIMPLE_MLP_BACKWARD_TEST(F32, float, 2x6x6x6x4, 6, 3, 1);
ADD_SIMPLE_MLP_BACKWARD_TEST(F32, float, 2x4_batch10, 8, 0, 10);
ADD_SIMPLE_MLP_BACKWARD_TEST(F32, float, 2x8x8x4_batch10, 8, 2, 10);
ADD_SIMPLE_MLP_BACKWARD_TEST(F32, float, 2x6x6x6x4_batch10, 6, 3, 10);
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

ADD_SIMPLE_MLP_BACKWARD_TEST(F16, half_float::half, 2x4, 8, 0, 1);
ADD_SIMPLE_MLP_BACKWARD_TEST(F16, half_float::half, 2x8x8x4, 8, 2, 1);
ADD_SIMPLE_MLP_BACKWARD_TEST(F16, half_float::half, 2x6x6x6x4, 6, 3, 1);
ADD_SIMPLE_MLP_BACKWARD_TEST(F16, half_float::half, 2x4_batch10, 8, 0, 10);
ADD_SIMPLE_MLP_BACKWARD_TEST(F16, half_float::half, 2x8x8x4_batch10, 8, 2, 10);
ADD_SIMPLE_MLP_BACKWARD_TEST(F16, half_float::half, 2x6x6x6x4_batch10, 6, 3, 10);

INSTANTIATE_TEST_SUITE_P(CoopVecTest, CoopVecTest, testing::Values(::g_testParams));
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

// ============================================================================
// CppFallbackTest macros
// ============================================================================

// ---- Smoke test ----
TEST_P(CppFallbackTest, Forward_F16_Identity_Sigmoid)
{
  ::testCppFallbackForwardSmoke<half_float::half>(*this);
}

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
TEST_P(CppFallbackTest, Forward_F32_Identity_Sigmoid)
{
  ::testCppFallbackForwardSmoke<float>(*this);
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

// ---- Linear algebra tests ----
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
#define ADD_MATRIX_MUL_FALLBACK_TEST(typeName, type, inputDim, outputDim) \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), false, true); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), false, true); \
  } \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), false, true); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), false, true); \
  }
#else
#define ADD_MATRIX_MUL_FALLBACK_TEST(typeName, type, inputDim, outputDim) \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params()); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params()); \
  } \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params()); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params()); \
  }
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

ADD_MATRIX_MUL_FALLBACK_TEST(F16, half_float::half, 2, 2)
ADD_MATRIX_MUL_FALLBACK_TEST(F16, half_float::half, 16, 4)
ADD_MATRIX_MUL_FALLBACK_TEST(F16, half_float::half, 64, 64)

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
ADD_MATRIX_MUL_FALLBACK_TEST(F32, float, 2, 2)
ADD_MATRIX_MUL_FALLBACK_TEST(F32, float, 16, 4)
ADD_MATRIX_MUL_FALLBACK_TEST(F32, float, 64, 64)
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

// ---- VectorAcc tests ----
TEST_P(CppFallbackTest, VectorAcc_F16)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testVectorAcc<half_float::half>(params(), context(), false, true);
#else
  ::testVectorAcc<half_float::half>(params());
#endif
}

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
TEST_P(CppFallbackTest, VectorAcc_F32)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testVectorAcc<float>(params(), context(), false, true);
#else
  ::testVectorAcc<float>(params());
#endif
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

// ---- AtomicFetchAdd tests ----
TEST_P(CppFallbackTest, AtomicFetchAdd_F16)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testAtomicFetchAdd<half_float::half>(params(), context(), true);
#else
  ::testAtomicFetchAdd<half_float::half>(params());
#endif
}

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
TEST_P(CppFallbackTest, AtomicFetchAdd_F32)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testAtomicFetchAdd<float>(params(), context(), true);
#else
  ::testAtomicFetchAdd<float>(params());
#endif
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
#define ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(typeName, type, layerLabel, inputDim, outputDim, hiddenDim, numBackboneLayers) \
  TEST_P(CppFallbackTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ## _NoBias) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, true, inputDim, outputDim, hiddenDim, numBackboneLayers, false); \
  } \
  TEST_P(CppFallbackTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, true, inputDim, outputDim, hiddenDim, numBackboneLayers, true); \
  }
#else
#define ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(typeName, type, layerLabel, inputDim, outputDim, hiddenDim, numBackboneLayers) \
  TEST_P(CppFallbackTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ## _NoBias) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), inputDim, outputDim, hiddenDim, numBackboneLayers, false); \
  } \
  TEST_P(CppFallbackTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), inputDim, outputDim, hiddenDim, numBackboneLayers, true); \
  }
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(F16, half_float::half, 2x2, 2, 2, 1, 0)
ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(F16, half_float::half, 16x4, 16, 4, 1, 0)
ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(F16, half_float::half, 2x2x2, 2, 2, 2, 1)
ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(F16, half_float::half, 8x16x16x4, 8, 4, 16, 2)

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(F32, float, 2x2, 2, 2, 1, 0)
ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(F32, float, 16x4, 16, 4, 1, 0)
ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(F32, float, 2x2x2, 2, 2, 2, 1)
ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(F32, float, 8x16x16x4, 8, 4, 16, 2)
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

// ---- Backward tests (matching GPU test configurations) ----
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
#define ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(typeName, type, layerLabel, hiddenDim, numBackboneLayers, batchSize) \
  TEST_P(CppFallbackTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ## _NoBias) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, true, hiddenDim, numBackboneLayers, batchSize, false); \
  } \
  TEST_P(CppFallbackTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, true, hiddenDim, numBackboneLayers, batchSize, true); \
  }
#else
#define ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(typeName, type, layerLabel, hiddenDim, numBackboneLayers, batchSize) \
  TEST_P(CppFallbackTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ## _NoBias) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), hiddenDim, numBackboneLayers, batchSize, false); \
  } \
  TEST_P(CppFallbackTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), hiddenDim, numBackboneLayers, batchSize, true); \
  }
#endif // !MINIDXNN_CPP_FALLBACK_ONLY

ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F16, half_float::half, 2x4, 8, 0, 1)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F16, half_float::half, 2x8x8x4, 8, 2, 1)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F16, half_float::half, 2x6x6x6x4, 6, 3, 1)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F16, half_float::half, 2x4_batch10, 8, 0, 10)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F16, half_float::half, 2x8x8x4_batch10, 8, 2, 10)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F16, half_float::half, 2x6x6x6x4_batch10, 6, 3, 10)

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F32, float, 2x4, 8, 0, 1)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F32, float, 2x8x8x4, 8, 2, 1)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F32, float, 2x6x6x6x4, 6, 3, 1)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F32, float, 2x4_batch10, 8, 0, 10)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F32, float, 2x8x8x4_batch10, 8, 2, 10)
ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(F32, float, 2x6x6x6x4_batch10, 6, 3, 10)
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

INSTANTIATE_TEST_SUITE_P(CppFallbackTest, CppFallbackTest, testing::Values(::g_testParams));

namespace {

auto createCommandLineParser(test::TestParameters& params) -> std::unique_ptr<CLI::App>
{
  // Setup CLI11
  const std::string appDesc = "MiniDXNN Unit Tests - Run compute shader tests with configurable parameters";
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
  gtestArgs.reserve(remainingArgs.size() + 1);
  gtestArgs.push_back(argv[0]);
  std::ranges::for_each(remainingArgs, [&gtestArgs](const std::string& arg)
  {
    gtestArgs.push_back(arg.c_str());
  });
  int gtestArgc = static_cast<int>(gtestArgs.size());

  ::testing::InitGoogleTest(&gtestArgc, const_cast<char**>(gtestArgs.data()));
  return RUN_ALL_TESTS();
}
