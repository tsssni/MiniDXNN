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
#include <cstring>
#include <filesystem>
#include <functional>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string_view>
#include <thread>
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
using test::LinearAlgebraMatrixTest;
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
  const double l = static_cast<double>(lhs);
  const double r = static_cast<double>(rhs);
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
                             const bool useSoftwareLinAlgImpl,
                             const ex::MatrixLayout weightMatrixLayout,
                             const size_t matrixAlignment,
                             const size_t vectorStrideAlignment,
                             const size_t biasAlignment) -> std::vector<ex::OptionString>
{
  return {
      ex::createOptionString("MINIDXNN_HAS_BIAS={}", hasBias ? 1 : 0),
      ex::createOptionString("MINIDXNN_INPUT_DIMENSION={}", inputDim),
      ex::createOptionString("MINIDXNN_OUTPUT_DIMENSION={}", outputDim),
      ex::createOptionString("MINIDXNN_HIDDEN_LAYER_DIMENSIONS={}", hiddenLayerDim),
      ex::createOptionString("MINIDXNN_NUM_LAYERS={}", numLayers),
      ex::createOptionString("MINIDXNN_ACTIVATION_HIDDEN_TYPE={}", ex::getActivationTypeString(activationHidden)),
      ex::createOptionString("MINIDXNN_ACTIVATION_LAST_TYPE={}", ex::getActivationTypeString(activationLast)),
      ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", ex::toHlslMatrixLayout(weightMatrixLayout)),
      ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_ALIGNMENT={}", matrixAlignment),
      ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", vectorStrideAlignment),
      ex::createOptionString("MINIDXNN_BIAS_VECTOR_ALIGNMENT={}", biasAlignment),
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
  const double e = static_cast<double>(expected);
  const double v = static_cast<double>(value);
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
    std::cerr << errorMessage.str() << '\n';
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

// C++ fallback helper functions, kernel dispatch, and smoke test
#include "cpp_fallback_path.hpp"

// ----------------------------------------------------------------------------
// Unified linear algebra, vector accumulation, and atomic test functions
// Shared between LinearAlgebraMatrixTest (GPU) and CppFallbackTest (C++ fallback).
// GPU-specific code is guarded by #ifndef MINIDXNN_CPP_FALLBACK_ONLY.
// ----------------------------------------------------------------------------

template <ex::Arithmetic Type, int ROW_SIZE, int COLUMN_SIZE, bool IS_TRANSPOSED, bool HAS_BIAS>
auto testLinearAlgebraMul(const test::TestParameters& testParams,
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
                          GfxContext& gfxContext,
                          const bool useSoftwareLinAlgImpl,
                          const bool useCppFallback,
#endif
                          ex::MatrixLayout weightMatrixLayout,
                          const size_t numOfTests = 5) -> void
{
#ifdef MINIDXNN_CPP_FALLBACK_ONLY
  (void)weightMatrixLayout;
#endif
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

      std::shared_ptr inputBuffer = ex::createGfxBuffer<Type>(gfxContext, inputVec);
      const size_t outputSize = testParams.m_numThreadsX * rowSize;
      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(gfxContext, outputSize);

      // Build D3D12MatrixInfo for the single weight matrix
      ex::D3D12MatrixInfo<Type> weightInfo;
      weightInfo.m_srcData = matrixData;
      if constexpr (IS_TRANSPOSED) {
        // Transposed: physical storage is columnSize × rowSize
        weightInfo.m_rowSize = columnSize;
        weightInfo.m_columnSize = rowSize;
      } else {
        weightInfo.m_rowSize = rowSize;
        weightInfo.m_columnSize = columnSize;
      }
      weightInfo.m_layout = weightMatrixLayout;
      std::vector<ex::D3D12MatrixInfo<Type>> matrixInfoList{weightInfo};
      std::shared_ptr weightBuffer = ex::packAsD3D12MatrixBuffer<Type>(gfxContext, matrixInfoList);
      ASSERT_TRUE(weightBuffer) << "packAsD3D12MatrixBuffer failed for layout " << static_cast<int>(weightMatrixLayout);

      // Build D3D12VectorInfo for bias
      ex::D3D12VectorInfo<Type> biasInfo;
      biasInfo.m_srcData = bias;
      std::vector<ex::D3D12VectorInfo<Type>> vectorInfoList{biasInfo};
      std::shared_ptr biasBuffer = ex::packAsD3D12VectorBuffer<Type>(gfxContext, vectorInfoList);

      {
        const ex::OptionString kernelName = ex::createOptionString("testLinearAlgebraMulF{}Kernel", 8 * sizeof(Type));
        const std::array kernelDefinitions = std::to_array<ex::OptionString>({
            ex::createOptionString("MINIDXNN_HAS_BIAS={}", HAS_BIAS ? 1 : 0),
            ex::createOptionString("MINIDXNN_ROW_SIZE={}", rowSize),
            ex::createOptionString("MINIDXNN_COLUMN_SIZE={}", columnSize),
            ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_LAYOUT={}", ex::toHlslMatrixLayout(matrixInfoList[0].m_layout)),
            ex::createOptionString("MINIDXNN_WEIGHT_MATRIX_IS_TRANSPOSED={}", IS_TRANSPOSED ? 1 : 0),
            ex::createOptionString("MINIDXNN_MATRIX_VECTOR_STRIDE_ALIGNMENT={}", matrixInfoList[0].m_vectorStrideAlignment),
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
      const Type expected = static_cast<Type>(static_cast<float>(numTasks << i));
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
  constexpr size_t size = 8;
  // For F16, cap thread count so the total sum stays within half precision range
  // (half can only accumulate small integers exactly up to 2^11 = 2048).
  // For F32, use high thread count (2^17) for maximum CAS contention.
  constexpr size_t numTasks = sizeof(Type) >= 4
      ? (1ull << (std::min)(std::numeric_limits<Type>::digits, 17))
      : 512;
  // High iteration count to maximize CAS contention and expose optimization bugs
  constexpr size_t numIterations = sizeof(Type) >= 4 ? 32 : 1;

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
            ex::createOptionString("MINIDXNN_NUM_ITERATIONS={}", numIterations),
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

      cppFallbackAtomicFetchAdd<Type>(outputBuf, numTasks, numIterations);

      for (size_t i = 0; i < size; ++i)
        std::memcpy(&result[i], outputData.data() + i * sizeof(Type), sizeof(Type));
    }

    const auto assertTest = [&testParams](const char* expectedLabel, const char* valueLabel, const Type expected, const Type value)
    {
      return assertSimilarity(expectedLabel, valueLabel, expected, value, testParams.m_similarityThreshold);
    };

    const double N = static_cast<double>(numTasks);
    const double I = static_cast<double>(numIterations);
    // Slot 0-3: constant accumulation
    for (size_t i = 0; i < 4; ++i) {
      const Type expected = static_cast<Type>(static_cast<float>(N * I * static_cast<double>(1u << i)));
      EXPECT_PRED_FORMAT2(assertTest, expected, result[i]);
    }
    // Slot 4: 0.125 * N * I
    {
      const Type expected = static_cast<Type>(static_cast<float>(N * I * 0.125));
      EXPECT_PRED_FORMAT2(assertTest, expected, result[4]);
    }
    // Slot 5: 0.0625 * N * I
    {
      const Type expected = static_cast<Type>(static_cast<float>(N * I * 0.0625));
      EXPECT_PRED_FORMAT2(assertTest, expected, result[5]);
    }
    // Slot 6: alternating +1/-1 by thread ID; even threads add +1, odd add -1
    // Net = numIterations * (numEvenThreads - numOddThreads) where numTasks is even → net = 0
    {
      const bool evenTasks = (numTasks & 1u) == 0u;
      const double net = evenTasks ? 0.0 : I;
      const Type expected = static_cast<Type>(static_cast<float>(net));
      // Near-zero expected value: use absolute tolerance
      EXPECT_NEAR(static_cast<double>(result[6]), static_cast<double>(expected),
                  std::max(1.0, static_cast<double>(numTasks) * 0.01));
    }
    // Slot 7: 3 * N * I
    {
      const Type expected = static_cast<Type>(static_cast<float>(N * I * 3.0));
      EXPECT_PRED_FORMAT2(assertTest, expected, result[7]);
    }
  }
}

template <ex::Arithmetic Type, int ROW_SIZE, int COLUMN_SIZE>
auto testOuterProductAcc(const test::TestParameters& testParams,
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
                         GfxContext& gfxContext,
                         const bool useSoftwareLinAlgImpl,
                         const bool useCppFallback,
#endif
                         [[maybe_unused]] ex::MatrixLayout weightMatrixLayout,
                         const size_t numOfTests = 5) -> void
{
  // For F16, cap tasks to stay within half precision range
  constexpr size_t numTasks = sizeof(Type) >= 4
      ? (1ull << (std::min)(std::numeric_limits<Type>::digits, 16))
      : 512;

  for (size_t trial = 0; trial < numOfTests; ++trial) {
    // Compute expected outer product accumulation:
    // lhs[i] = i+1, rhs[j] = 1.0 => matrix[i][j] = numTasks * (i+1)
    constexpr size_t rowSize = static_cast<size_t>(ROW_SIZE);
    constexpr size_t columnSize = static_cast<size_t>(COLUMN_SIZE);
    const size_t vectorStride = ex::alignBytes(columnSize * sizeof(Type), ex::MATRIX_VECTOR_STRIDE_ALIGNMENT);
    const size_t totalBytes = ex::alignBytes(rowSize * vectorStride, ex::MATRIX_ALIGNMENT);

    std::vector<Type> result(rowSize * columnSize, static_cast<Type>(0));

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
    if (!useCppFallback) {
      // GPU path (placeholder — currently delegates to SW via outerProductAccHW)
      (void)gfxContext;
      (void)useSoftwareLinAlgImpl;
      // TODO: Add GPU kernel dispatch for outerProductAcc test
      // For now, fall through to C++ fallback
      std::vector<std::uint8_t> outputData(totalBytes, 0);
      RWByteAddressBuffer outputBuf{outputData};

      cppFallbackOuterProductAcc<Type, ROW_SIZE, COLUMN_SIZE>(outputBuf, vectorStride, numTasks);

      for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < columnSize; ++c)
          std::memcpy(&result[r * columnSize + c],
                      outputData.data() + r * vectorStride + c * sizeof(Type),
                      sizeof(Type));
    } else
#endif // !MINIDXNN_CPP_FALLBACK_ONLY
    {
      std::vector<std::uint8_t> outputData(totalBytes, 0);
      RWByteAddressBuffer outputBuf{outputData};

      cppFallbackOuterProductAcc<Type, ROW_SIZE, COLUMN_SIZE>(outputBuf, vectorStride, numTasks);

      for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < columnSize; ++c)
          std::memcpy(&result[r * columnSize + c],
                      outputData.data() + r * vectorStride + c * sizeof(Type),
                      sizeof(Type));
    }

    const auto assertTest = [&testParams](const char* expectedLabel, const char* valueLabel, const Type expected, const Type value)
    {
      return assertSimilarity(expectedLabel, valueLabel, expected, value, testParams.m_similarityThreshold);
    };

    for (size_t r = 0; r < rowSize; ++r) {
      for (size_t c = 0; c < columnSize; ++c) {
        // lhs[r] = r+1, rhs[c] = 1.0 => expected = numTasks * (r+1)
        const Type expected = static_cast<Type>(static_cast<float>(numTasks * (r + 1)));
        EXPECT_PRED_FORMAT2(assertTest, expected, result[r * columnSize + c]);
      }
    }
  }
}
// ============================================================================
// Unified MLP test functions — shared between LinearAlgebraMatrixTest and CppFallbackTest
// ============================================================================

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
                               ex::MatrixLayout weightMatrixLayout,
                               const size_t numOfTests = 5) -> void
{
#ifdef MINIDXNN_CPP_FALLBACK_ONLY
  (void)weightMatrixLayout;
#endif
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

      // Build D3D12MatrixInfo list from MLP layers
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

      // Create buffers
      std::shared_ptr inputBuffer = ex::createGfxBuffer<Type>(gfxContext, inputs);
      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(gfxContext, references.size());
      std::shared_ptr weightBuffer = ex::packAsD3D12MatrixBuffer<Type>(gfxContext, matrixInfoList);
      ASSERT_TRUE(weightBuffer) << "packAsD3D12MatrixBuffer failed for layout " << static_cast<int>(weightMatrixLayout);

      // Build D3D12VectorInfo list from MLP layers
      std::vector<ex::D3D12VectorInfo<Type>> vectorInfoList;
      vectorInfoList.reserve(mlpData.size());
      for (const auto& layer : mlpData) {
        ex::D3D12VectorInfo<Type> info;
        info.m_srcData = layer.biasData();
        vectorInfoList.push_back(info);
      }
      std::shared_ptr biasBuffer = ex::packAsD3D12VectorBuffer<Type>(gfxContext, vectorInfoList);

      // Create and run the test kernel
      {
        const ex::OptionString kernelName = ex::createOptionString("testMlpInferenceF{}Kernel", 8 * sizeof(Type));
        std::vector kernelDefinitions = buildMlpTestDefinitions(testParams, inputDim, outputDim, hiddenLayerDim, numBackboneLayers + 1, activationHidden, activationLast, hasBias, useSoftwareLinAlgImpl, matrixInfoList[0].m_layout, matrixInfoList[0].m_alignment, matrixInfoList[0].m_vectorStrideAlignment, vectorInfoList[0].m_alignment);
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
              ex::bind(static_cast<std::int32_t>(matrixInfoList.front().m_dataSize), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
              ex::bind(static_cast<std::int32_t>((matrixInfoList.size() > 1) ? matrixInfoList.at(1).m_dataSize : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
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
                                ex::MatrixLayout weightMatrixLayout,
                                const size_t numOfTests = 5) -> void
{
#ifdef MINIDXNN_CPP_FALLBACK_ONLY
  (void)weightMatrixLayout;
#endif
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
    // Pre-check: verify the GPU supports matrix conversion for optimal layouts
    if (ex::needsMatrixConversion(weightMatrixLayout)) {
      constexpr uint32_t dataType = ex::toD3D12DataType<Type>();
      const uint32_t d3dLayout = ex::toD3D12MatrixLayout(weightMatrixLayout);
      ASSERT_NE(gfxGetMatrixMemorySize(gfxContext, static_cast<uint32_t>(hiddenLayerDim), static_cast<uint32_t>(inputDim), d3dLayout, dataType, 0), 0u)
          << "GPU does not support layout " << static_cast<int>(weightMatrixLayout) << " conversion";
    }

    // Query representative D3D12 info for kernel define values
    ex::D3D12MatrixInfo<Type> representativeMatrixInfo;
    representativeMatrixInfo.m_rowSize = hiddenLayerDim;
    representativeMatrixInfo.m_columnSize = inputDim;
    representativeMatrixInfo.m_layout = weightMatrixLayout;
    ex::getD3D12MatrixInfo(representativeMatrixInfo);
    ex::D3D12VectorInfo<Type> representativeVectorInfo;
    representativeVectorInfo.m_srcData = {};
    ex::getD3D12VectorInfo(representativeVectorInfo);

    const std::filesystem::path shaderDir = ex::getComputeShaderDir();
    const std::array includeDirList = ex::getHlslIncludeDirList();
    gfxProgram = ex::createGfxProgram(gfxContext, "simple_mlp_training_test", shaderDir, includeDirList);

    std::vector kernelDefinitions = buildMlpTestDefinitions(testParams, inputDim, outputDim, hiddenLayerDim, numBackboneLayers + 1, activationHidden, activationLast, hasBias, useSoftwareLinAlgImpl, representativeMatrixInfo.m_layout, representativeMatrixInfo.m_alignment, representativeMatrixInfo.m_vectorStrideAlignment, representativeVectorInfo.m_alignment);
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
      // Build D3D12 matrix info list from MLP layers
      std::vector<ex::D3D12MatrixInfo<Type>> matrixInfoList;
      matrixInfoList.reserve(mlpData.size());
      for (const LayerT& layer : mlpData) {
        ex::D3D12MatrixInfo<Type> info;
        info.m_srcData = layer.weightData();
        info.m_rowSize = layer.outputDimension();
        info.m_columnSize = layer.inputDimension();
        info.m_layout = weightMatrixLayout;
        matrixInfoList.push_back(info);
      }
      // Build D3D12 vector info list from MLP layers
      std::vector<ex::D3D12VectorInfo<Type>> vectorInfoList;
      vectorInfoList.reserve(mlpData.size());
      for (const LayerT& layer : mlpData) {
        ex::D3D12VectorInfo<Type> info;
        info.m_srcData = layer.biasData();
        vectorInfoList.push_back(info);
      }
      // Create buffers
      std::shared_ptr inputBuffer = ex::createGfxBuffer<Type>(gfxContext, inputs);
      std::shared_ptr targetBuffer = ex::createGfxBuffer<Type>(gfxContext, targets);
      std::shared_ptr outputBuffer = ex::createGfxBuffer<Type>(gfxContext, outputs.size());
      std::shared_ptr weightBuffer = ex::packAsD3D12MatrixBuffer<Type>(gfxContext, matrixInfoList);
      ASSERT_TRUE(weightBuffer) << "packAsD3D12MatrixBuffer failed for layout " << static_cast<int>(weightMatrixLayout);
      std::shared_ptr biasBuffer = ex::packAsD3D12VectorBuffer<Type>(gfxContext, vectorInfoList);
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
            ex::bind(static_cast<std::int32_t>(matrixInfoList.front().m_dataSize), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
            ex::bind(static_cast<std::int32_t>((matrixInfoList.size() > 1) ? matrixInfoList.at(1).m_dataSize : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
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
            const std::ptrdiff_t srcBegin = static_cast<std::ptrdiff_t>(srcIndex);
            const std::ptrdiff_t srcEnd = static_cast<std::ptrdiff_t>(srcIndex + layerOutDim);
            const std::ptrdiff_t dstBegin = static_cast<std::ptrdiff_t>(dstIndex);
            std::copy(data.begin() + srcBegin, data.begin() + srcEnd, logitsData.begin() + dstBegin);
          }
        }

        const std::string testLabel = std::format("MLP{} forward:  logits", trial + 1);
        ASSERT_TRUE(assertSimilarityBatch<Type>(logitsCache, logitsData, testParams.m_similarityThreshold, testLabel, testParams.m_enableDebugMode));
      }
      gfxCommandClearBuffer(gfxContext, *weightGradBuffer);
      gfxCommandClearBuffer(gfxContext, *biasGradBuffer);
      gfxFinish(gfxContext);
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
            ex::bind(static_cast<std::int32_t>(matrixInfoList.front().m_dataSize), "TEST_WEIGHT_MATRIX_SIZE_FIRST"),
            ex::bind(static_cast<std::int32_t>((matrixInfoList.size() > 1) ? matrixInfoList.at(1).m_dataSize : 0), "TEST_WEIGHT_MATRIX_SIZE_HIDDEN"),
            ex::bind(static_cast<std::int32_t>(biasStride * sizeof(Type)), "TEST_BIAS_STRIDE"),
          });
      { // weight grad
        const std::vector<Type> actualGrads = ex::unpackD3D12MatrixBuffer<Type>(gfxContext, *weightGradBuffer, matrixInfoList);

        std::vector<Type> expectedGrads;
        expectedGrads.reserve(actualGrads.size());
        for (const LayerT& layer : mlpData) {
          const ex::MatrixRef expected = layer.weightGradMatrix();
          for (size_t row = 0; row < expected.rowSize(); ++row)
            for (size_t column = 0; column < expected.columnSize(); ++column)
              expectedGrads.emplace_back(expected(row, column));
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
// GPU test macros (LinearAlgebraMatrixTest)
// ============================================================================

#ifndef MINIDXNN_CPP_FALLBACK_ONLY

// --- Linear algebra test macros ---
#define ADD_MATRIX_MUL_TEST(typeName, type, inputDim, outputDim) \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _RowMajor ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), false, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _RowMajor ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), false, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _ColumnMajor ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), false, false, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _ColumnMajor ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), false, false, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _MulOptimal ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), false, false, ex::MatrixLayout::MUL_OPTIMAL); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _MulOptimal ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), false, false, ex::MatrixLayout::MUL_OPTIMAL); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _OuterProductOptimal ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), false, false, ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _OuterProductOptimal ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), false, false, ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _RowMajor_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), true, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _RowMajor_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), true, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _ColumnMajor_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), true, false, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _ColumnMajor_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), true, false, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_RowMajor ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), false, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), false, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_ColumnMajor ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), false, false, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), false, false, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_MulOptimal ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), false, false, ex::MatrixLayout::MUL_OPTIMAL); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_MulOptimal) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), false, false, ex::MatrixLayout::MUL_OPTIMAL); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_OuterProductOptimal ) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), false, false, ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_OuterProductOptimal) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), false, false, ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_RowMajor_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), true, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_RowMajor_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), true, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_ColumnMajor_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), true, false, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_ColumnMajor_Software) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), true, false, ex::MatrixLayout::COLUMN_MAJOR); \
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
TEST_P(LinearAlgebraMatrixTest, VectorAcc_F32)
{
  ::testVectorAcc<float>(params(), context(), false, false);
}
TEST_P(LinearAlgebraMatrixTest, VectorAcc_F32_Software)
{
  ::testVectorAcc<float>(params(), context(), true, false);
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

TEST_P(LinearAlgebraMatrixTest, VectorAcc_F16)
{
  ::testVectorAcc<half_float::half>(params(), context(), false, false);
}
TEST_P(LinearAlgebraMatrixTest, VectorAcc_F16_Software)
{
  ::testVectorAcc<half_float::half>(params(), context(), true, false);
}

// --- AtomicFetchAdd test macros ---
#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
TEST_P(LinearAlgebraMatrixTest, AtomicFetchAdd_F32)
{
  ::testAtomicFetchAdd<float>(params(), context(), false);
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

TEST_P(LinearAlgebraMatrixTest, AtomicFetchAdd_F16)
{
  ::testAtomicFetchAdd<half_float::half>(params(), context(), false);
}

// --- OuterProductAcc test macros ---
#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
TEST_P(LinearAlgebraMatrixTest, OuterProductAcc_F32_4x4)
{
  ::testOuterProductAcc<float, 4, 4>(params(), context(), false, false, ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL);
}
TEST_P(LinearAlgebraMatrixTest, OuterProductAcc_F32_4x4_Software)
{
  ::testOuterProductAcc<float, 4, 4>(params(), context(), true, false, ex::MatrixLayout::ROW_MAJOR);
}
TEST_P(LinearAlgebraMatrixTest, OuterProductAcc_F32_8x4)
{
  ::testOuterProductAcc<float, 8, 4>(params(), context(), false, false, ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL);
}
TEST_P(LinearAlgebraMatrixTest, OuterProductAcc_F32_8x4_Software)
{
  ::testOuterProductAcc<float, 8, 4>(params(), context(), true, false, ex::MatrixLayout::ROW_MAJOR);
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS

TEST_P(LinearAlgebraMatrixTest, OuterProductAcc_F16_4x4)
{
  ::testOuterProductAcc<half_float::half, 4, 4>(params(), context(), false, false, ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL);
}
TEST_P(LinearAlgebraMatrixTest, OuterProductAcc_F16_4x4_Software)
{
  ::testOuterProductAcc<half_float::half, 4, 4>(params(), context(), true, false, ex::MatrixLayout::ROW_MAJOR);
}
TEST_P(LinearAlgebraMatrixTest, OuterProductAcc_F16_8x4)
{
  ::testOuterProductAcc<half_float::half, 8, 4>(params(), context(), false, false, ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL);
}
TEST_P(LinearAlgebraMatrixTest, OuterProductAcc_F16_8x4_Software)
{
  ::testOuterProductAcc<half_float::half, 8, 4>(params(), context(), true, false, ex::MatrixLayout::ROW_MAJOR);
}

// --- MLP forward test macros ---

#define ADD_SIMPLE_MLP_FORWARD_TEST(typeName, type, layerLabel, inputDim, outputDim, hiddenDim, numBackboneLayers) \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_NoBias ) \
  { \
    const auto layout = params().m_mlpTestUseRowMajor ? ex::MatrixLayout::ROW_MAJOR : ex::MatrixLayout::MUL_OPTIMAL; \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, false, inputDim, outputDim, hiddenDim, numBackboneLayers, false, layout); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ) \
  { \
    const auto layout = params().m_mlpTestUseRowMajor ? ex::MatrixLayout::ROW_MAJOR : ex::MatrixLayout::MUL_OPTIMAL; \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, false, inputDim, outputDim, hiddenDim, numBackboneLayers, true, layout); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_NoBias_Software ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), true, false, inputDim, outputDim, hiddenDim, numBackboneLayers, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_Software ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), true, false, inputDim, outputDim, hiddenDim, numBackboneLayers, true, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_NoBias_CppFallback ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, true, inputDim, outputDim, hiddenDim, numBackboneLayers, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ##_CppFallback ) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, true, inputDim, outputDim, hiddenDim, numBackboneLayers, true, ex::MatrixLayout::ROW_MAJOR); \
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
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_NoBias ) \
  { \
    const auto layout = params().m_mlpTestUseRowMajor ? ex::MatrixLayout::ROW_MAJOR : ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL; \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, false, hiddenDim, numBackboneLayers, batchSize, false, layout); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ) \
  { \
    const auto layout = params().m_mlpTestUseRowMajor ? ex::MatrixLayout::ROW_MAJOR : ex::MatrixLayout::OUTER_PRODUCT_OPTIMAL; \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, false, hiddenDim, numBackboneLayers, batchSize, true, layout); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_NoBias_Software ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), true, false, hiddenDim, numBackboneLayers, batchSize, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_Software ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), true, false, hiddenDim, numBackboneLayers, batchSize, true, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_NoBias_CppFallback ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, true, hiddenDim, numBackboneLayers, batchSize, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(LinearAlgebraMatrixTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ##_CppFallback ) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, true, hiddenDim, numBackboneLayers, batchSize, true, ex::MatrixLayout::ROW_MAJOR); \
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

// ============================================================================
// LinAlg Matrix Feature Support Tests
// ============================================================================

// D3D12_LINEAR_ALGEBRA_DATATYPE values
namespace LinAlgDataType {
  constexpr uint32_t kSINT8 = 18;
  constexpr uint32_t kUINT8 = 19;
  constexpr uint32_t kSINT32 = 4;
  constexpr uint32_t kFLOAT16 = 7;
  constexpr uint32_t kFLOAT32 = 8;
  constexpr uint32_t kFLOAT8_E4M3FN = 20;
  constexpr uint32_t kFLOAT8_E5M2 = 21;
}

TEST_P(LinearAlgebraMatrixTest, CheckFeatureSupport_LinearAlgebraTier)
{
  const uint32_t tier = gfxGetLinearAlgebraTier(context());
  const std::string tierName = gfxGetLinearAlgebraTierName(context());
  std::cout << std::format("[LinAlg] Linear Algebra Tier: {} (0x{:x})\n", tierName, tier);
  EXPECT_TRUE(true) << "CheckFeatureSupport query completed";
}

TEST_P(LinearAlgebraMatrixTest, CheckFeatureSupport_MatrixMultiply)
{
  using namespace LinAlgDataType;
  const uint32_t tier = gfxGetLinearAlgebraTier(context());
  if (tier == 0) {
    GTEST_SKIP() << "Linear algebra not supported on this device";
  }

  struct MulTestCase {
    const char* name;
    uint32_t vecType;
    uint32_t matType;
    uint32_t resType;
  };

  const MulTestCase testCases[] = {
    {"(Sint8, Sint8, Sint32)",       kSINT8,    kSINT8,        kSINT32},
    {"(Sint32, Sint8, Sint32)",      kSINT32,   kSINT8,        kSINT32},
    {"(Uint8, Uint8, Sint32)",       kUINT8,    kUINT8,        kSINT32},
    {"(Sint32, Uint8, Sint32)",      kSINT32,   kUINT8,        kSINT32},
    {"(Fp32, Sint8, Sint32)",       kFLOAT32,    kSINT8,        kSINT32},
    {"(Fp16, Fp16, Fp16)",           kFLOAT16,  kFLOAT16,      kFLOAT16},
    {"(Fp16, Fp16, Fp32)",           kFLOAT16,  kFLOAT16,      kFLOAT32},
    {"(Fp32, Fp32, Fp32)",           kFLOAT32,  kFLOAT32,      kFLOAT32},
    {"(Fp16, Fp8_E4M3, Fp16)",       kFLOAT16,  kFLOAT8_E4M3FN, kFLOAT16},
    {"(Fp16, Fp8_E5M2, Fp16)",       kFLOAT16,  kFLOAT8_E5M2,  kFLOAT16},
  };

  std::cout << "\n[LinAlg] === Matrix Multiplication Support ===" << '\n';
  std::cout << std::left;
  std::cout << "  " << std::setw(28) << "Combination"
            << std::setw(12) << "Supported"
            << std::setw(12) << "HW Accel"
            << "Transpose" << '\n';
  std::cout << "  " << std::string(52, '-') << '\n';

  for (const auto& tc : testCases) {
    auto result = gfxCheckMatrixMultiplySupport(context(), tc.vecType, tc.matType, tc.resType);
    std::cout << "  " << std::setw(28) << tc.name
              << std::setw(12) << (result.supported ? "YES" : "no")
              << std::setw(12) << (result.hardwareAccelerated ? "YES" : "no")
              << (result.transposeSupported ? "YES" : "no") << '\n';
  }
}

TEST_P(LinearAlgebraMatrixTest, CheckFeatureSupport_MatrixMultiplyAdd)
{
  using namespace LinAlgDataType;
  const uint32_t tier = gfxGetLinearAlgebraTier(context());
  if (tier == 0) {
    GTEST_SKIP() << "Linear algebra not supported on this device";
  }

  struct MulAddTestCase {
    const char* name;
    uint32_t vecType;
    uint32_t matType;
    uint32_t biasType;
    uint32_t resType;
  };

  const MulAddTestCase testCases[] = {
    {"(Fp16, Fp16, Fp16, Fp16)",         kFLOAT16,  kFLOAT16,      kFLOAT16,  kFLOAT16},
    {"(Fp16, Fp16, Fp32, Fp32)",         kFLOAT16,  kFLOAT16,      kFLOAT32,  kFLOAT32},
    {"(Fp32, Fp32, Fp32, Fp32)",         kFLOAT32,  kFLOAT32,      kFLOAT32,  kFLOAT32},
    {"(Fp16, Fp8_E4M3, Fp16, Fp16)",     kFLOAT16,  kFLOAT8_E4M3FN, kFLOAT16,  kFLOAT16},
    {"(Fp16, Fp8_E5M2, Fp16, Fp16)",     kFLOAT16,  kFLOAT8_E5M2,  kFLOAT16,  kFLOAT16},
  };

  std::cout << "\n[LinAlg] === Matrix Multiplication + Add Support ===" << '\n';
  std::cout << std::left;
  std::cout << "  " << std::setw(32) << "Combination"
            << std::setw(12) << "Supported"
            << std::setw(12) << "HW Accel"
            << "Transpose" << '\n';
  std::cout << "  " << std::string(56, '-') << '\n';

  for (const auto& tc : testCases) {
    auto result = gfxCheckMatrixMultiplyAddSupport(context(), tc.vecType, tc.matType, tc.biasType, tc.resType);
    std::cout << "  " << std::setw(32) << tc.name
              << std::setw(12) << (result.supported ? "YES" : "no")
              << std::setw(12) << (result.hardwareAccelerated ? "YES" : "no")
              << (result.transposeSupported ? "YES" : "no") << '\n';
  }
}

TEST_P(LinearAlgebraMatrixTest, CheckFeatureSupport_OuterProduct)
{
  using namespace LinAlgDataType;
  const uint32_t tier = gfxGetLinearAlgebraTier(context());
  if (tier == 0) {
    GTEST_SKIP() << "Linear algebra not supported on this device";
  }

  struct OuterProductTestCase {
    const char* name;
    uint32_t inputType;
    uint32_t resultType;
  };

  const OuterProductTestCase testCases[] = {
    {"(Fp16, Fp16)", kFLOAT16, kFLOAT16},
    {"(Fp16, Fp32)", kFLOAT16, kFLOAT32},
    {"(Fp32, Fp32)", kFLOAT32, kFLOAT32},
  };

  std::cout << "\n[LinAlg] === Outer Product Support ===" << '\n';
  std::cout << std::left;
  std::cout << "  " << std::setw(20) << "Combination"
            << "Supported" << '\n';
  std::cout << "  " << std::string(29, '-') << '\n';

  for (const auto& tc : testCases) {
    auto result = gfxCheckMatrixOuterProductSupport(context(), tc.inputType, tc.resultType);
    std::cout << "  " << std::setw(20) << tc.name
              << (result.supported ? "YES" : "no") << '\n';
  }
}

TEST_P(LinearAlgebraMatrixTest, CheckFeatureSupport_AtomicAccumulation)
{
  using namespace LinAlgDataType;
  const uint32_t tier = gfxGetLinearAlgebraTier(context());
  if (tier == 0) {
    GTEST_SKIP() << "Linear algebra not supported on this device";
  }

  struct AtomicTestCase {
    const char* name;
    uint32_t componentType;
  };

  const AtomicTestCase testCases[] = {
    {"Fp16",  kFLOAT16},
    {"Fp32",  kFLOAT32},
  };

  std::cout << "\n[LinAlg] === Atomic Accumulation Support ===" << '\n';
  std::cout << std::left;
  std::cout << "  " << std::setw(12) << "Type"
            << std::setw(24) << "RWByteAddressBuffer"
            << "GroupShared" << '\n';
  std::cout << "  " << std::string(47, '-') << '\n';

  for (const auto& tc : testCases) {
    auto result = gfxCheckMatrixAtomicAccumulationSupport(context(), tc.componentType);
    std::cout << "  " << std::setw(12) << tc.name
              << std::setw(24) << (result.rwByteAddressBufferSupported ? "YES" : "no")
              << (result.groupSharedSupported ? "YES" : "no") << '\n';
  }
}

INSTANTIATE_TEST_SUITE_P(LinearAlgebraMatrixTest, LinearAlgebraMatrixTest, testing::Values(::g_testParams));
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
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), false, true, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), false, true, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), context(), false, true, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), context(), false, true, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), false, true, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), false, true, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), context(), false, true, ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), context(), false, true, ex::MatrixLayout::COLUMN_MAJOR); \
  }
#else
#define ADD_MATRIX_MUL_FALLBACK_TEST(typeName, type, inputDim, outputDim) \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, false>(params(), ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, false, true>(params(), ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_RowMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMul_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, false>(params(), ex::MatrixLayout::COLUMN_MAJOR); \
  } \
  TEST_P(CppFallbackTest, MatrixMulAdd_ ## typeName ## _ ## inputDim ## x ## outputDim ## _Transposed_ColumnMajor) \
  { \
    ::testLinearAlgebraMul<type, inputDim, outputDim, true, true>(params(), ex::MatrixLayout::COLUMN_MAJOR); \
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
TEST_P(CppFallbackTest, VectorAcc_F16_Software)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testVectorAcc<half_float::half>(params(), context(), true, true);
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
TEST_P(CppFallbackTest, VectorAcc_F32_Software)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testVectorAcc<float>(params(), context(), true, true);
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

// ---- OuterProductAcc tests ----
TEST_P(CppFallbackTest, OuterProductAcc_F16_4x4)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testOuterProductAcc<half_float::half, 4, 4>(params(), context(), false, true, ex::MatrixLayout::ROW_MAJOR);
#else
  ::testOuterProductAcc<half_float::half, 4, 4>(params(), ex::MatrixLayout::ROW_MAJOR);
#endif
}
TEST_P(CppFallbackTest, OuterProductAcc_F16_4x4_Software)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testOuterProductAcc<half_float::half, 4, 4>(params(), context(), true, true, ex::MatrixLayout::ROW_MAJOR);
#else
  ::testOuterProductAcc<half_float::half, 4, 4>(params(), ex::MatrixLayout::ROW_MAJOR);
#endif
}
TEST_P(CppFallbackTest, OuterProductAcc_F16_8x4)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testOuterProductAcc<half_float::half, 8, 4>(params(), context(), false, true, ex::MatrixLayout::ROW_MAJOR);
#else
  ::testOuterProductAcc<half_float::half, 8, 4>(params(), ex::MatrixLayout::ROW_MAJOR);
#endif
}
TEST_P(CppFallbackTest, OuterProductAcc_F16_8x4_Software)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testOuterProductAcc<half_float::half, 8, 4>(params(), context(), true, true, ex::MatrixLayout::ROW_MAJOR);
#else
  ::testOuterProductAcc<half_float::half, 8, 4>(params(), ex::MatrixLayout::ROW_MAJOR);
#endif
}

#if defined(MINIDXNN_TEST_ENABLE_FP32_TESTS) && (MINIDXNN_TEST_ENABLE_FP32_TESTS != 0)
TEST_P(CppFallbackTest, OuterProductAcc_F32_4x4)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testOuterProductAcc<float, 4, 4>(params(), context(), false, true, ex::MatrixLayout::ROW_MAJOR);
#else
  ::testOuterProductAcc<float, 4, 4>(params(), ex::MatrixLayout::ROW_MAJOR);
#endif
}
TEST_P(CppFallbackTest, OuterProductAcc_F32_4x4_Software)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testOuterProductAcc<float, 4, 4>(params(), context(), true, true, ex::MatrixLayout::ROW_MAJOR);
#else
  ::testOuterProductAcc<float, 4, 4>(params(), ex::MatrixLayout::ROW_MAJOR);
#endif
}
TEST_P(CppFallbackTest, OuterProductAcc_F32_8x4)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testOuterProductAcc<float, 8, 4>(params(), context(), false, true, ex::MatrixLayout::ROW_MAJOR);
#else
  ::testOuterProductAcc<float, 8, 4>(params(), ex::MatrixLayout::ROW_MAJOR);
#endif
}
TEST_P(CppFallbackTest, OuterProductAcc_F32_8x4_Software)
{
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
  ::testOuterProductAcc<float, 8, 4>(params(), context(), true, true, ex::MatrixLayout::ROW_MAJOR);
#else
  ::testOuterProductAcc<float, 8, 4>(params(), ex::MatrixLayout::ROW_MAJOR);
#endif
}
#endif // MINIDXNN_TEST_ENABLE_FP32_TESTS
#ifndef MINIDXNN_CPP_FALLBACK_ONLY
#define ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(typeName, type, layerLabel, inputDim, outputDim, hiddenDim, numBackboneLayers) \
  TEST_P(CppFallbackTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ## _NoBias) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, true, inputDim, outputDim, hiddenDim, numBackboneLayers, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), context(), false, true, inputDim, outputDim, hiddenDim, numBackboneLayers, true, ex::MatrixLayout::ROW_MAJOR); \
  }
#else
#define ADD_SIMPLE_MLP_FORWARD_FALLBACK_TEST(typeName, type, layerLabel, inputDim, outputDim, hiddenDim, numBackboneLayers) \
  TEST_P(CppFallbackTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel ## _NoBias) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), inputDim, outputDim, hiddenDim, numBackboneLayers, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, SimpleMlpForward_ ## typeName ## _ ## layerLabel) \
  { \
    ::testSimpleMlpForwardFloat<type>(params(), inputDim, outputDim, hiddenDim, numBackboneLayers, true, ex::MatrixLayout::ROW_MAJOR); \
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
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, true, hiddenDim, numBackboneLayers, batchSize, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), context(), false, true, hiddenDim, numBackboneLayers, batchSize, true, ex::MatrixLayout::ROW_MAJOR); \
  }
#else
#define ADD_SIMPLE_MLP_BACKWARD_FALLBACK_TEST(typeName, type, layerLabel, hiddenDim, numBackboneLayers, batchSize) \
  TEST_P(CppFallbackTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel ## _NoBias) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), hiddenDim, numBackboneLayers, batchSize, false, ex::MatrixLayout::ROW_MAJOR); \
  } \
  TEST_P(CppFallbackTest, SimpleMlpBackward_ ## typeName ## _ ## layerLabel) \
  { \
    ::testSimpleMlpBackwardFloat<type>(params(), hiddenDim, numBackboneLayers, batchSize, true, ex::MatrixLayout::ROW_MAJOR); \
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
    const std::string desc = "Use ROW_MAJOR layout instead of MUL_OPTIMAL/OUTER_PRODUCT_OPTIMAL in MLP tests";
    parser->add_flag("--mlp-test-use-row-major", params.m_mlpTestUseRowMajor, desc)
        ->default_val(params.m_mlpTestUseRowMajor);
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
