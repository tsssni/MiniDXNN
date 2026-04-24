/*!
  \file mlp_layer.hpp
  \author Sho Ikeda
  \brief MLP layer data structure and CPU-side forward/backward pass implementations
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_MLP_LAYER_HPP
#define MINIDXNN_EXAMPLE_MLP_LAYER_HPP 1

// Standard C++ library
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
// Example
#include "activation.hpp"
#include "matrix.hpp"
#include "utility.hpp"
#include "xoshiro128plus.hpp"

namespace ex {

enum class MatrixLayout
{
  ROW_MAJOR = 0,
  COLUMN_MAJOR,
  MUL_OPTIMAL,
  OUTER_PRODUCT_OPTIMAL,
};

//! MLP layer configuration
struct LayerConfiguration
{
  size_t m_inputDim;
  size_t m_outputDim;
  ActivationType m_activation;
};

// The base address of matrix resource and matrix offset must be 128-byte aligned. Also note that the size of the underlying allocation is guaranteed to be a multiple of 16 bytes ensuring that the 16 bytes access of the last row/column of the matrix is valid memory.
static constexpr size_t MATRIX_ALIGNMENT = 128;
// The matrix stride is 16-byte aligned.
static constexpr size_t MATRIX_VECTOR_STRIDE_ALIGNMENT = 16;
// The base address of bias vector resource and bias vector offset must be 64-byte aligned.
static constexpr size_t VECTOR_ALIGNMENT = 64;
// GLSL_NV_cooperative_vector requires: offset must be a multiple of 16. For buffer storage, the start of 'buf' must be 16B aligned.
//static constexpr size_t VECTOR_ALIGNMENT = 16;

//! MLP layer holding weight/bias data and their gradients
template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
class MlpLayer
{
 public:
  using Configuration = LayerConfiguration;
  using WeightType = WeightT;
  using ConstWeightT = std::add_const_t<WeightType>;
  using ReferenceWeightT = std::add_lvalue_reference_t<WeightType>;
  using ConstReferenceWeightT = std::add_lvalue_reference_t<ConstWeightT>;
  using BiasType = BiasT;
  using ConstBiasT = std::add_const_t<BiasType>;
  using ReferenceBiasT = std::add_lvalue_reference_t<BiasType>;
  using ConstReferenceBiasT = std::add_lvalue_reference_t<ConstBiasT>;
  using WeightGradType = WeightGradT;
  using BiasGradType = BiasGradT;


  MlpLayer(const Configuration& configuration) : m_configuration{configuration}
  {
    initialize();
  }


  [[nodiscard]]
  auto biasData() noexcept -> std::span<BiasType> {return m_biasData;}

  [[nodiscard]]
  auto biasData() const noexcept -> std::span<ConstBiasT> {return m_biasData;}

  [[nodiscard]]
  auto biasGrads() noexcept -> std::span<BiasGradType> {return m_biasGrads;}

  [[nodiscard]]
  auto biasGrads() const noexcept -> std::span<const BiasGradType> {return m_biasGrads;}

  [[nodiscard]]
  auto configuration() const noexcept -> const Configuration& {return m_configuration;}

  [[nodiscard]]
  auto inputDimension() const noexcept -> size_t {return configuration().m_inputDim;}

  [[nodiscard]]
  auto outputDimension() const noexcept -> size_t {return configuration().m_outputDim;}

  auto resetGrads() noexcept -> void
  {
    std::ranges::fill(m_weightGrads, static_cast<WeightGradType>(0));
    std::ranges::fill(m_biasGrads, static_cast<BiasGradType>(0));
  }

  [[nodiscard]]
  auto transposedWeightMatrix() noexcept -> TransposedMatrixRef<WeightType>
  {
    return TransposedMatrixRef<WeightType>{outputDimension(), inputDimension(), weightData()};
  }

  [[nodiscard]]
  auto transposedWeightMatrix() const noexcept -> TransposedMatrixRef<ConstWeightT>
  {
    return TransposedMatrixRef<ConstWeightT>{outputDimension(), inputDimension(), weightData()};
  }

  [[nodiscard]]
  auto weightData() noexcept -> std::span<WeightType> {return m_weightData;}

  [[nodiscard]]
  auto weightData() const noexcept -> std::span<ConstWeightT> {return m_weightData;}

  [[nodiscard]]
  auto weightGrads() noexcept -> std::span<WeightGradType> {return m_weightGrads;}

  [[nodiscard]]
  auto weightGrads() const noexcept -> std::span<const WeightGradType> {return m_weightGrads;}

  [[nodiscard]]
  auto weightGradMatrix() noexcept -> MatrixRef<WeightGradType>
  {
    return MatrixRef<WeightGradType>{outputDimension(), inputDimension(), weightGrads()};
  }

  [[nodiscard]]
  auto weightGradMatrix() const noexcept -> MatrixRef<const WeightGradType>
  {
    return MatrixRef<const WeightGradType>{outputDimension(), inputDimension(), weightGrads()};
  }

  [[nodiscard]]
  auto weightMatrix() noexcept -> MatrixRef<WeightType>
  {
    return MatrixRef<WeightType>{outputDimension(), inputDimension(), weightData()};
  }

  [[nodiscard]]
  auto weightMatrix() const noexcept -> MatrixRef<ConstWeightT>
  {
    return MatrixRef<ConstWeightT>{outputDimension(), inputDimension(), weightData()};
  }

 private:
  auto initialize() noexcept -> void
  {
    m_weightData.resize(inputDimension() * outputDimension());
    m_biasData.resize(outputDimension(), static_cast<BiasType>(0));
    m_weightGrads.resize(inputDimension() * outputDimension(), static_cast<WeightGradType>(0));
    m_biasGrads.resize(outputDimension(), static_cast<BiasGradType>(0));
  }


  std::vector<WeightType> m_weightData;
  std::vector<BiasType> m_biasData;
  std::vector<WeightGradType> m_weightGrads;
  std::vector<BiasGradType> m_biasGrads;
  Configuration m_configuration;
};

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
[[nodiscard]]
auto createMlp(const std::span<const LayerConfiguration> mlpConfiguration, const bool enableRandomBias, Xoshiro128Plus& rng) noexcept -> std::vector<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>>;

template <Arithmetic Type, Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
auto forwardBatch(const std::span<const Type> inputs,
                  const std::span<const MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> mlpData,
                  std::span<Type> logitsCache = {}) noexcept -> std::vector<Type>;

//! Forward pass for a single sample. Stores pre-activation logits in the provided logitsCache.
//! Returns the post-activation output of the last layer.
template <Arithmetic Type, Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
auto forward(std::span<Type> outputs,
             const std::span<const Type> input,
             const std::span<const MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> mlpData,
             std::span<Type> logitsCache = {}) noexcept -> void;

//! Backward pass for a single sample. Computes and accumulates weight/bias gradients into each layer's m_weightGrads/m_biasGrads.
template <Arithmetic Type, Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
[[nodiscard]]
auto backward(const std::span<const Type> lossGrad,
              const std::span<const Type> input,
              std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> mlpData,
              const std::span<Type> logitsCache) noexcept -> std::vector<Type>;

// Implementation

namespace detail {

//! Generate a standard normal random sample using the Box-Muller transform
[[nodiscard]] inline
auto randn(Xoshiro128Plus& rng) noexcept -> float
{
  const float u1 = rng.draw();
  const float u2 = rng.draw();
  const float u1_safe = std::max(u1, 1e-10f);
  return std::sqrt(-2.0f * std::log(u1_safe)) * std::cos(2.0f * std::numbers::pi_v<float> * u2);
}

} // namespace detail

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
[[nodiscard]] inline
auto createMlp(const std::span<const LayerConfiguration> mlpConfiguration, const bool enableRandomBias, Xoshiro128Plus& rng) noexcept -> std::vector<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>>
{
  using LayerT = MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>;

  std::vector<LayerT> mlpData;
  mlpData.reserve(mlpConfiguration.size());
  for (const LayerConfiguration& configuration : mlpConfiguration) {
    LayerT& layer = mlpData.emplace_back(configuration);
    // He/Kaiming normal initialization
    // W ~ N(0, sqrt(2/fan_in))
    // Reference: He et al. "Delving Deep into Rectifiers" (2015)
    const size_t fanIn = layer.inputDimension();
    const float stdDev = std::sqrt(2.0f / static_cast<float>(fanIn));

    {
      std::ranges::for_each(layer.weightData(), [&rng, stdDev](WeightT& v)
      {
        v = static_cast<WeightT>(detail::randn(rng) * stdDev);
        validateValue(v);
      });
    }
    if (enableRandomBias) {
      // Bias: Uniform initialization b ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
      // Reference: PyTorch kaiming_uniform_
      const float bound = 1.0f / std::sqrt(static_cast<float>(fanIn));
      std::ranges::for_each(layer.biasData(), [&rng, bound](BiasT& v)
      {
        v = static_cast<BiasT>((rng.draw() * 2.0f - 1.0f) * bound);
        validateValue(v);
      });
    }
    else {
      std::ranges::fill(layer.biasData(), static_cast<BiasT>(0));
    }
  }

  return mlpData;
}

template <Arithmetic Type, Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
inline
auto forwardBatch(const std::span<const Type> inputs,
                  const std::span<const MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> mlpData,
                  std::span<Type> logitsCache) noexcept -> std::vector<Type>
{
  const size_t inputDim = mlpData.front().inputDimension();
  const size_t outputDim = mlpData.back().outputDimension();
  const size_t numTasks = inputs.size() / inputDim;

  std::vector<Type> outputs;
  outputs.resize(numTasks * outputDim);

  const size_t cacheStride = (mlpData.size() - 1) * mlpData.front().outputDimension() + outputDim;
  for (size_t taskId = 0; taskId < numTasks; ++taskId) {
    std::span<Type> outputSpan{outputs.data() + (taskId * outputDim), outputDim};
    const std::span<const Type> inputSpan{inputs.data() + (taskId * inputDim), inputDim};
    std::span<Type> cache{};
    if (!logitsCache.empty())
      cache = std::span<Type>{logitsCache.data() + (taskId *  cacheStride), cacheStride};
    forward(outputSpan, inputSpan, mlpData, cache);
  }

  return outputs;
}

template <Arithmetic Type, Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
inline
auto forward(std::span<Type> outputs,
             const std::span<const Type> input,
             const std::span<const MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> mlpData,
             std::span<Type> logitsCache) noexcept -> void
{
  const size_t numLayers = mlpData.size();
  std::span<const Type> currentInput = input;
  std::vector<Type> activated;
  activated.resize(mlpData.front().outputDimension());

  for (size_t i = 0; i < numLayers; ++i) {
    const auto& layer = mlpData[i];

    // z = W * input + bias
    const std::vector result = mulAdd<Type, const WeightT, Type, BiasT>(layer.weightMatrix(), currentInput, layer.biasData());

    // Cache pre-activation logits for backward pass
    if (!logitsCache.empty()) {
      std::span<Type> cache{logitsCache.data() + (i * activated.size()), layer.outputDimension()};
      std::ranges::copy(result, cache.begin());
    }

    // Compute activation for next layer's input (not needed for the last layer)
    std::unique_ptr act = createActivationFunction<Type>(layer.configuration().m_activation);
    const bool isLastLayer = i == (numLayers - 1);
    std::span<Type> actOut = isLastLayer ? outputs : std::span<Type>{activated};
    act->forward(actOut, result);
    currentInput = actOut;
  }
}

template <Arithmetic Type, Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
inline
auto backward(const std::span<const Type> lossGrad,
              const std::span<const Type> input,
              std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> mlpData,
              const std::span<Type> logitsCache) noexcept -> std::vector<Type>
{
  const size_t numLayers = mlpData.size();

  std::vector<Type> delta(lossGrad.begin(), lossGrad.end());
  std::vector<Type> actTmp;

  for (size_t layerIdx = numLayers; layerIdx > 0; --layerIdx) {
    const size_t i = layerIdx - 1;
    auto& layer = mlpData[i];
    const size_t outDim = layer.outputDimension();
    const size_t inDim = layer.inputDimension();


    // Compute activation gradient at z[i] (stored in logitsCache)
    {
      const std::span<const Type> logits{logitsCache.data() + (i * mlpData.front().outputDimension()), outDim};
      actTmp.resize(outDim);
      std::span<Type> actError{actTmp};
      std::unique_ptr act = createActivationFunction<Type>(layer.configuration().m_activation);
      act->backward(actError, logits);
      // delta = delta * activation_derivative (element-wise)
      for (size_t j = 0; j < outDim; ++j) {
        delta[j] *= actError[j];
      }
    }

    // Compute input to this layer from logitsCache
    std::span<const Type> inputToLayer;
    if (i == 0) {
      inputToLayer = input;
    } else {
      const std::span<const Type> prevLogits{logitsCache.data() + ((i - 1) * mlpData.front().outputDimension()), inDim};
      actTmp.resize(inDim);
      std::span<Type> layerIn{actTmp};
      std::unique_ptr prevAct = createActivationFunction<Type>(mlpData[i - 1].configuration().m_activation);
      prevAct->forward(layerIn, prevLogits);
      inputToLayer = layerIn;
    }

    // Accumulate weight gradient: dW += delta * a[i]^T (outer product)
    auto wGrads = layer.weightGrads();
    for (size_t r = 0; r < outDim; ++r) {
      for (size_t c = 0; c < inDim; ++c) {
        wGrads[r * inDim + c] += static_cast<WeightGradT>(delta[r] * inputToLayer[c]);
      }
    }

    // Accumulate bias gradient: db += delta
    auto bGrads = layer.biasGrads();
    for (size_t j = 0; j < outDim; ++j) {
      bGrads[j] += static_cast<BiasGradT>(delta[j]);
    }

    // Propagate to previous layer: delta_prev = W^T * delta
    auto transposedW = layer.transposedWeightMatrix();
    std::vector prevDelta = mul<Type>(transposedW, std::span<const Type>(delta));
    delta = std::move(prevDelta);
  }
  return delta;
}

template <Arithmetic Type> inline
auto packMatrixData(const std::span<const std::span<const Type>> dataList,
                    const std::span<const std::tuple<size_t, size_t>> layerInfoList,
                    [[maybe_unused]] const MatrixLayout layout,
                    std::span<size_t> matrixStrideListOut,
                    const size_t alignment = 128,
                    const size_t strideAlignment = 16) -> std::vector<Type>
{
  std::vector<Type> memory;
  size_t memorySize = 0;
  for (size_t i = 0; i < dataList.size(); ++i) {
    std::span<Type> data{const_cast<Type*>(dataList[i].data()), dataList[i].size()};
    const auto [inputDim, outputDim] = layerInfoList[i];
    const MatrixRef<Type> matrix{outputDim, inputDim, data};
    const size_t vectorStride = alignN<Type>(matrix.columnSize(), strideAlignment);
    const size_t stride = alignN<Type>(vectorStride * matrix.rowSize(), alignment);
    matrixStrideListOut[i] = stride * sizeof(Type);
    memorySize += stride;
  }
  memory.resize(memorySize, static_cast<Type>(0));
  for (size_t i = 0, offset = 0; i < dataList.size(); ++i) {
    std::span<Type> data{const_cast<Type*>(dataList[i].data()), dataList[i].size()};
    const auto [inputDim, outputDim] = layerInfoList[i];
    const MatrixRef<Type> matrix{outputDim, inputDim, data};
    const size_t vectorStride = alignN<Type>(matrix.columnSize(), strideAlignment);
    for (size_t r = 0, o = offset; r < matrix.rowSize(); ++r) {
      for (size_t c = 0; c < matrix.columnSize(); ++c) {
        memory[o + c] = matrix(r, c);
      }
      o += vectorStride;
    }
    offset += alignN<Type>(vectorStride * matrix.rowSize(), alignment);
  }
  return memory;
}

template <Arithmetic Type> inline
auto packVectorData(const std::span<const std::span<const Type>> dataList,
                    const size_t alignment = 64) -> std::vector<Type>
{
  std::vector<Type> memory;
  size_t memorySize = 0;
  for (size_t i = 0; i < dataList.size(); ++i) {
    const std::span<const Type> data{dataList[i]};
    const size_t stride = alignN<Type>(data.size(), alignment);
    memorySize += stride;
  }
  memory.resize(memorySize, static_cast<Type>(0));
  for (size_t i = 0, offset = 0; i < dataList.size(); ++i) {
    const std::span<const Type> data{dataList[i]};
    std::ranges::copy(data, memory.begin() + static_cast<std::ptrdiff_t>(offset));
    const size_t stride = alignN<Type>(data.size(), alignment);
    offset += stride;
  }
  return memory;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_MLP_LAYER_HPP */
