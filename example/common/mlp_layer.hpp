/*!
  \file mlp_layer.hpp
  \author Sho Ikeda
  \brief No brief description
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
// Test
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

//
struct LayerConfiguration
{
  size_t m_inputDim;
  size_t m_outputDim;
  ActivationType m_activation;
};

// The base address of matrix resource and matrix offset must be 128-byte aligned. Also note that the size of the underlying allocation is guaranteed to be a multiple of 16 bytes ensuring that the 16 bytes access of the last row/column of the matrix is valid memory.
static constexpr size_t MATRIX_ALIGNMENT = 128;
// The matrix stride is 16-byte aligned.
static constexpr size_t MATRIX_STRIDE_ALIGNMENT = 16;
// The base address of bias vector resource and bias vector offset must be 64-byte aligned.
static constexpr size_t VECTOR_ALIGNMENT = 64;
// GLSL_NV_cooperative_vector requires: offset must be a multiple of 16. For buffer storage, the start of 'buf' must be 16B aligned.
//static constexpr size_t VECTOR_ALIGNMENT = 16;

//
template <Arithmetic WeightT, Arithmetic BiasT>
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


  MlpLayer(const Configuration& configuration) : m_configuration{configuration}
  {
    initialize();
  }


  [[nodiscard]]
  auto biasData() noexcept -> std::span<BiasType> {return m_biasData;}

  [[nodiscard]]
  auto biasData() const noexcept -> std::span<ConstBiasT> {return m_biasData;}

  [[nodiscard]]
  auto configuration() const noexcept -> const Configuration& {return m_configuration;}

  [[nodiscard]]
  auto inputDimension() const noexcept -> size_t {return configuration().m_inputDim;}

  [[nodiscard]]
  auto outputDimension() const noexcept -> size_t {return configuration().m_outputDim;}

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
  }


  std::vector<WeightType> m_weightData;
  std::vector<BiasType> m_biasData;
  Configuration m_configuration;
};

template <Arithmetic WeightT, Arithmetic BiasT> 
[[nodiscard]]
auto createMlp(const std::span<const LayerConfiguration> mlpConfiguration, const bool enableRandomBias, Xoshiro128Plus& rng) noexcept -> std::vector<MlpLayer<WeightT, BiasT>>;

template <Arithmetic OutputT, Arithmetic WeightT, Arithmetic BiasT, Arithmetic InputT, Arithmetic StagingT = InputT>
[[nodiscard]]
auto forward(const std::span<const MlpLayer<WeightT, BiasT>> mlpData, const std::span<const InputT> inputs) noexcept -> std::vector<OutputT>;

// Implementation

namespace {

[[nodiscard]]
auto randn(Xoshiro128Plus& rng) noexcept -> float
{
  const float u1 = rng.draw();
  const float u2 = rng.draw();
  const float u1_safe = std::max(u1, 1e-10f);
  return std::sqrt(-2.0f * std::log(u1_safe)) * std::cos(2.0f * std::numbers::pi_v<float> * u2);
}

} // namespace

//
template <Arithmetic WeightT, Arithmetic BiasT> 
[[nodiscard]] inline
auto createMlp(const std::span<const LayerConfiguration> mlpConfiguration, const bool enableRandomBias, Xoshiro128Plus& rng) noexcept -> std::vector<MlpLayer<WeightT, BiasT>>
{
  using LayerT = MlpLayer<WeightT, BiasT>;

  std::vector<LayerT> mlpData;
  mlpData.reserve(mlpConfiguration.size());
  for (const LayerConfiguration& configuration : mlpConfiguration) {
    LayerT& layer = mlpData.emplace_back(configuration);
    // Initialize the layer with He initialization
    const size_t fanIn = layer.inputDimension();
    const float std = std::sqrt(2.0f / static_cast<float>(fanIn));

    { // Weight: He/Kaiming normal initialization
      // Reference: He et al. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" (2015)
      // W ~ N(0, sqrt(2/fan_in))
      std::ranges::for_each(layer.weightData(), [&rng, std](WeightT& v)
      {
        v = static_cast<WeightT>(randn(rng) * std);
        validateValue(v);
      });
    }
    if (enableRandomBias) {
      // Bias: Uniform initialization (PyTorch's kaiming_uniform_ method)
      // b ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
      // Reference: PyTorch implementation (torch.nn.init.kaiming_uniform_)
      const float bound = 1.0f / std::sqrt(static_cast<float>(fanIn));
      std::ranges::for_each(layer.biasData(), [&rng, bound](BiasT& v)
      {
        v = static_cast<BiasT>((rng.draw() * 2.0f - 1.0f) * bound);
        validateValue(v);
      });
    }
    else {
      std::ranges::for_each(layer.biasData(), [&rng](BiasT& v)
      {
        v = static_cast<BiasT>(0);
      });
    }
  }

  return mlpData;
}

template <Arithmetic OutputT, Arithmetic WeightT, Arithmetic BiasT, Arithmetic InputT, Arithmetic StagingT>
[[nodiscard]] inline
auto forward(const std::span<const MlpLayer<WeightT, BiasT>> mlpData, const std::span<const InputT> inputs) noexcept -> std::vector<OutputT>
{
  const size_t inputDim = mlpData.front().inputDimension();
  const size_t outputDim = mlpData.back().outputDimension();

  std::vector<OutputT> outputs;
  const size_t numTasks = inputs.size() / inputDim;
  outputs.resize(numTasks * outputDim);

  using LayerT = MlpLayer<WeightT, BiasT>;
  const bool hasHiddenLayers = mlpData.size() > 1;
  for (size_t taskId = 0; taskId < numTasks; ++taskId) {
    const std::span<const InputT> input{inputs.data() + (taskId * inputDim), inputDim};
    std::vector<OutputT> output;
    if (hasHiddenLayers) {
      std::vector<StagingT> staging;
      { // First layer
        const LayerT& layer = mlpData.front();
        assert(input.size() == layer.inputDimension());
        staging = mulAdd<StagingT, const WeightT, InputT, BiasT>(layer.weightMatrix(), input, layer.biasData());
        assert(staging.size() == layer.outputDimension());
        std::unique_ptr activation = createActivationFunction<StagingT>(layer.configuration().m_activation);
        activation->forward(staging, staging);
      }
      for (size_t i = 1; i < mlpData.size() - 1; ++i) { // Hidden layers
        const LayerT& layer = mlpData[i];
        assert(staging.size() == layer.inputDimension());
        staging = mulAdd<StagingT, const WeightT, StagingT, BiasT>(layer.weightMatrix(), staging, layer.biasData());
        assert(staging.size() == layer.outputDimension());
        std::unique_ptr activation = createActivationFunction<StagingT>(layer.configuration().m_activation);
        activation->forward(staging, staging);
      }
      { // Output layer
        const LayerT& layer = mlpData.back();
        assert(staging.size() == layer.inputDimension());
        output = mulAdd<OutputT, const WeightT, StagingT, BiasT>(layer.weightMatrix(), staging, layer.biasData());
        assert(output.size() == layer.outputDimension());
        std::unique_ptr activation = createActivationFunction<StagingT>(layer.configuration().m_activation);
        activation->forward(output, output);
      }
    }
    else {
      const LayerT& layer = mlpData[0];
      assert(input.size() == layer.inputDimension());
      output = mulAdd<OutputT, const WeightT, InputT, BiasT>(layer.weightMatrix(), input, layer.biasData());
      assert(output.size() == layer.outputDimension());
      std::unique_ptr activation = createActivationFunction<StagingT>(layer.configuration().m_activation);
      activation->forward(output, output);
    }
    std::copy(output.cbegin(), output.cend(), outputs.begin() + (taskId * outputDim));
  }

  return outputs;
}

template <Arithmetic Type> inline
auto packMatrixData(const std::span<const std::span<const Type>> dataList,
                    const std::span<const std::tuple<size_t, size_t>> layerInfoList,
                    [[maybe_unused]] const MatrixLayout layout,
                    const size_t alignment = 128,
                    const size_t strideAlignment = 16) -> std::vector<Type>
{
  std::vector<Type> memory;
  size_t memorySize = 0;
  for (size_t i = 0; i < dataList.size(); ++i) {
    std::span<Type> data{const_cast<Type*>(dataList[i].data()), dataList[i].size()};
    const auto [inputDim, outputDim] = layerInfoList[i];
    const MatrixRef<Type> matrix{outputDim, inputDim, data};
    const size_t stride = alignN<Type>(matrix.columnSize(), strideAlignment);
    memorySize += alignN<Type>(stride * matrix.rowSize(), alignment);
  }
  memory.resize(memorySize, static_cast<Type>(0));
  for (size_t i = 0, offset = 0; i < dataList.size(); ++i) {
    std::span<Type> data{const_cast<Type*>(dataList[i].data()), dataList[i].size()};
    const auto [inputDim, outputDim] = layerInfoList[i];
    const MatrixRef<Type> matrix{outputDim, inputDim, data};
    const size_t stride = alignN<Type>(matrix.columnSize(), strideAlignment);
    for (size_t r = 0, o = offset; r < matrix.rowSize(); ++r) {
      for (size_t c = 0; c < matrix.columnSize(); ++c) {
        memory[o + c] = matrix(r, c);
      }
      o += stride;
    }
    offset += alignN<Type>(stride * matrix.rowSize(), alignment);
  }
  return memory;
}

template <Arithmetic Type> inline
auto packVectorData(const std::span<const std::span<const Type>> dataList, const size_t alignment = 64) -> std::vector<Type>
{
  std::vector<Type> memory;
  size_t memorySize = 0;
  for (size_t i = 0; i < dataList.size(); ++i) {
    const std::span<const Type> data{dataList[i]};
    memorySize += alignN<Type>(data.size(), alignment);
  }
  memory.resize(memorySize, static_cast<Type>(0));
  for (size_t i = 0, offset = 0; i < dataList.size(); ++i) {
    const std::span<const Type> data{dataList[i]};
    std::ranges::copy(data, memory.begin() + offset);
    offset += alignN<Type>(data.size(), alignment);
  }
  return memory;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_MLP_LAYER_HPP */
