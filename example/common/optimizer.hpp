/*!
  \file optimizer.hpp
  \author Sho Ikeda
  \brief Optimizer definitions for MLP training (SGD, Adam, Lion)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_OPTIMIZER_HPP
#define MINIDXNN_EXAMPLE_OPTIMIZER_HPP 1

// Standard C++ library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>
// Example
#include "mlp_layer.hpp"
#include "utility.hpp"

namespace ex {

//! Clamp value to safe range for the target type to prevent FP16 overflow
template <Arithmetic T>
constexpr auto clampForType(const float v) noexcept -> float
{
  if constexpr (sizeof(T) < sizeof(float)) {
    constexpr float maxVal = 65504.0f; // max finite value for IEEE 754 half-precision
    return std::clamp(v, -maxVal, maxVal);
  }
  return v;
}

enum class OptimizerType
{
  SGD = 0,
  ADAM,
  LION,
};

[[nodiscard]]
auto getOptimizerTypeFromString(const std::string_view name) noexcept -> OptimizerType;

[[nodiscard]]
auto getOptimizerTypeString(const OptimizerType type) noexcept -> std::string_view;

//! Base optimizer interface
template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
class Optimizer
{
 public:
  virtual ~Optimizer() = default;

  virtual auto step(std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers, const float lr) noexcept -> void = 0;
};

//! Stochastic Gradient Descent
template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
class SgdOptimizer : public Optimizer<WeightT, BiasT, WeightGradT, BiasGradT>
{
 public:
  auto step(std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers, const float lr) noexcept -> void override;
};

//! Adam optimizer (Kingma & Ba, 2015)
template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
class AdamOptimizer : public Optimizer<WeightT, BiasT, WeightGradT, BiasGradT>
{
 public:
  auto step(std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers, const float lr) noexcept -> void override;

 private:
  auto initializeIfNeeded(const std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers) noexcept -> void;

 public:
  float m_beta1 = 0.9f;
  float m_beta2 = 0.999f;
  float m_epsilon = 1e-8f;

 private:
  std::vector<std::vector<float>> m_mWeights;
  std::vector<std::vector<float>> m_vWeights;
  std::vector<std::vector<float>> m_mBiases;
  std::vector<std::vector<float>> m_vBiases;
  size_t m_timestep = 0;
};

//! Lion optimizer (Chen et al., 2023)
template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
class LionOptimizer : public Optimizer<WeightT, BiasT, WeightGradT, BiasGradT>
{
 public:
  auto step(std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers, const float lr) noexcept -> void override;

 private:
  auto initializeIfNeeded(const std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers) noexcept -> void;

 public:
  float m_beta1 = 0.9f;
  float m_beta2 = 0.99f;
  float m_weightDecay = 0.3f;

 private:
  std::vector<std::vector<float>> m_mWeights;
  std::vector<std::vector<float>> m_mBiases;
};

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT = WeightT, Arithmetic BiasGradT = BiasT>
[[nodiscard]]
auto createOptimizer(const OptimizerType type) noexcept -> std::unique_ptr<Optimizer<WeightT, BiasT, WeightGradT, BiasGradT>>;

// ============================================================================
// Implementation
// ============================================================================

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
auto SgdOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>::step(std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers,
                                         const float lr) noexcept -> void
{
  for (auto& layer : layers) {
    std::span<WeightT> wSpan = layer.weightData();
    std::span<BiasT> bSpan = layer.biasData();
    const std::span<const WeightGradT> wGrads = layer.weightGrads();
    const std::span<const BiasGradT> bGrads = layer.biasGrads();

    for (size_t j = 0; j < wSpan.size(); ++j) {
      const float w = static_cast<float>(wSpan[j]) - lr * static_cast<float>(wGrads[j]);
      wSpan[j] = static_cast<WeightT>(w);
    }
    for (size_t j = 0; j < bSpan.size(); ++j) {
      const float b = static_cast<float>(bSpan[j]) - lr * static_cast<float>(bGrads[j]);
      bSpan[j] = static_cast<BiasT>(b);
    }
  }
}

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
auto AdamOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>::initializeIfNeeded(
    const std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers) noexcept -> void
{
  if (!m_mWeights.empty()) return;
  const size_t numLayers = layers.size();
  m_mWeights.resize(numLayers);
  m_vWeights.resize(numLayers);
  m_mBiases.resize(numLayers);
  m_vBiases.resize(numLayers);
  for (size_t i = 0; i < numLayers; ++i) {
    m_mWeights[i].resize(layers[i].weightData().size(), 0.0f);
    m_vWeights[i].resize(layers[i].weightData().size(), 0.0f);
    m_mBiases[i].resize(layers[i].biasData().size(), 0.0f);
    m_vBiases[i].resize(layers[i].biasData().size(), 0.0f);
  }
}

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
auto AdamOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>::step(std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers,
                                          const float lr) noexcept -> void
{
  initializeIfNeeded(layers);
  m_timestep++;
  const float bc1 = 1.0f - std::pow(m_beta1, static_cast<float>(m_timestep));
  const float bc2 = 1.0f - std::pow(m_beta2, static_cast<float>(m_timestep));

  for (size_t i = 0; i < layers.size(); ++i) {
    std::span<WeightT> wSpan = layers[i].weightData();
    std::span<BiasT> bSpan = layers[i].biasData();
    const std::span<const WeightGradT> wGrads = layers[i].weightGrads();
    const std::span<const BiasGradT> bGrads = layers[i].biasGrads();

    for (size_t j = 0; j < wSpan.size(); ++j) {
      const float g = static_cast<float>(wGrads[j]);
      m_mWeights[i][j] = m_beta1 * m_mWeights[i][j] + (1.0f - m_beta1) * g;
      m_vWeights[i][j] = m_beta2 * m_vWeights[i][j] + (1.0f - m_beta2) * g * g;
      const float mHat = m_mWeights[i][j] / bc1;
      const float vHat = m_vWeights[i][j] / bc2;
      const float w = static_cast<float>(wSpan[j]) - lr * mHat / (std::sqrt(vHat) + m_epsilon);
      wSpan[j] = static_cast<WeightT>(w);
    }
    for (size_t j = 0; j < bSpan.size(); ++j) {
      const float g = static_cast<float>(bGrads[j]);
      m_mBiases[i][j] = m_beta1 * m_mBiases[i][j] + (1.0f - m_beta1) * g;
      m_vBiases[i][j] = m_beta2 * m_vBiases[i][j] + (1.0f - m_beta2) * g * g;
      const float mHat = m_mBiases[i][j] / bc1;
      const float vHat = m_vBiases[i][j] / bc2;
      const float b = static_cast<float>(bSpan[j]) - lr * mHat / (std::sqrt(vHat) + m_epsilon);
      bSpan[j] = static_cast<BiasT>(b);
    }
  }
}

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
auto LionOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>::initializeIfNeeded(
    const std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers) noexcept -> void
{
  if (!m_mWeights.empty()) return;
  const size_t numLayers = layers.size();
  m_mWeights.resize(numLayers);
  m_mBiases.resize(numLayers);
  for (size_t i = 0; i < numLayers; ++i) {
    m_mWeights[i].resize(layers[i].weightData().size(), 0.0f);
    m_mBiases[i].resize(layers[i].biasData().size(), 0.0f);
  }
}

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
auto LionOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>::step(std::span<MlpLayer<WeightT, BiasT, WeightGradT, BiasGradT>> layers,
                                          const float lr) noexcept -> void
{
  initializeIfNeeded(layers);

  // Epsilon-thresholded sign: suppress updates from near-zero residual momentum after convergence
  constexpr float signEps = 1e-6f;
  const auto sign = [](const float x) -> float { return (x > signEps) ? 1.0f : ((x < -signEps) ? -1.0f : 0.0f); };

  for (size_t i = 0; i < layers.size(); ++i) {
    std::span<WeightT> wSpan = layers[i].weightData();
    std::span<BiasT> bSpan = layers[i].biasData();
    const std::span<const WeightGradT> wGrads = layers[i].weightGrads();
    const std::span<const BiasGradT> bGrads = layers[i].biasGrads();

    for (size_t j = 0; j < wSpan.size(); ++j) {
      const float g = static_cast<float>(wGrads[j]);
      const float w = static_cast<float>(wSpan[j]);
      // When gradient is near-zero (converged), clear momentum to prevent
      // stale sign updates from accumulating and overflowing FP16 weights.
      if (std::abs(g) < 1e-8f) {
        m_mWeights[i][j] = 0.0f;
        // Only weight decay when converged
        wSpan[j] = static_cast<WeightT>(clampForType<WeightT>(w - lr * m_weightDecay * w));
        continue;
      }
      // Update uses sign of interpolation between momentum and gradient
      const float update = sign(m_beta1 * m_mWeights[i][j] + (1.0f - m_beta1) * g);
      // Apply weight decay (Chen et al., 2023): θ = θ - η(c + λθ)
      wSpan[j] = static_cast<WeightT>(clampForType<WeightT>(w - lr * (update + m_weightDecay * w)));
      // Momentum update
      m_mWeights[i][j] = m_beta2 * m_mWeights[i][j] + (1.0f - m_beta2) * g;
    }
    for (size_t j = 0; j < bSpan.size(); ++j) {
      const float g = static_cast<float>(bGrads[j]);
      const float b = static_cast<float>(bSpan[j]);
      if (std::abs(g) < 1e-8f) {
        m_mBiases[i][j] = 0.0f;
        bSpan[j] = static_cast<BiasT>(clampForType<BiasT>(b - lr * m_weightDecay * b));
        continue;
      }
      const float update = sign(m_beta1 * m_mBiases[i][j] + (1.0f - m_beta1) * g);
      bSpan[j] = static_cast<BiasT>(clampForType<BiasT>(b - lr * (update + m_weightDecay * b)));
      m_mBiases[i][j] = m_beta2 * m_mBiases[i][j] + (1.0f - m_beta2) * g;
    }
  }
}

template <Arithmetic WeightT, Arithmetic BiasT, Arithmetic WeightGradT, Arithmetic BiasGradT>
[[nodiscard]] inline
auto createOptimizer(const OptimizerType type) noexcept -> std::unique_ptr<Optimizer<WeightT, BiasT, WeightGradT, BiasGradT>>
{
  switch (type) {
    case OptimizerType::SGD:
      return std::make_unique<SgdOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>>();
    case OptimizerType::ADAM:
      return std::make_unique<AdamOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>>();
    case OptimizerType::LION:
      return std::make_unique<LionOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>>();
    default:
      return std::make_unique<SgdOptimizer<WeightT, BiasT, WeightGradT, BiasGradT>>();
  }
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_OPTIMIZER_HPP */
