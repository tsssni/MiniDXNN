/*!
  \file loss.hpp
  \author Sho Ikeda
  \brief Loss function definitions for MLP training
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_LOSS_HPP
#define MINIDXNN_EXAMPLE_LOSS_HPP 1

// Standard C++ library
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>
// Example
#include "utility.hpp"

namespace ex {

enum class LossType
{
  MSE = 0,
};

//! Compute MSE loss: (1/N) * sum((output - target)^2) where N = output.size()
template <Arithmetic Type>
[[nodiscard]] inline
auto mseLoss(const std::span<const Type> output,
             const std::span<const Type> target) noexcept -> Type 
{
  assert(output.size() == target.size());
  const size_t n = output.size();
  auto sum = static_cast<Type>(0);
  for (size_t i = 0; i < n; ++i) {
    const Type diff = output[i] - target[i];
    sum += diff * diff;
  }
  return sum / static_cast<Type>(static_cast<float>(n));
}

//! Compute gradient of MSE loss: dL/dOutput = (2/N) * (output - target)
template <Arithmetic Type>
[[nodiscard]] inline
auto mseLossGradient(const std::span<const Type> output,
                     const std::span<const Type> target) noexcept -> std::vector<Type>
{
  assert(output.size() == target.size());
  const size_t n = output.size();
  const Type scale = static_cast<Type>(2) / static_cast<Type>(static_cast<float>(n));
  std::vector<Type> grad;
  grad.resize(n);
  for (size_t i = 0; i < n; ++i) {
    grad[i] = scale * (output[i] - target[i]);
  }
  return grad;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_LOSS_HPP */
