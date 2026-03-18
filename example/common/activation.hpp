/*!
  \file activation.hpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_ACTIVATION_HPP
#define MINIDXNN_EXAMPLE_ACTIVATION_HPP 1

// Standard C++ library
#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <type_traits>
// Half
#include "half.hpp"
// Test
#include "utility.hpp"

namespace ex {

enum class ActivationType
{
  IDENTITY = 0,
  SIGMOID,
  TANH,
  RELU,
  LEAKY_RELU,
};

template <Arithmetic T>
class ActivationFunction
{
 public:
  using Type = T;
  using ConstT = std::add_const_t<Type>;


  virtual auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void = 0;

  virtual auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void = 0;
};

template <Arithmetic T>
class IdentityActivation : public ActivationFunction<T>
{
 public:
  using BaseT = ActivationFunction<T>;
  using Type = BaseT::Type;
  using ConstT = BaseT::ConstT;


  auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;

  auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;
};

template <Arithmetic T>
class SigmoidActivation : public ActivationFunction<T>
{
 public:
  using BaseT = ActivationFunction<T>;
  using Type = BaseT::Type;
  using ConstT = BaseT::ConstT;


  auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;

  auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;
};

template <Arithmetic T>
class LeakyReluActivation : public ActivationFunction<T>
{
 public:
  using BaseT = ActivationFunction<T>;
  using Type = BaseT::Type;
  using ConstT = BaseT::ConstT;


  auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;

  auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;


  Type NEGATIVE_SLOPE = Type{0.01f};
};

[[nodiscard]]
auto getActivationTypeString(const ActivationType type) noexcept -> std::string_view;

// Implementation

template <Arithmetic T>
inline
auto IdentityActivation<T>::forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::copy(input.begin(), input.end(), output.begin());
}

template <Arithmetic T>
inline
auto IdentityActivation<T>::backward(std::span<Type> output,
                                     [[maybe_unused]] const std::span<ConstT> input) noexcept -> void
{
  std::ranges::fill(output, static_cast<T>(1));
}

template <Arithmetic T>
inline
auto SigmoidActivation<T>::forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::transform(input.begin(), input.end(), output.begin(), [](const T x) -> T
  {
    using half_float::abs;
    using half_float::exp;
    using std::abs;
    using std::exp;
    const auto zero = static_cast<T>(0);
    const auto one = static_cast<T>(1);
    const T e = exp(-abs(x));
    const T oneOverEPlusOne = one / (e + one);
    const T y = (zero > x) ? one - oneOverEPlusOne : oneOverEPlusOne;
    return y;
  });
}

template <Arithmetic T>
inline
auto SigmoidActivation<T>::backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::transform(input.begin(), input.end(), output.begin(), [](const T x) -> T
  {
    using half_float::abs;
    using half_float::exp;
    using std::abs;
    using std::exp;
    const auto one = static_cast<T>(1);
    const T e = exp(-abs(x));
    const T oneOverEPlusOne = one / (e + one);
    const T error = (one - oneOverEPlusOne) * oneOverEPlusOne;
    return error;
  });
}

template <Arithmetic T>
inline
auto LeakyReluActivation<T>::forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::transform(input.begin(), input.end(), output.begin(), [this](const T x) -> T
  {
    using half_float::fmax;
    using std::fmax;
    const T y = fmax(NEGATIVE_SLOPE * x, x);
    return y;
  });
}

template <Arithmetic T>
inline
auto LeakyReluActivation<T>::backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::transform(input.begin(), input.end(), output.begin(), [this](const T x) -> T
  {
    const auto zero = static_cast<T>(0);
    const auto one = static_cast<T>(1);
    const T error = (x < zero) ? NEGATIVE_SLOPE : one;
    return error;
  });
}

template <Arithmetic T>
auto createActivationFunction(const ActivationType type) noexcept -> std::unique_ptr<ActivationFunction<T>>
{
  std::unique_ptr<ActivationFunction<T>> function;
  switch (type) {
    case ActivationType::IDENTITY: {
      function = std::make_unique<IdentityActivation<T>>();
      break;
    }
    case ActivationType::SIGMOID: {
      function = std::make_unique<SigmoidActivation<T>>();
      break;
    }
    case ActivationType::LEAKY_RELU: {
      function = std::make_unique<LeakyReluActivation<T>>();
      break;
    }
    default:
      break;
  }
  return function;
}

inline
auto getActivationTypeString(const ActivationType type) noexcept -> std::string_view
{
  using namespace std::string_view_literals;
  std::string_view result;
  switch (type) {
    case ActivationType::IDENTITY: {
      result = "IdentityActivation"sv;
      break;
    }
    case ActivationType::SIGMOID: {
      result = "SigmoidActivation"sv;
      break;
    }
    case ActivationType::TANH: {
      result = "TanhActivation"sv;
      break;
    }
    case ActivationType::RELU: {
      result = "ReluActivation"sv;
      break;
    }
    case ActivationType::LEAKY_RELU: {
      result = "LeakyReluActivation"sv;
      break;
    }
    default:
      break;
  }
  return result;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_ACTIVATION_HPP */
