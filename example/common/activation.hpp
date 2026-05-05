/*!
  \file activation.hpp
  \author Sho Ikeda
  \brief Activation function definitions and implementations for MLP layers
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
// Example
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

  virtual ~ActivationFunction() = default;

  virtual auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void = 0;

  virtual auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void = 0;
};

template <Arithmetic T>
class IdentityActivation : public ActivationFunction<T>
{
 public:
  using BaseT = ActivationFunction<T>;
  using Type = typename BaseT::Type;
  using ConstT = typename BaseT::ConstT;


  auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;

  auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;
};

template <Arithmetic T>
class SigmoidActivation : public ActivationFunction<T>
{
 public:
  using BaseT = ActivationFunction<T>;
  using Type = typename BaseT::Type;
  using ConstT = typename BaseT::ConstT;


  auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;

  auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;
};

template <Arithmetic T>
class TanhActivation : public ActivationFunction<T>
{
 public:
  using BaseT = ActivationFunction<T>;
  using Type = typename BaseT::Type;
  using ConstT = typename BaseT::ConstT;


  auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;

  auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;
};

template <Arithmetic T>
class ReluActivation : public ActivationFunction<T>
{
 public:
  using BaseT = ActivationFunction<T>;
  using Type = typename BaseT::Type;
  using ConstT = typename BaseT::ConstT;


  auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;

  auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;
};

template <Arithmetic T>
class LeakyReluActivation : public ActivationFunction<T>
{
 public:
  using BaseT = ActivationFunction<T>;
  using Type = typename BaseT::Type;
  using ConstT = typename BaseT::ConstT;


  auto forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;

  auto backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void override;


  Type NEGATIVE_SLOPE = Type{0.01f};
};

[[nodiscard]]
auto getActivationTypeString(const ActivationType type) noexcept -> std::string_view;

[[nodiscard]]
auto getActivationTypeFromString(const std::string_view name) noexcept -> ActivationType;

// Implementation

template <Arithmetic T>
inline
auto IdentityActivation<T>::forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::ranges::copy(input, output.begin());
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
  std::ranges::transform(input, output.begin(), [](const T x) -> T
  {
    using half_float::abs;
    using half_float::exp;
    using std::abs;
    using std::exp;
    const T zero = static_cast<T>(0);
    const T one = static_cast<T>(1);
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
  std::ranges::transform(input, output.begin(), [](const T x) -> T
  {
    using half_float::abs;
    using half_float::exp;
    using std::abs;
    using std::exp;
    const T one = static_cast<T>(1);
    const T e = exp(-abs(x));
    const T oneOverEPlusOne = one / (e + one);
    const T derivative = (one - oneOverEPlusOne) * oneOverEPlusOne;
    return derivative;
  });
}

template <Arithmetic T>
inline
auto TanhActivation<T>::forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::ranges::transform(input, output.begin(), [](const T x) -> T
  {
    using half_float::tanh;
    using std::tanh;
    return tanh(x);
  });
}

template <Arithmetic T>
inline
auto TanhActivation<T>::backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::ranges::transform(input, output.begin(), [](const T x) -> T
  {
    using half_float::tanh;
    using std::tanh;
    const T one = static_cast<T>(1);
    const T t = tanh(x);
    return one - t * t;
  });
}

template <Arithmetic T>
inline
auto ReluActivation<T>::forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::ranges::transform(input, output.begin(), [](const T x) -> T
  {
    using half_float::fmax;
    using std::fmax;
    return fmax(static_cast<T>(0), x);
  });
}

template <Arithmetic T>
inline
auto ReluActivation<T>::backward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::ranges::transform(input, output.begin(), [](const T x) -> T
  {
    const T zero = static_cast<T>(0);
    const T one = static_cast<T>(1);
    return (x > zero) ? one : zero;
  });
}

template <Arithmetic T>
inline
auto LeakyReluActivation<T>::forward(std::span<Type> output, const std::span<ConstT> input) noexcept -> void
{
  std::ranges::transform(input, output.begin(), [this](const T x) -> T
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
  std::ranges::transform(input, output.begin(), [this](const T x) -> T
  {
    const T zero = static_cast<T>(0);
    const T one = static_cast<T>(1);
    const T derivative = (x < zero) ? NEGATIVE_SLOPE : one;
    return derivative;
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
    case ActivationType::TANH: {
      function = std::make_unique<TanhActivation<T>>();
      break;
    }
    case ActivationType::RELU: {
      function = std::make_unique<ReluActivation<T>>();
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
  const std::string_view result = (type == ActivationType::IDENTITY)   ? "IdentityActivation"sv :
                                  (type == ActivationType::SIGMOID)    ? "SigmoidActivation"sv :
                                  (type == ActivationType::TANH)       ? "TanhActivation"sv :
                                  (type == ActivationType::RELU)       ? "ReluActivation"sv :
                                  (type == ActivationType::LEAKY_RELU) ? "LeakyReluActivation"sv
                                                                       : ""sv;
  return result;
}

inline
auto getActivationTypeFromString(const std::string_view name) noexcept -> ActivationType
{
  using namespace std::string_view_literals;
  const ActivationType type = (name == "identity"sv)   ? ActivationType::IDENTITY :
                              (name == "sigmoid"sv)    ? ActivationType::SIGMOID :
                              (name == "tanh"sv)       ? ActivationType::TANH :
                              (name == "relu"sv)       ? ActivationType::RELU :
                              (name == "leaky_relu"sv) ? ActivationType::LEAKY_RELU
                                                       : ActivationType::LEAKY_RELU;
  return type;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_ACTIVATION_HPP */
