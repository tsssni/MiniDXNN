/*!
  \file utility.hpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_UTILITY_HPP
#define MINIDXNN_EXAMPLE_UTILITY_HPP 1

// Standard C++ library
#include <array>
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <istream>
#include <optional>
#include <ostream>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>
// Half
#include "half.hpp"

namespace ex {

//
using OptionString = std::array<char, 256>;

//
template <typename Type>
using OptionalRef = std::optional<std::reference_wrapper<Type>>;

//
template <typename ...Args>
[[nodiscard]]
inline
auto createOptionString(std::format_string<Args...> format, Args&&... args) noexcept -> OptionString
{
  OptionString option{};
  std::format_to(option.begin(), format, std::forward<Args>(args)...);
  return option;
}

//
template <typename Type>
concept Arithmetic = std::is_arithmetic_v<Type> or
                     std::is_same_v<half_float::half, std::remove_cv_t<Type>>;

// 
template <typename Type>
concept FloatingPoint = std::is_floating_point_v<Type> or
                        std::is_same_v<half_float::half, std::remove_cv_t<Type>>;

template <typename Type>
auto read(Type* data, std::istream& input, const std::streamsize size = sizeof(Type)) noexcept -> std::istream&;

template <typename Type>
auto write(const Type* data, std::ostream& output, const std::streamsize size = sizeof(Type)) noexcept -> std::ostream&;

// Impl

template <typename Type> inline
auto read(Type* data, std::istream& input, const std::streamsize size) noexcept -> std::istream&
{
  using CharT = std::istream::char_type;
  static_assert(sizeof(CharT) == 1);
  return input.read(reinterpret_cast<CharT*>(data), size);
}

template <typename Type> inline
auto write(const Type* data, std::ostream& output, const std::streamsize size) noexcept -> std::ostream&
{
  using CharT = std::ostream::char_type;
  static_assert(sizeof(CharT) == 1);
  return output.write(reinterpret_cast<const CharT*>(data), size);
}

inline
constexpr auto align(const size_t sizeInBytes, const size_t alignmentInBytes) noexcept -> size_t
{
  assert(std::has_single_bit(alignmentInBytes));
  const size_t a = alignmentInBytes - 1;
  const size_t bytes = (sizeInBytes + a) bitand (~a);
  return bytes;
}

template <typename Type> inline
constexpr auto alignN(const size_t n, const size_t alignmentInBytes) noexcept -> size_t
{
  assert(std::has_single_bit(alignmentInBytes) && (alignmentInBytes >= sizeof(Type)));
  const size_t bytes = align(n * sizeof(Type), alignmentInBytes);
  return bytes / sizeof(Type);
}

template <Arithmetic Type> inline
auto validateValue(const Type value) noexcept -> void
{
  using std::isnormal;
  using half_float::isnormal;
  const auto zero = static_cast<Type>(0);
  const bool ok = (value == zero) || isnormal(value);
  if (not ok) {
    std::cerr << "value is not normal: " << static_cast<double>(value) << std::endl;
  }
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_UTILITY_HPP */

