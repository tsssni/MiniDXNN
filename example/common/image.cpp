/*!
  \file image.cpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#include "image.hpp"
// Standard C++ library
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <ostream>
#include <string_view>
// Examples
#include "pixmap.hpp"
#include "utility.hpp"

namespace ex {

template <size_t size> struct UIntType;

template <>
struct UIntType<1>
{
  using T = std::uint8_t;
};

template <>
struct UIntType<2>
{
  using T = std::uint16_t;
};

template <typename T>
auto writeAsPpm(const Pixmap<T>& pixmap, std::ostream& output) noexcept -> void
{
  using UIntT = typename UIntType<sizeof(T)>::T;
  using namespace std::string_view_literals;

  std::array<char, 32> buffer;
  auto intToStr = [&buffer](const size_t u) noexcept -> std::string_view
  {
    std::ranges::fill(buffer, '\0');
    std::format_to(buffer.begin(), "{}", u);
    return std::string_view{buffer.data()};
  };

  // Header
  const std::string_view type = (pixmap.channel() == 1) ? "P5"sv : "P6"sv;
  output << type << "\n";
  output << "# comment" << "\n";
  output << intToStr(pixmap.width()) << " " << intToStr(pixmap.height()) << "\n";
  constexpr size_t umax = (std::numeric_limits<UIntT>::max)();
  output << intToStr(umax) << "\n";

  // Data
  write<T>(pixmap.data().data(), output, sizeof(T) * pixmap.data().size());
}

template
auto writeAsPpm<std::uint8_t>(const Pixmap<std::uint8_t>&, std::ostream&) noexcept -> void;

} /* namespace ex */
