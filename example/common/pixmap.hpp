/*!
  \file pixmap.hpp
  \author Sho Ikeda
  \brief Pixmap (2D image buffer) template class
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_PIXMAP_HPP
#define MINIDXNN_EXAMPLE_PIXMAP_HPP 1

// Standard C++ library
#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <type_traits>
#include <vector>

namespace ex {

//! 2D pixel buffer template class
template <typename T, size_t CHANNEL = 1>
class Pixmap
{
 public:
  using Type = T;
  using ConstT = std::add_const_t<Type>;
  using Reference = std::add_lvalue_reference_t<Type>;
  using ConstReference = std::add_lvalue_reference_t<ConstT>;
  using ValueT = std::array<Type, CHANNEL>;

  static constexpr size_t NUM_CHANNELS = CHANNEL;


  Pixmap(const size_t width, const size_t height) noexcept
      : m_width{static_cast<std::uint32_t>(width)},
        m_height{static_cast<std::uint32_t>(height)}
  {
    init();
  }


  [[nodiscard]]
  static constexpr auto channel() noexcept -> size_t {return NUM_CHANNELS;}

  [[nodiscard]]
  auto data() noexcept -> std::span<ValueT> {return m_data;}

  [[nodiscard]]
  auto data() const noexcept -> std::span<const ValueT> {return m_data;}

  [[nodiscard]]
  auto height() const noexcept -> size_t {return m_height;}

  [[nodiscard]]
  auto width() const noexcept -> size_t {return m_width;}

  auto set(const size_t row, const size_t col, const ValueT& values) noexcept -> void
  {
    m_data[row * m_width + col] = values;
  }

  [[nodiscard]]
  auto get(const size_t row, const size_t col) const noexcept -> ValueT
  {
    return m_data[row * m_width + col];
  }

  [[nodiscard]]
  auto sample(const float u, const float v) const noexcept -> ValueT
  {
    const auto px = static_cast<size_t>(u * static_cast<float>(m_width - 1));
    const auto py = static_cast<size_t>(v * static_cast<float>(m_height - 1));
    return get(py, px);
  }

 private:
  auto init() noexcept -> void
  {
    m_data.resize(width() * height());
  }


  std::vector<ValueT> m_data;
  std::uint32_t m_width;
  std::uint32_t m_height;
};

static_assert(sizeof(std::uint8_t) == 1);
using PixmapU8 = Pixmap<std::uint8_t>;

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_PIXMAP_HPP */
