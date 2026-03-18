/*!
  \file pixmap.hpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_PIXMAP_HPP
#define MINIDXNN_EXAMPLE_PIXMAP_HPP 1

// Standard C++ library
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <type_traits>
#include <vector>

namespace ex {

//
template <typename T>
class Pixmap
{
 public:
  using Type = T;
  using ConstT = std::add_const_t<Type>;
  using Reference = std::add_lvalue_reference_t<Type>;
  using ConstReference = std::add_lvalue_reference_t<ConstT>;


  Pixmap(const size_t width, const size_t height, const size_t channel) noexcept
      : m_width{static_cast<std::uint32_t>(width)},
        m_height{static_cast<std::uint32_t>(height)},
        m_channel{static_cast<std::uint32_t>(channel)}
  {
    init();
  }


  auto channel() const noexcept -> size_t {return m_channel;}

  auto data() noexcept -> std::span<Type> {return m_data;}

  auto data() const noexcept -> std::span<ConstT> {return m_data;}

  auto height() const noexcept -> size_t {return m_height;}

  auto width() const noexcept -> size_t {return m_width;}

 private:
  auto init() noexcept -> void
  {
    m_data.resize(channel() * width() * height());
  }


  std::vector<Type> m_data;
  std::uint32_t m_width;
  std::uint32_t m_height;
  std::uint32_t m_channel;
};

static_assert(sizeof(std::uint8_t) == 1);
using PixmapU8 = Pixmap<std::uint8_t>;

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_PIXMAP_HPP */
