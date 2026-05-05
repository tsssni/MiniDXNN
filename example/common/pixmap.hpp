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
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <type_traits>
#include <vector>
// Half
#include "half.hpp"

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
  auto sample(const float u, const float v) const noexcept -> ValueT;

 private:
  auto init() noexcept -> void { m_data.resize(width() * height()); }


  std::vector<ValueT> m_data;
  std::uint32_t m_width;
  std::uint32_t m_height;
};

// ============================================================================
// Implementation
// ============================================================================

template <typename T, size_t CHANNEL>
inline
auto Pixmap<T, CHANNEL>::sample(const float u, const float v) const noexcept -> typename Pixmap<T, CHANNEL>::ValueT
{
  const size_t px = static_cast<size_t>(u * static_cast<float>(m_width - 1));
  const size_t py = static_cast<size_t>(v * static_cast<float>(m_height - 1));
  return get(py, px);
}

static_assert(sizeof(std::uint8_t) == 1);
using PixmapU8 = Pixmap<std::uint8_t>;
using PixmapRgb = Pixmap<std::uint8_t, 3>;

// ============================================================================
// HDR to LDR conversion (2-channel grayscale)
// ============================================================================

/*!
  \brief Convert floating-point MLP output to 8-bit grayscale.

  The MLP outputs 2 channels per pixel. This function extracts the first channel,
  clamps it to [0,1], and quantizes to [0,255] for image output.

  \param hdr  MLP output (2 values per pixel: [ch0, ch1, ch0, ch1, ...])
  \param ldr  Target pixmap for 8-bit grayscale output
*/
template <typename Type>
auto mapToLdr(const std::span<const Type> hdr, PixmapU8& ldr) noexcept -> void
{
  std::span out = ldr.data();
  const size_t numPixels = ldr.width() * ldr.height();
  for (size_t i = 0; i < numPixels; ++i) {
    using half_float::round;
    using std::round;
    using std::clamp;
    Type x = hdr[2 * i];
    x = clamp(x, static_cast<Type>(0), static_cast<Type>(1));
    x = round(x * static_cast<Type>(255));
    out[i] = {{static_cast<std::uint8_t>(x)}};
  }
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_PIXMAP_HPP */
