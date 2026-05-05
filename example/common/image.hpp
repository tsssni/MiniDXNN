/*!
  \file image.hpp
  \author Sho Ikeda
  \brief Image file I/O declarations (PPM format)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_IMAGE_HPP
#define MINIDXNN_EXAMPLE_IMAGE_HPP 1

// Standard C++ library
#include <filesystem>
#include <ostream>

namespace ex {

// Forward declaration
template <typename, size_t> class Pixmap;
using Texture3Ch = Pixmap<float, 3>;


template <typename T, size_t Channel>
auto writeAsPpm(const Pixmap<T, Channel>& pixmap, std::ostream& output) noexcept -> void;

auto writeAsPng(const Pixmap<std::uint8_t, 1>& pixmap, const std::filesystem::path& filePath) -> bool;
auto writeAsPng(const Pixmap<std::uint8_t, 3>& pixmap, const std::filesystem::path& filePath) -> bool;

[[nodiscard]]
auto loadTextureFromPng(const std::filesystem::path& filePath) -> Texture3Ch;

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_IMAGE_HPP */
