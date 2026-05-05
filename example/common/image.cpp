/*!
  \file image.cpp
  \author Sho Ikeda
  \brief Image file I/O implementations (PPM format)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#include "image.hpp"
// Standard C++ library
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string_view>
// stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
// Examples
#include "pixmap.hpp"
#include "texture.hpp"
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

template <typename T, size_t Channel>
auto writeAsPpm(const Pixmap<T, Channel>& pixmap, std::ostream& output) noexcept -> void
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
  const std::span<const typename Pixmap<T, Channel>::ValueT> pixelData = pixmap.data();
  write<T>(reinterpret_cast<const T*>(pixelData.data()), output, static_cast<std::streamsize>(pixelData.size_bytes()));
}

template
auto writeAsPpm<std::uint8_t, 1>(const Pixmap<std::uint8_t, 1>&, std::ostream&) noexcept -> void;

auto writeAsPng(const Pixmap<std::uint8_t, 1>& pixmap, const std::filesystem::path& filePath) -> bool
{
  const std::string pathStr = filePath.string();
  const auto pixels = pixmap.data();
  const int result = stbi_write_png(pathStr.c_str(),
      static_cast<int>(pixmap.width()), static_cast<int>(pixmap.height()),
      1, static_cast<const void*>(pixels.data()), static_cast<int>(pixmap.width()));
  return result != 0;
}

auto writeAsPng(const Pixmap<std::uint8_t, 3>& pixmap, const std::filesystem::path& filePath) -> bool
{
  const std::string pathStr = filePath.string();
  const auto pixels = pixmap.data();
  const int stride = static_cast<int>(pixmap.width()) * 3;
  const int result = stbi_write_png(pathStr.c_str(),
      static_cast<int>(pixmap.width()), static_cast<int>(pixmap.height()),
      3, static_cast<const void*>(pixels.data()), stride);
  return result != 0;
}

auto loadTextureFromPng(const std::filesystem::path& filePath) -> Texture3Ch
{
  int w = 0, h = 0, channels = 0;
  const std::string pathStr = filePath.string();
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> data{
      stbi_load(pathStr.c_str(), &w, &h, &channels, 0),
      &stbi_image_free};

  if (!data) {
    std::cerr << std::format("[Error] Failed to load image: {} ({})\n",
        pathStr, stbi_failure_reason());
    std::exit(1);
  }

  Texture3Ch texture{static_cast<size_t>(w), static_cast<size_t>(h)};

  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      const int pixelIndex = row * w + col;
      float r = 0.0f, g = 0.0f, b = 0.0f;
      if (channels == 1) {
        r = g = b = static_cast<float>(data.get()[pixelIndex]) / 255.0f;
      } else if (channels == 2) {
        r = g = b = static_cast<float>(data.get()[pixelIndex * 2]) / 255.0f;
      } else {
        r = static_cast<float>(data.get()[pixelIndex * channels + 0]) / 255.0f;
        g = static_cast<float>(data.get()[pixelIndex * channels + 1]) / 255.0f;
        b = static_cast<float>(data.get()[pixelIndex * channels + 2]) / 255.0f;
      }
      texture.set(static_cast<size_t>(row), static_cast<size_t>(col), {{r, g, b}});
    }
  }

  std::cout << std::format("Loaded image: {} ({}x{}, {} channels)\n",
      pathStr, w, h, channels);

  return texture;
}

} /* namespace ex */
