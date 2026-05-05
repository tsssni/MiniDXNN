/*!
  \file texture.hpp
  \author Sho Ikeda
  \brief Texture pattern generation for MLP training
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_TEXTURE_HPP
#define MINIDXNN_EXAMPLE_TEXTURE_HPP 1

// Standard C++ library
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string_view>
#include <utility>
#include <vector>
// Example
#include "pixmap.hpp"

namespace ex {

//! 2-channel float texture type
using Texture2Ch = Pixmap<float, 2>;
//! 3-channel float texture type (RGB)
using Texture3Ch = Pixmap<float, 3>;

enum class TexturePattern
{
  GRADIENT,
  CHECKERBOARD,
  STRIPES,
  CIRCLE,
  PERLIN,
};

[[nodiscard]]
auto createGradientTexture(const size_t width, const size_t height) -> Texture3Ch;

[[nodiscard]]
auto createCheckerboardTexture(const size_t width, const size_t height, const size_t squareSize = 300) -> Texture3Ch;

[[nodiscard]]
auto createStripesTexture(const size_t width, const size_t height, const size_t stripeWidth = 160) -> Texture3Ch;

[[nodiscard]]
auto createCircleTexture(const size_t width, const size_t height) -> Texture3Ch;

[[nodiscard]]
auto perlinNoise(const float x, const float y, const int seed = 0) -> float;

[[nodiscard]]
auto createPerlinTexture(const size_t width, const size_t height, const float scale = 0.05f) -> Texture3Ch;

[[nodiscard]]
auto createTexture(const TexturePattern pattern, const size_t width, const size_t height) -> Texture3Ch;

[[nodiscard]]
auto getTexturePatternFromString(const std::string_view name) noexcept -> TexturePattern;

[[nodiscard]]
auto getTexturePatternString(const TexturePattern pattern) noexcept -> std::string_view;

// ============================================================================
// Implementation
// ============================================================================

[[nodiscard]] inline
auto createGradientTexture(const size_t width, const size_t height) -> Texture3Ch
{
  Texture3Ch texture{width, height};
  const float centerX = static_cast<float>(width) / 2.0f;
  const float centerY = static_cast<float>(height) / 2.0f;
  const float maxDist = std::sqrt(centerX * centerX + centerY * centerY);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      const float dx = static_cast<float>(j) - centerX;
      const float dy = static_cast<float>(i) - centerY;
      const float dist = std::sqrt(dx * dx + dy * dy);
      const float value = 1.0f - std::min(dist / maxDist, 1.0f);
      texture.set(i, j, {{value, value, value}});
    }
  }
  return texture;
}

[[nodiscard]] inline
auto createCheckerboardTexture(const size_t width, const size_t height, const size_t squareSize) -> Texture3Ch
{
  Texture3Ch texture{width, height};
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      const size_t checkX = (j / squareSize) % 2;
      const size_t checkY = (i / squareSize) % 2;
      const float value = static_cast<float>((checkX + checkY) % 2);
      texture.set(i, j, {{value, value, value}});
    }
  }
  return texture;
}

[[nodiscard]] inline
auto createStripesTexture(const size_t width, const size_t height, const size_t stripeWidth) -> Texture3Ch
{
  Texture3Ch texture{width, height};
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      const float value = 0.5f + 0.5f * std::sin(
          2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(stripeWidth));
      texture.set(i, j, {{value, value, value}});
    }
  }
  return texture;
}

[[nodiscard]] inline
auto createCircleTexture(const size_t width, const size_t height) -> Texture3Ch
{
  Texture3Ch texture{width, height};
  const float centerX = static_cast<float>(width) / 2.0f;
  const float centerY = static_cast<float>(height) / 2.0f;
  const float maxDist = std::sqrt(centerX * centerX + centerY * centerY);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      const float dx = static_cast<float>(j) - centerX;
      const float dy = static_cast<float>(i) - centerY;
      const float dist = std::sqrt(dx * dx + dy * dy);
      const float normalizedDist = dist / maxDist;
      const float value = 0.5f + 0.5f * std::sin(5.0f * std::numbers::pi_v<float> * normalizedDist);
      texture.set(i, j, {{value, value, value}});
    }
  }
  return texture;
}

[[nodiscard]] inline
auto perlinNoise(const float x, const float y, const int seed) -> float
{
  const int n = static_cast<int>(x * 57.0f + y * 131.0f + static_cast<float>(seed) * 13.0f);
  const int n2 = (n << 13) ^ n;
  return (1.0f - static_cast<float>((n2 * (n2 * n2 * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
}

[[nodiscard]] inline
auto createPerlinTexture(const size_t width, const size_t height, const float scale) -> Texture3Ch
{
  Texture3Ch texture{width, height};
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      const float noise = perlinNoise(static_cast<float>(j) * scale, static_cast<float>(i) * scale, 12345);
      const float value = (noise + 1.0f) * 0.5f;
      texture.set(i, j, {{value, value, value}});
    }
  }
  return texture;
}

[[nodiscard]] inline
auto createTexture(const TexturePattern pattern, const size_t width, const size_t height) -> Texture3Ch
{
  const Texture3Ch texture =
    (pattern == TexturePattern::GRADIENT)     ? createGradientTexture(width, height) :
    (pattern == TexturePattern::CHECKERBOARD) ? createCheckerboardTexture(width, height) :
    (pattern == TexturePattern::STRIPES)      ? createStripesTexture(width, height) :
    (pattern == TexturePattern::CIRCLE)       ? createCircleTexture(width, height) :
    (pattern == TexturePattern::PERLIN)       ? createPerlinTexture(width, height)
                                              : createGradientTexture(width, height);
  return texture;
}

[[nodiscard]] inline
auto getTexturePatternFromString(const std::string_view name) noexcept -> TexturePattern
{
  using namespace std::string_view_literals;
  const TexturePattern pattern = (name == "gradient"sv)     ? TexturePattern::GRADIENT :
                                 (name == "checkerboard"sv) ? TexturePattern::CHECKERBOARD :
                                 (name == "stripes"sv)      ? TexturePattern::STRIPES :
                                 (name == "circle"sv)       ? TexturePattern::CIRCLE :
                                 (name == "perlin"sv)       ? TexturePattern::PERLIN
                                                            : TexturePattern::GRADIENT;
  return pattern;
}

[[nodiscard]] inline
auto getTexturePatternString(const TexturePattern pattern) noexcept -> std::string_view
{
  using namespace std::string_view_literals;
  const std::string_view name =
    (pattern == TexturePattern::GRADIENT)     ? "gradient"sv :
    (pattern == TexturePattern::CHECKERBOARD) ? "checkerboard"sv :
    (pattern == TexturePattern::STRIPES)      ? "stripes"sv :
    (pattern == TexturePattern::CIRCLE)       ? "circle"sv :
    (pattern == TexturePattern::PERLIN)       ? "perlin"sv
                                              : "gradient"sv;
  return name;
}

// ============================================================================
// UV coordinate generation
// ============================================================================

/*!
  \brief Generate normalized UV coordinates for every pixel in a texture.

  Creates a flat array of interleaved (u, v) pairs where u,v in [0,1], ordered
  row-by-row from top-left to bottom-right.

  \return Vector of size (width * height * 2) containing [u0, v0, u1, v1, ...].
*/
template <typename Type>
[[nodiscard]]
auto createUvData(const size_t width, const size_t height) -> std::vector<Type>
{
  const size_t numPixels = width * height;
  std::vector<Type> uvData;
  uvData.reserve(numPixels * 2);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      const float u = (width > 1) ? static_cast<float>(j) / static_cast<float>(width - 1) : 0.0f;
      const float v = (height > 1) ? static_cast<float>(i) / static_cast<float>(height - 1) : 0.0f;
      uvData.push_back(static_cast<Type>(u));
      uvData.push_back(static_cast<Type>(v));
    }
  }

  return uvData;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_TEXTURE_HPP */
