/*!
  \file xoshiro128plus.cpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_XOSHIRO_128_PLUS_HPP
#define MINIDXNN_EXAMPLE_XOSHIRO_128_PLUS_HPP 1

// Standard C++ library
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <ranges>

namespace ex {

class Xoshiro128Plus
{
 public:
  using result_type = std::uint32_t; // For STL
  using ResultT = result_type;


  Xoshiro128Plus(const ResultT seed) : m_state{{0, 0, 0, 0}}
  {
    initialize(seed);
  }


  auto operator()() noexcept -> ResultT
  {
    return generate();
  }


  auto generate() noexcept -> ResultT
  {
    const ResultT result = m_state[0] + m_state[3];
    const ResultT t = m_state[1] << 9;

    m_state[2] ^= m_state[0];
    m_state[3] ^= m_state[1];
    m_state[1] ^= m_state[2];
    m_state[0] ^= m_state[3];

    m_state[2] ^= t;
    m_state[3] = rotateLeft(m_state[3], 11);

    return result;
  }

  static constexpr auto min() noexcept -> ResultT
  {
    return 1u;
  }

  static constexpr auto max() noexcept -> ResultT
  {
    return std::numeric_limits<ResultT>::max() - 1;
  }

  static auto rotateLeft(const ResultT x, int k) noexcept -> ResultT
  {
    return (x << k) | (x >> (32 - k));
  }

  static auto mapToUniFloat(const ResultT u) noexcept -> float
  {
    constexpr size_t int_digits = 32;
    constexpr size_t float_digits = 23;
    constexpr size_t offset = int_digits - float_digits;
    //using FLimits = std::numeric_limits<float>;
    const float result = static_cast<float>(u >> offset) / static_cast<float>(1 << float_digits);
    assert((0.0f <= result) && (result < 1.0f));
    return result;
  }

  [[nodiscard]] auto draw() noexcept -> float
  {
    const ResultT x = generate();
    const float u = mapToUniFloat(x);
    return u;
  }

 private:
  void initialize(const ResultT seed)
  {
    ResultT z = seed;
    std::ranges::for_each(m_state, [&z](ResultT& state)
    {
      z = z + 0x9e3779b9;
      z = (z ^ (z >> 16)) * 0x85ebca6b;
      z = (z ^ (z >> 13)) * 0xc2b2ae35;
      z = z ^ (z >> 16);
      state = z;
    });
  }

  std::array<ResultT, 4> m_state;
};

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_XOSHIRO_128_PLUS_HPP */
