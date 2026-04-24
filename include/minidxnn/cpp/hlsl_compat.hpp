/*!
  \file hlsl_compat.hpp
  \author Sho Ikeda
  \brief C++ compatibility shim for compiling mlp.hlsl as C++
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This header provides C++ implementations of HLSL intrinsics, types, and
  dx::linalg constructs so that include/minidxnn/hlsl/mlp.hlsl can be compiled as
  a regular C++ header on Ubuntu (clang / gcc) without DirectX dependencies.

  When mlp.hlsl is included in a C++ translation unit, this header must be
  included first. The macros MINIDXNN_NO_INCLUDE_DX_LINALG and
  MINIDXNN_USE_SOFTWARE_LINALG_IMPL must be defined.

  Prerequisites:
    - MINIDXNN_CPP_FALLBACK_HALF_TYPE must be defined (e.g. half_float::half)
*/

#ifndef MINIDXNN_CPP_HLSL_COMPAT_HPP
#define MINIDXNN_CPP_HLSL_COMPAT_HPP 1

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

// Verify that the half type macro is defined
#ifndef MINIDXNN_CPP_FALLBACK_HALF_TYPE
  #error "MINIDXNN_CPP_FALLBACK_HALF_TYPE must be defined before including hlsl_compat.hpp"
#endif

#ifndef MINIDXNN_NO_INCLUDE_DX_LINALG
#define MINIDXNN_NO_INCLUDE_DX_LINALG 1
#endif

#ifndef MINIDXNN_USE_SOFTWARE_LINALG_IMPL
#define MINIDXNN_USE_SOFTWARE_LINALG_IMPL 1
#endif

// ============================================================================
// HLSL scalar type aliases
// ============================================================================

using uint = std::uint32_t;
using float16_t = MINIDXNN_CPP_FALLBACK_HALF_TYPE;
using float32_t = float;
using int16_t  = std::int16_t;
using uint16_t = std::uint16_t;
using int32_t  = std::int32_t;
using uint32_t = std::uint32_t;
using half     = float16_t;

// Packed types (treated as int32 in C++ — used only in type-mapping tables)
struct int8_t4_packed  { std::int32_t  v; };
struct uint8_t4_packed { std::uint32_t v; };

// ============================================================================
// HLSL vector type — vector<T, N>
// ============================================================================

namespace minidxnn_detail {
template <typename T, int N>
struct VectorAlignment {
  static constexpr std::size_t value = (N * sizeof(T) > 16) ? 32 : 16;
};

// DataSize: actual data bytes in a type (excludes alignment padding for vector)
template <typename T>
struct DataSize {
  static constexpr std::size_t value = sizeof(T);
};
} // namespace minidxnn_detail

template <typename T, int N>
struct alignas(minidxnn_detail::VectorAlignment<T, N>::value) vector
{
  std::array<T, N> data{};

  vector() { data.fill(T{}); }

  vector(T val) { data.fill(val); }

  template <typename U>
    requires (std::is_arithmetic_v<U> && !std::is_same_v<U, T>)
  vector(U val) { data.fill(static_cast<T>(val)); }

  template <int M = N> requires (M >= 2)
  vector(T a, T b)
  {
    data.fill(T{});
    data[0] = a;
    data[1] = b;
  }

  template <typename U, int M>
  explicit vector(const vector<U, M>& other)
  {
    constexpr size_t minN = static_cast<size_t>((N < M) ? N : M);
    for (size_t i = 0; i < minN; ++i)
      data[i] = static_cast<T>(other[i]);
    for (size_t i = minN; i < static_cast<size_t>(N); ++i)
      data[i] = T{};
  }

  T& operator[](size_t i) { return data[i]; }
  const T& operator[](size_t i) const { return data[i]; }

  // Arithmetic: vector op vector
  vector operator+(const vector& rhs) const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = data[i] + rhs[i];
    return r;
  }
  vector operator-(const vector& rhs) const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = data[i] - rhs[i];
    return r;
  }
  vector operator*(const vector& rhs) const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = data[i] * rhs[i];
    return r;
  }
  vector operator/(const vector& rhs) const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = data[i] / rhs[i];
    return r;
  }

  // Arithmetic: vector op scalar
  vector operator+(T s) const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = data[i] + s;
    return r;
  }
  vector operator-(T s) const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = data[i] - s;
    return r;
  }
  vector operator*(T s) const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = data[i] * s;
    return r;
  }
  vector operator/(T s) const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = data[i] / s;
    return r;
  }

  vector operator-() const
  {
    vector r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = -data[i];
    return r;
  }

  // Comparison
  vector<bool, N> operator<(const vector& rhs) const
  {
    vector<bool, N> r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = static_cast<float>(data[i]) < static_cast<float>(rhs[i]);
    return r;
  }

  vector<bool, N> operator<(T s) const
  {
    vector<bool, N> r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = static_cast<float>(data[i]) < static_cast<float>(s);
    return r;
  }

  vector<bool, N> operator>(const vector& rhs) const
  {
    vector<bool, N> r;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
      r[i] = static_cast<float>(data[i]) > static_cast<float>(rhs[i]);
    return r;
  }
};

namespace minidxnn_detail {
template <typename T, int N>
struct DataSize<vector<T, N>> {
  static constexpr std::size_t value = static_cast<std::size_t>(N) * sizeof(T);
};
} // namespace minidxnn_detail

// scalar op vector
template <typename T, int N> vector<T, N> operator*(T s, const vector<T, N>& v) { return v * s; }
template <typename T, int N> vector<T, N> operator+(T s, const vector<T, N>& v) { return v + s; }

template <typename T, int N>
vector<T, N> operator/(T s, const vector<T, N>& v)
{
  vector<T, N> r;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    r[i] = s / v[i];
  return r;
}

template <typename T, int N>
vector<T, N> operator-(T s, const vector<T, N>& v)
{
  vector<T, N> r;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    r[i] = s - v[i];
  return r;
}

template <typename T, int N>
vector<bool, N> operator>(T s, const vector<T, N>& v)
{
  vector<bool, N> r;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    r[i] = static_cast<float>(s) > static_cast<float>(v[i]);
  return r;
}

// ============================================================================
// uint2 / uint32_t2 / int2
// ============================================================================

using uint2 = vector<uint32_t, 2>;
using uint32_t2 = vector<uint32_t, 2>;
using int2 = vector<int32_t, 2>;

// ============================================================================
// HLSL intrinsic functions
// ============================================================================

template <typename T, int N>
T dot(const vector<T, N>& a, const vector<T, N>& b)
{
  T sum = T{};
  for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    sum = sum + a[i] * b[i];
  return sum;
}

template <typename T, int N>
vector<T, N> abs(const vector<T, N>& v)
{
  vector<T, N> r;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
    using std::abs;
    if constexpr (std::is_same_v<T, float16_t>) {
      r[i] = static_cast<T>(abs(static_cast<float>(v[i])));
    } else {
      r[i] = abs(v[i]);
    }
  }
  return r;
}

template <typename T, int N>
vector<T, N> exp(const vector<T, N>& v)
{
  vector<T, N> r;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
    using std::exp;
    if constexpr (std::is_same_v<T, float16_t>) {
      r[i] = static_cast<T>(exp(static_cast<float>(v[i])));
    } else {
      r[i] = exp(v[i]);
    }
  }
  return r;
}

template <typename T, int N>
vector<T, N> max(const vector<T, N>& a, const vector<T, N>& b)
{
  vector<T, N> r;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    r[i] = (static_cast<float>(a[i]) > static_cast<float>(b[i])) ? a[i] : b[i];
  return r;
}

template <typename T, int N>
vector<T, N> select(const vector<bool, N>& cond,
                    const vector<T, N>& trueVal,
                    const vector<T, N>& falseVal)
{
  vector<T, N> r;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    r[i] = cond[i] ? trueVal[i] : falseVal[i];
  return r;
}

template <typename T, int N>
vector<T, N> select(T cond,
                    const vector<T, N>& trueVal,
                    const vector<T, N>& falseVal)
{
  return (static_cast<float>(cond) != 0.0f) ? trueVal : falseVal;
}

// ============================================================================
// HLSL float16/float32 bit manipulation — using std::bit_cast
// ============================================================================

inline float16_t asfloat16(uint16_t bits)
{
  return std::bit_cast<float16_t>(bits);
}

inline uint16_t asuint16(float16_t value)
{
  return std::bit_cast<uint16_t>(value);
}

inline float asfloat(uint32_t bits)
{
  return std::bit_cast<float>(bits);
}

inline uint32_t asuint(float value)
{
  return std::bit_cast<uint32_t>(value);
}

inline uint32_t f32tof16(float value)
{
  return static_cast<uint32_t>(asuint16(static_cast<float16_t>(value)));
}

inline float f16tof32(uint32_t bits)
{
  return static_cast<float>(asfloat16(static_cast<uint16_t>(bits)));
}

// ============================================================================
// ByteAddressBuffer / RWByteAddressBuffer — CPU-side buffer emulation
//
// Uses std::span<const std::byte> / std::span<std::byte> as sole storage.
// Supports direct construction from std::vector and std::span:
//   std::vector<std::uint8_t> data(1024);
//   ByteAddressBuffer buf = data;            // implicit conversion
//   RWByteAddressBuffer rwBuf = data;        // implicit conversion
// ============================================================================

struct ByteAddressBuffer
{
  std::span<const std::byte> m_storage{};

  ByteAddressBuffer() = default;

  ByteAddressBuffer(const std::uint8_t* ptr, std::size_t size)
    : m_storage{reinterpret_cast<const std::byte*>(ptr), size} {}

  // Construct from contiguous byte containers
  ByteAddressBuffer(const std::vector<std::uint8_t>& v)
    : m_storage{reinterpret_cast<const std::byte*>(v.data()), v.size()} {}
  ByteAddressBuffer(std::span<const std::uint8_t> s)
    : m_storage{reinterpret_cast<const std::byte*>(s.data()), s.size()} {}

  // Construct from typed containers (reinterprets as bytes)
  template <typename T>
    requires (std::is_trivially_copyable_v<T> && !std::is_same_v<std::remove_cv_t<T>, std::uint8_t>)
  ByteAddressBuffer(const std::vector<T>& v)
    : m_storage{reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(T)} {}

  template <typename T>
    requires (std::is_trivially_copyable_v<T> && !std::is_same_v<std::remove_cv_t<T>, std::uint8_t>)
  ByteAddressBuffer(std::span<const T> s)
    : m_storage{reinterpret_cast<const std::byte*>(s.data()), s.size_bytes()} {}

  explicit operator bool() const { return !m_storage.empty(); }

  [[nodiscard]] const std::byte* data() const { return m_storage.data(); }
  [[nodiscard]] std::size_t size() const { return m_storage.size(); }

  template <typename T>
  T Load(uint offset) const
  {
    T value{};
    std::memcpy(&value, m_storage.data() + offset, minidxnn_detail::DataSize<T>::value);
    return value;
  }
};

struct RWByteAddressBuffer
{
  std::span<std::byte> m_storage{};

  RWByteAddressBuffer() = default;

  RWByteAddressBuffer(std::uint8_t* ptr, std::size_t size)
    : m_storage{reinterpret_cast<std::byte*>(ptr), size} {}

  // Construct from contiguous byte containers
  RWByteAddressBuffer(std::vector<std::uint8_t>& v)
    : m_storage{reinterpret_cast<std::byte*>(v.data()), v.size()} {}
  RWByteAddressBuffer(std::span<std::uint8_t> s)
    : m_storage{reinterpret_cast<std::byte*>(s.data()), s.size()} {}

  // Construct from typed containers (reinterprets as bytes)
  template <typename T>
    requires (std::is_trivially_copyable_v<T> && !std::is_same_v<std::remove_cv_t<T>, std::uint8_t>)
  RWByteAddressBuffer(std::vector<T>& v)
    : m_storage{reinterpret_cast<std::byte*>(v.data()), v.size() * sizeof(T)} {}

  template <typename T>
    requires (std::is_trivially_copyable_v<T> && !std::is_same_v<std::remove_cv_t<T>, std::uint8_t>)
  RWByteAddressBuffer(std::span<T> s)
    : m_storage{reinterpret_cast<std::byte*>(s.data()), s.size_bytes()} {}

  explicit operator bool() const { return !m_storage.empty(); }

  [[nodiscard]] std::byte* data() { return m_storage.data(); }
  [[nodiscard]] const std::byte* data() const { return m_storage.data(); }
  [[nodiscard]] std::size_t size() const { return m_storage.size(); }

  template <typename T>
  T Load(uint offset) const
  {
    T value{};
    std::memcpy(&value, m_storage.data() + offset, minidxnn_detail::DataSize<T>::value);
    return value;
  }

  template <typename T>
  void Store(uint offset, const T& value)
  {
    std::memcpy(m_storage.data() + offset, &value, minidxnn_detail::DataSize<T>::value);
  }

  // Thread-safe compare-exchange using std::atomic_ref
  void InterlockedCompareExchange(uint offset,
                                  uint32_t compare,
                                  uint32_t newValue,
                                  uint32_t& original)
  {
    auto* addr = reinterpret_cast<uint32_t*>(m_storage.data() + offset);
    std::atomic_ref<uint32_t> atom{*addr};
    original = compare;
    atom.compare_exchange_strong(original, newValue, std::memory_order_relaxed);
  }
};

// ============================================================================
// dx::linalg namespace — types and stubs for C++ fallback
// ============================================================================

namespace dx {
namespace linalg {

enum DataType
{
  DATA_TYPE_SINT16 = 0,
  DATA_TYPE_UINT16,
  DATA_TYPE_SINT32,
  DATA_TYPE_UINT32,
  DATA_TYPE_FLOAT16,
  DATA_TYPE_FLOAT32,
  DATA_TYPE_SINT8_T4_PACKED,
  DATA_TYPE_UINT8_T4_PACKED,
};

enum MatrixLayout
{
  MATRIX_LAYOUT_ROW_MAJOR = 0,
  MATRIX_LAYOUT_COLUMN_MAJOR,
  MATRIX_LAYOUT_MUL_OPTIMAL,
  MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL,
};

template <typename BufferT, DataType ELEM_TYPE, uint ROW_SIZE, uint COLUMN_SIZE,
          MatrixLayout LAYOUT, bool IS_TRANSPOSED>
struct MatrixRefImpl
{
  BufferT Buffer;
  uint StartOffset = 0;
  uint Stride = 0;
};

template <typename BufferT, DataType ELEM_TYPE>
struct VectorRefImpl
{
  BufferT Buffer;
  uint StartOffset = 0;
};

template <DataType ELEM_TYPE, uint ROW_SIZE, uint COLUMN_SIZE,
          MatrixLayout LAYOUT, bool IS_TRANSPOSED = false>
using RWMatrixRef = MatrixRefImpl<RWByteAddressBuffer, ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, LAYOUT, IS_TRANSPOSED>;

template <DataType ELEM_TYPE>
using RWVectorRef = VectorRefImpl<RWByteAddressBuffer, ELEM_TYPE>;

template <typename ElemT, int N, DataType ELEM_TYPE>
struct InterpretedVector
{
  vector<ElemT, N> Data;
};

template <DataType ELEM_TYPE, typename ElemT, int N>
InterpretedVector<ElemT, N, ELEM_TYPE>
MakeInterpretedVector(const vector<ElemT, N>& v)
{
  return InterpretedVector<ElemT, N, ELEM_TYPE>{v};
}

// Mul / MulAdd stubs — not used in software fallback path
template <typename OutputElemT, typename BufferT, DataType MATRIX_ELEM_TYPE,
          uint ROW_SIZE, uint COLUMN_SIZE, MatrixLayout LAYOUT, bool IS_TRANSPOSED,
          typename InputElemT, int INPUT_COUNT, DataType INPUT_ELEM_TYPE>
vector<OutputElemT, ROW_SIZE>
Mul(const MatrixRefImpl<BufferT, MATRIX_ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, LAYOUT, IS_TRANSPOSED>&,
    const InterpretedVector<InputElemT, INPUT_COUNT, INPUT_ELEM_TYPE>&)
{
  assert(false && "dx::linalg::Mul should not be called in CPP fallback mode");
  return vector<OutputElemT, ROW_SIZE>{};
}

template <typename OutputElemT, typename BufferT, DataType MATRIX_ELEM_TYPE,
          uint ROW_SIZE, uint COLUMN_SIZE, MatrixLayout LAYOUT, bool IS_TRANSPOSED,
          typename InputElemT, int INPUT_COUNT, DataType INPUT_ELEM_TYPE,
          typename BiasBufferT, DataType BIAS_ELEM_TYPE>
vector<OutputElemT, ROW_SIZE>
MulAdd(const MatrixRefImpl<BufferT, MATRIX_ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, LAYOUT, IS_TRANSPOSED>&,
       const InterpretedVector<InputElemT, INPUT_COUNT, INPUT_ELEM_TYPE>&,
       const VectorRefImpl<BiasBufferT, BIAS_ELEM_TYPE>&)
{
  assert(false && "dx::linalg::MulAdd should not be called in CPP fallback mode");
  return vector<OutputElemT, ROW_SIZE>{};
}

} // namespace linalg
} // namespace dx

#endif /* MINIDXNN_CPP_HLSL_COMPAT_HPP */
