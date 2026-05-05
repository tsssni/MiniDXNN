/*!
  \file d3d12_format.hpp
  \author Sho Ikeda
  \brief D3D12 linear algebra format definitions, alignment constants, and matrix/vector conversion utilities
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This header consolidates all D3D12-compatible memory layout definitions used by
  the LinAlg Matrix feature (HLSL spec 0035). It provides:
    - MatrixLayout enum
    - Alignment constants (MATRIX_ALIGNMENT, MATRIX_VECTOR_STRIDE_ALIGNMENT, VECTOR_ALIGNMENT)
    - D3D12 data type and layout mapping utilities
    - D3D12MatrixInfo / D3D12VectorInfo structs for describing matrix/vector data
    - Functions to compute D3D12 buffer sizes, strides, and convert CPU data to D3D12 format
    - Pack functions to build contiguous D3D12-format buffers from lists of matrices/vectors

  References:
    - https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md
*/

#ifndef MINIDXNN_EXAMPLE_D3D12_FORMAT_HPP
#define MINIDXNN_EXAMPLE_D3D12_FORMAT_HPP 1

// Standard C++ library
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>
// Half
#include "half.hpp"
// Example
#include "utility.hpp"

namespace ex {

// ============================================================================
// Matrix layout enum
// ============================================================================

enum class MatrixLayout
{
  ROW_MAJOR = 0,
  COLUMN_MAJOR,
  MUL_OPTIMAL,
  OUTER_PRODUCT_OPTIMAL,
};

// ============================================================================
// Alignment constants
//
// The base address of matrix resource and matrix offset must be 128-byte aligned.
// Also note that the size of the underlying allocation is guaranteed to be a
// multiple of 16 bytes ensuring that the 16 bytes access of the last row/column
// of the matrix is valid memory.
// ============================================================================

static constexpr size_t MATRIX_ALIGNMENT = 128;
// The matrix stride is 16-byte aligned.
static constexpr size_t MATRIX_VECTOR_STRIDE_ALIGNMENT = 16;
// The base address of bias vector resource and bias vector offset must be 128-byte aligned.
static constexpr size_t VECTOR_ALIGNMENT = 128;

// ============================================================================
// D3D12 type and layout mapping
// ============================================================================

//! Map MatrixLayout to D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT value
[[nodiscard]]
inline constexpr auto toD3D12MatrixLayout(const MatrixLayout layout) -> uint32_t
{
  // D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT values
  constexpr uint32_t ROW_MAJOR = 0;
  constexpr uint32_t COLUMN_MAJOR = 1;
  constexpr uint32_t MUL_OPTIMAL = 2;
  constexpr uint32_t OUTER_PRODUCT_OPTIMAL = 3;
  switch (layout) {
    case MatrixLayout::ROW_MAJOR: return ROW_MAJOR;
    case MatrixLayout::COLUMN_MAJOR: return COLUMN_MAJOR;
    case MatrixLayout::MUL_OPTIMAL: return MUL_OPTIMAL;
    case MatrixLayout::OUTER_PRODUCT_OPTIMAL: return OUTER_PRODUCT_OPTIMAL;
    default: return ROW_MAJOR;
  }
}

//! Map element type to D3D12_LINEAR_ALGEBRA_DATATYPE value
template <Arithmetic Type>
[[nodiscard]]
constexpr auto toD3D12DataType() -> uint32_t
{
  // D3D12_LINEAR_ALGEBRA_DATATYPE values
  constexpr uint32_t FLOAT16 = 7;
  constexpr uint32_t FLOAT32 = 8;
  if constexpr (std::is_same_v<std::remove_cv_t<Type>, half_float::half>)
    return FLOAT16;
  else if constexpr (std::is_same_v<std::remove_cv_t<Type>, float>)
    return FLOAT32;
  else
    static_assert(!sizeof(Type*), "Unsupported type for D3D12 linear algebra datatype");
}

//! Map MatrixLayout to dx::linalg::MatrixLayoutEnum value (HLSL side)
//! The HLSL enum values differ from D3D12 values since linalg.h SM 6.10:
//!   RowMajor=0, ColMajor=1, MulOptimal=2, MulOptimalTranspose=3,
//!   OuterProductOptimal=4, OuterProductOptimalTranspose=5
[[nodiscard]]
inline constexpr auto toHlslMatrixLayout(const MatrixLayout layout) -> int
{
  // dx::linalg::MatrixLayout::MatrixLayoutEnum values (SM 6.10 linalg.h)
  constexpr int ROW_MAJOR = 0;
  constexpr int COL_MAJOR = 1;
  constexpr int MUL_OPTIMAL = 2;
  constexpr int OUTER_PRODUCT_OPTIMAL = 4;
  switch (layout) {
    case MatrixLayout::ROW_MAJOR: return ROW_MAJOR;
    case MatrixLayout::COLUMN_MAJOR: return COL_MAJOR;
    case MatrixLayout::MUL_OPTIMAL: return MUL_OPTIMAL;
    case MatrixLayout::OUTER_PRODUCT_OPTIMAL: return OUTER_PRODUCT_OPTIMAL;
    default: return ROW_MAJOR;
  }
}

//! Check if a layout requires GPU matrix conversion (not directly packable on CPU)
[[nodiscard]]
inline constexpr auto needsMatrixConversion(const MatrixLayout layout) -> bool
{
  return layout == MatrixLayout::MUL_OPTIMAL || layout == MatrixLayout::OUTER_PRODUCT_OPTIMAL;
}

// ============================================================================
// D3D12 matrix/vector info structs
// ============================================================================

//! Describes a single matrix for D3D12 format conversion.
//! Input fields should be set by the caller before calling getD3D12MatrixInfo.
//! Input+Output fields may be modified by getD3D12MatrixInfo or packAsD3D12MatrixBuffer.
//! Output fields (m_stride, m_dataSize, m_dataType) are populated by getD3D12MatrixInfo.
template <Arithmetic Type>
struct D3D12MatrixInfo
{
  // Input fields (set by caller, not modified)
  std::span<const Type> m_srcData{};          //!< Source matrix data (row major without padding)
  size_t m_rowSize = 0;                       //!< Number of rows
  size_t m_columnSize = 0;                    //!< Number of columns

  // Input+Output fields (caller sets initial values; updated by getD3D12MatrixInfo or packAsD3D12MatrixBuffer)
  size_t m_alignment = 0;                     //!< Matrix alignment in bytes (0 = use MATRIX_ALIGNMENT)
  size_t m_vectorStrideAlignment = 0;         //!< Vector stride alignment in bytes (0 = use MATRIX_VECTOR_STRIDE_ALIGNMENT)
  MatrixLayout m_layout = MatrixLayout::ROW_MAJOR; //!< Desired layout; updated to reflect the effective layout after packing

  // Output fields (set by getD3D12MatrixInfo)
  size_t m_stride = 0;                        //!< Row/column stride in bytes
  size_t m_dataSize = 0;                      //!< Total buffer size in bytes for this matrix
  uint32_t m_dataType = 0;                    //!< D3D12_LINEAR_ALGEBRA_DATATYPE value
};

//! Describes a single vector (e.g., bias) for D3D12 format conversion.
//! Input fields should be set by the caller before calling getD3D12VectorInfo.
//! Input+Output fields may be modified by getD3D12VectorInfo.
//! Output fields (m_dataSize, m_dataType) are populated by getD3D12VectorInfo.
template <Arithmetic Type>
struct D3D12VectorInfo
{
  // Input fields (set by caller, not modified)
  std::span<const Type> m_srcData{};          //!< Source vector data

  // Input+Output fields (caller sets initial values; updated by getD3D12VectorInfo)
  size_t m_alignment = 0;                     //!< Vector alignment in bytes (0 = use VECTOR_ALIGNMENT)

  // Output fields (set by getD3D12VectorInfo)
  size_t m_dataSize = 0;                      //!< Total buffer size in bytes for this vector
  uint32_t m_dataType = 0;                    //!< D3D12_LINEAR_ALGEBRA_DATATYPE value
};

// ============================================================================
// Function declarations
// ============================================================================

//! Compute D3D12 matrix buffer layout information.
//! Sets m_stride, m_dataSize, m_dataType on the info struct.
//! If m_alignment or m_vectorStrideAlignment are 0, defaults are used and set.
//! MUL_OPTIMAL and OUTER_PRODUCT_OPTIMAL are treated as ROW_MAJOR.
//! Returns the effective MatrixLayout used (ROW_MAJOR or COLUMN_MAJOR).
template <Arithmetic Type>
auto getD3D12MatrixInfo(D3D12MatrixInfo<Type>& info) -> MatrixLayout;

//! Convert a single matrix to D3D12 format. Caller must have called getD3D12MatrixInfo first.
//! Writes the converted matrix data to dest (which must be at least info.m_dataSize bytes).
template <Arithmetic Type>
auto convertToD3D12Matrix(const D3D12MatrixInfo<Type>& info, std::span<uint8_t> dest) -> void;

//! Convert a D3D12-format matrix (with stride padding) back to dense row-major data.
//! This is the inverse of convertToD3D12Matrix. The info struct must have been populated
//! by getD3D12MatrixInfo (m_stride, m_dataSize fields must be valid).
template <Arithmetic Type>
[[nodiscard]]
auto convertToRowMatrix(const D3D12MatrixInfo<Type>& info, std::span<const Type> src) -> std::vector<Type>;

//! Compute D3D12 vector buffer layout information.
//! Sets m_dataSize and m_dataType on the info struct.
//! If m_alignment is 0, VECTOR_ALIGNMENT is used and set.
template <Arithmetic Type>
auto getD3D12VectorInfo(D3D12VectorInfo<Type>& info) -> void;

//! Pack multiple matrices into a single contiguous D3D12-format buffer.
//! Calls getD3D12MatrixInfo on each element, then convertToD3D12Matrix.
template <Arithmetic Type>
[[nodiscard]]
auto packAsD3D12Matrix(std::span<D3D12MatrixInfo<Type>> infoList) -> std::vector<Type>;

//! Pack multiple vectors into a single contiguous D3D12-format buffer.
//! Calls getD3D12VectorInfo on each element, then copies source data with alignment.
template <Arithmetic Type>
[[nodiscard]]
auto packAsD3D12Vector(std::span<D3D12VectorInfo<Type>> infoList) -> std::vector<Type>;

// ============================================================================
// Implementation
// ============================================================================

namespace detail {

//! Validate alignment meets D3D12/LinAlg Matrix requirements.
//! Alignment must be a power of 2 and at least minAlignment bytes.
inline auto validateAlignment([[maybe_unused]] const size_t alignment,
                              [[maybe_unused]] const size_t minAlignment) -> void
{
  assert(std::has_single_bit(alignment) && "Alignment must be a power of 2");
  assert(alignment >= minAlignment && "Alignment is below minimum required");
}

} // namespace detail

template <Arithmetic Type>
auto getD3D12MatrixInfo(D3D12MatrixInfo<Type>& info) -> MatrixLayout
{
  // Apply defaults for zero alignment values
  if (info.m_alignment == 0)
    info.m_alignment = MATRIX_ALIGNMENT;
  if (info.m_vectorStrideAlignment == 0)
    info.m_vectorStrideAlignment = MATRIX_VECTOR_STRIDE_ALIGNMENT;

  // Validate alignment requirements:
  // Matrix alignment must be power of 2 and >= 128 bytes
  // Vector stride alignment must be power of 2 and >= 16 bytes
  detail::validateAlignment(info.m_alignment, 128);
  detail::validateAlignment(info.m_vectorStrideAlignment, 16);

  // Determine effective layout: MUL_OPTIMAL and OUTER_PRODUCT_OPTIMAL are treated as ROW_MAJOR
  MatrixLayout effectiveLayout = info.m_layout;
  if (needsMatrixConversion(effectiveLayout))
    effectiveLayout = MatrixLayout::ROW_MAJOR;

  const bool isColumnMajor = (effectiveLayout == MatrixLayout::COLUMN_MAJOR);
  // For COLUMN_MAJOR: minor dimension is rows, major dimension is columns
  // For ROW_MAJOR: minor dimension is columns, major dimension is rows
  const size_t minorSize = isColumnMajor ? info.m_rowSize : info.m_columnSize;
  const size_t majorSize = isColumnMajor ? info.m_columnSize : info.m_rowSize;
  const size_t vectorStride = alignN<Type>(minorSize, info.m_vectorStrideAlignment);

  info.m_stride = vectorStride * sizeof(Type);
  info.m_dataSize = alignN<Type>(vectorStride * majorSize, info.m_alignment) * sizeof(Type);
  info.m_dataType = toD3D12DataType<Type>();

  return effectiveLayout;
}

template <Arithmetic Type>
auto convertToD3D12Matrix(const D3D12MatrixInfo<Type>& info, std::span<uint8_t> dest) -> void
{
  assert(dest.size() >= info.m_dataSize);

  // Zero-initialize destination
  std::memset(dest.data(), 0, info.m_dataSize);

  const size_t vectorStrideInElements = info.m_stride / sizeof(Type);
  Type* destTyped = reinterpret_cast<Type*>(dest.data());

  MatrixLayout effectiveLayout = info.m_layout;
  if (needsMatrixConversion(effectiveLayout))
    effectiveLayout = MatrixLayout::ROW_MAJOR;

  const bool isColumnMajor = (effectiveLayout == MatrixLayout::COLUMN_MAJOR);
  const size_t rowSize = info.m_rowSize;
  const size_t columnSize = info.m_columnSize;

  if (isColumnMajor) {
    // Pack column-by-column: dest[col * vectorStride + row]
    for (size_t c = 0, o = 0; c < columnSize; ++c) {
      for (size_t r = 0; r < rowSize; ++r) {
        destTyped[o + r] = info.m_srcData[r * columnSize + c];
      }
      o += vectorStrideInElements;
    }
  } else {
    // Pack row-by-row: dest[row * vectorStride + col]
    for (size_t r = 0, o = 0; r < rowSize; ++r) {
      for (size_t c = 0; c < columnSize; ++c) {
        destTyped[o + c] = info.m_srcData[r * columnSize + c];
      }
      o += vectorStrideInElements;
    }
  }
}

template <Arithmetic Type>
auto convertToRowMatrix(const D3D12MatrixInfo<Type>& info, std::span<const Type> src) -> std::vector<Type>
{
  const size_t vectorStrideInElements = info.m_stride / sizeof(Type);
  const size_t rowSize = info.m_rowSize;
  const size_t columnSize = info.m_columnSize;

  MatrixLayout effectiveLayout = info.m_layout;
  if (needsMatrixConversion(effectiveLayout))
    effectiveLayout = MatrixLayout::ROW_MAJOR;

  const bool isColumnMajor = (effectiveLayout == MatrixLayout::COLUMN_MAJOR);

  std::vector<Type> result(rowSize * columnSize);

  if (isColumnMajor) {
    for (size_t c = 0, o = 0; c < columnSize; ++c) {
      for (size_t r = 0; r < rowSize; ++r)
        result[r * columnSize + c] = src[o + r];
      o += vectorStrideInElements;
    }
  } else {
    for (size_t r = 0, o = 0; r < rowSize; ++r) {
      for (size_t c = 0; c < columnSize; ++c)
        result[r * columnSize + c] = src[o + c];
      o += vectorStrideInElements;
    }
  }
  return result;
}

template <Arithmetic Type>
auto getD3D12VectorInfo(D3D12VectorInfo<Type>& info) -> void
{
  // Apply default for zero alignment
  if (info.m_alignment == 0)
    info.m_alignment = VECTOR_ALIGNMENT;

  // Validate alignment: must be power of 2 and >= 64 bytes
  detail::validateAlignment(info.m_alignment, 64);

  info.m_dataSize = alignN<Type>(info.m_srcData.size(), info.m_alignment) * sizeof(Type);
  info.m_dataType = toD3D12DataType<Type>();
}

template <Arithmetic Type>
[[nodiscard]]
auto packAsD3D12Matrix(std::span<D3D12MatrixInfo<Type>> infoList) -> std::vector<Type>
{
  // Compute total size
  size_t totalElements = 0;
  for (D3D12MatrixInfo<Type>& info : infoList) {
    getD3D12MatrixInfo(info);
    totalElements += info.m_dataSize / sizeof(Type);
  }

  std::vector<Type> memory(totalElements, static_cast<Type>(0));

  // Convert each matrix
  size_t byteOffset = 0;
  for (const D3D12MatrixInfo<Type>& info : infoList) {
    std::span<uint8_t> dest{reinterpret_cast<uint8_t*>(memory.data()) + byteOffset, info.m_dataSize};
    convertToD3D12Matrix(info, dest);
    byteOffset += info.m_dataSize;
  }

  return memory;
}

template <Arithmetic Type>
[[nodiscard]]
auto packAsD3D12Vector(std::span<D3D12VectorInfo<Type>> infoList) -> std::vector<Type>
{
  // Compute total size
  size_t totalElements = 0;
  for (D3D12VectorInfo<Type>& info : infoList) {
    getD3D12VectorInfo(info);
    totalElements += info.m_dataSize / sizeof(Type);
  }

  std::vector<Type> memory(totalElements, static_cast<Type>(0));

  // Copy each vector with alignment
  size_t offset = 0;
  for (const D3D12VectorInfo<Type>& info : infoList) {
    const size_t elemCount = info.m_dataSize / sizeof(Type);
    std::span<const Type> src = info.m_srcData;
    for (size_t i = 0; i < src.size(); ++i) {
      memory[offset + i] = src[i];
    }
    offset += elemCount;
  }

  return memory;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_D3D12_FORMAT_HPP */
