/*!
  \file matrix.hpp
  \author Sho Ikeda
  \brief Matrix and vector linear algebra utilities for CPU-side computation
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_MATRIX_HPP
#define MINIDXNN_EXAMPLE_MATRIX_HPP 1

// Standard C++ library
#include <bit>
#include <cassert>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
// Example
#include "utility.hpp"

namespace ex {

//! Row-major matrix reference with optional stride (non-owning view)
template <Arithmetic ArithT>
class MatrixRef
{
  using BaseT = MatrixRef<ArithT>;

 public:
  using Type = ArithT;
  using ConstT = std::add_const_t<Type>; 
  using ReferenceT = std::add_lvalue_reference_t<Type>;
  using ConstReferenceT = std::add_lvalue_reference_t<ConstT>;


  MatrixRef(const size_t rowSize, const size_t columnSize, std::span<Type> data) noexcept :
      m_rowSize{rowSize},
      m_columnSize{columnSize},
      m_stride{columnSize},
      m_data{data} {}

  MatrixRef(const size_t rowSize, const size_t columnSize, const size_t stride, std::span<Type> data) noexcept :
      m_rowSize{rowSize},
      m_columnSize{columnSize},
      m_stride{stride},
      m_data{data} {}


  MatrixRef(MatrixRef&& other) noexcept :
      m_rowSize{other.m_rowSize},
      m_columnSize{other.m_columnSize},
      m_stride{other.m_stride},
      m_data{other.m_data} {}

  virtual ~MatrixRef() = default;

  [[nodiscard]]
  auto operator()(const size_t row, const size_t column) noexcept -> ReferenceT {return get(row, column);}

  [[nodiscard]]
  auto operator()(const size_t row, const size_t column) const noexcept -> ConstReferenceT {return get(row, column);}


  [[nodiscard]]
  virtual auto columnSize() const noexcept -> size_t {return m_columnSize;}

  [[nodiscard]]
  auto data() noexcept -> std::span<Type> {return m_data;}

  [[nodiscard]]
  auto data() const noexcept -> std::span<ConstT> {return m_data;}

  [[nodiscard]]
  virtual auto get(const size_t row, const size_t column) noexcept -> ReferenceT
  {
    const size_t index = row * stride() + column;
    return m_data[index];
  }

  [[nodiscard]]
  virtual auto get(const size_t row, const size_t column) const noexcept -> ConstReferenceT
  {
    const size_t index = row * stride() + column;
    return m_data[index];
  }

  [[nodiscard]]
  virtual auto rowSize() const noexcept -> size_t {return m_rowSize;}

  [[nodiscard]]
  auto stride() const noexcept -> size_t {return m_stride;}

  [[nodiscard]]
  auto strideInBytes() const noexcept -> size_t {return m_stride * sizeof(Type);}

 private:
  size_t m_rowSize;
  size_t m_columnSize;
  size_t m_stride;
  std::span<ArithT> m_data;
};

//! Transposed view of a MatrixRef (swaps row/column access)
template <Arithmetic ArithT>
class TransposedMatrixRef : public MatrixRef<ArithT>
{
 public:
  using BaseT = MatrixRef<ArithT>;
  using Type = typename BaseT::Type;
  using ConstT = typename BaseT::ConstT; 
  using ReferenceT = typename BaseT::ReferenceT;
  using ConstReferenceT = typename BaseT::ConstReferenceT;


  TransposedMatrixRef(const size_t originalRowSize,
                      const size_t originalColumnSize,
                      std::span<Type> originalData) noexcept : BaseT(originalRowSize, originalColumnSize, originalData) {}

  TransposedMatrixRef(const size_t originalRowSize,
                      const size_t originalColumnSize,
                      const size_t originalStride,
                      std::span<Type> originalData) noexcept : BaseT(originalRowSize, originalColumnSize, originalStride, originalData) {}

  TransposedMatrixRef(TransposedMatrixRef&& other) noexcept : BaseT(std::move(other)) {}


  [[nodiscard]]
  virtual auto columnSize() const noexcept -> size_t override {return BaseT::rowSize();}

  [[nodiscard]]
  virtual auto get(const size_t row, const size_t column) noexcept -> ReferenceT override {return BaseT::get(column, row);}

  [[nodiscard]]
  virtual auto get(const size_t row, const size_t column) const noexcept -> ConstReferenceT override {return BaseT::get(column, row);}

  [[nodiscard]]
  virtual auto rowSize() const noexcept -> size_t override {return BaseT::columnSize();}
};

template <Arithmetic ArithT>
auto makeMatrix(const size_t rowSize, const size_t columnSize, std::span<ArithT> data) noexcept
    -> std::unique_ptr<MatrixRef<ArithT>>
{
  return std::make_unique<MatrixRef<ArithT>>(rowSize, columnSize, data);
}

template <Arithmetic ArithT>
auto makeTransposedMatrix(const size_t originalRowSize, const size_t originalColumnSize, std::span<ArithT> data) noexcept
    -> std::unique_ptr<MatrixRef<ArithT>>
{
  return std::make_unique<TransposedMatrixRef<ArithT>>(originalRowSize, originalColumnSize, data);
}

// A matrix and a vector multiplication
template <Arithmetic OutputT, Arithmetic MatrixElemT, Arithmetic VecElemT> inline
auto mul(const MatrixRef<MatrixElemT>& lhs, const std::span<const VecElemT> rhs) noexcept -> std::vector<OutputT>
{
  assert(lhs.columnSize() == rhs.size());

  std::vector<OutputT> result;
  result.resize(lhs.rowSize(), static_cast<OutputT>(0));
  for (const size_t r : std::views::iota(static_cast<size_t>(0), lhs.rowSize())) {
    for (const size_t c : std::views::iota(static_cast<size_t>(0), lhs.columnSize())) {
      result[r] += static_cast<OutputT>(lhs(r, c)) * static_cast<OutputT>(rhs[c]);
    }
  }
  return result;
}

// Addition of vectors
template <Arithmetic OutputT, Arithmetic LhsElemT, Arithmetic RhsElemT> inline
auto add(const std::span<const LhsElemT> lhs, const std::span<const RhsElemT> rhs) noexcept -> std::vector<OutputT>
{
  assert(lhs.size() == rhs.size());

  std::vector<OutputT> result;
  result.resize(lhs.size());
  for (const size_t index : std::views::iota(static_cast<size_t>(0), lhs.size())) {
    result[index] = static_cast<OutputT>(lhs[index]) + static_cast<OutputT>(rhs[index]);
  }
  return result;
}

//! Fused matrix-vector multiply and add: result = A * b + c
template <Arithmetic OutputT, Arithmetic MatrixElemT, Arithmetic Vec1ElemT, Arithmetic Vec2ElemT> inline
auto mulAdd(const MatrixRef<MatrixElemT>& a, const std::span<const Vec1ElemT> b, const std::span<const Vec2ElemT> c) noexcept -> std::vector<OutputT>
{
  std::vector result = mul<OutputT>(a, b);
  result = add<OutputT, OutputT>(result, c);
  return result;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_MATRIX_HPP */
