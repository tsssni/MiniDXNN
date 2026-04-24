/*!
  \file cpp_fallback.hpp
  \author Sho Ikeda
  \brief Shared C++ fallback infrastructure for running mlp.hlsl as C++
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This header consolidates common code used by examples and unit tests when
  running the mlp.hlsl MLP implementation via the C++ fallback path.

  Provides:
    - Fallback include chain (hlsl_compat.hpp + mlp.hlsl compiled as C++)
    - DxLinalgDataTypeOf type trait
    - PackedMlpBuffers for packing MLP layer data into flat byte buffers
    - Alignment helpers and weight offset calculation
    - Gradient unpacking/collection utilities

  Prerequisites:
    - "half.hpp" must be includable (half-float-dep linked)
    - minidxnn-core include directory must be on the include path

  Usage:
    #include "common/cpp_fallback.hpp"
    // Then include kernel-specific HLSL headers:
    #include "kernel/my_kernel.hlsl"
*/

#ifndef MINIDXNN_EXAMPLE_CPP_FALLBACK_HPP
#define MINIDXNN_EXAMPLE_CPP_FALLBACK_HPP 1

// Standard C++ library
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
// Half
#include "half.hpp"
// Example
#include "mlp_layer.hpp"
#include "utility.hpp"

// Set up HLSL C++ fallback shim and include mlp.hlsl
#ifndef MINIDXNN_CPP_FALLBACK_HALF_TYPE
#define MINIDXNN_CPP_FALLBACK_HALF_TYPE half_float::half
#endif
#include "minidxnn/cpp/hlsl_compat.hpp"
#include "minidxnn/hlsl/mlp.hlsl"

namespace ex {

// ============================================================================
// Compile-time flag: true when built with -DMINIDXNN_CPP_FALLBACK_ONLY=ON
// ============================================================================

constexpr bool isCppFallbackForced =
#ifdef MINIDXNN_CPP_FALLBACK_ONLY
  true;
#else
  false;
#endif

// ============================================================================
// DxLinalgDataTypeOf — map C++ type to dx::linalg::DataType at compile time
// ============================================================================

template <typename T>
struct DxLinalgDataTypeOf;

template <>
struct DxLinalgDataTypeOf<half_float::half> {
  [[maybe_unused]] static constexpr auto value = dx::linalg::DATA_TYPE_FLOAT16;
};

template <>
struct DxLinalgDataTypeOf<float> {
  [[maybe_unused]] static constexpr auto value = dx::linalg::DATA_TYPE_FLOAT32;
};

// ============================================================================
// Alignment helpers
// ============================================================================

[[nodiscard]]
constexpr auto alignBytes(const size_t v, const size_t a) noexcept -> size_t
{
  return ((v + a - 1) / a) * a;
}

// Compute the byte offset of a layer's weight matrix within a packed weight buffer
[[nodiscard]]
constexpr auto calcWeightOffset(const uint2& matrixSizes, const size_t layerIndex) noexcept -> size_t
{
  size_t offset = 0;
  if (layerIndex > 0) offset += alignBytes(matrixSizes[0], MATRIX_ALIGNMENT);
  if (layerIndex > 1) offset += alignBytes(matrixSizes[1], MATRIX_ALIGNMENT) * (layerIndex - 1);
  return offset;
}

// ============================================================================
// PackedMlpBuffers — pack MLP layer weights and biases into flat byte buffers
// matching the layout expected by mlp.hlsl (row-major matrices with alignment).
// ============================================================================

template <Arithmetic Type>
struct PackedMlpBuffers
{
  std::vector<std::uint8_t> weightBuf;
  std::vector<std::uint8_t> biasBuf;
  uint2 matrixSizes{};

  ByteAddressBuffer weightBAB() const { return ByteAddressBuffer{weightBuf}; }
  ByteAddressBuffer biasBAB()   const { return ByteAddressBuffer{biasBuf}; }

  void pack(const std::span<const MlpLayer<Type, Type>> mlpData, bool hasBias)
  {
    const size_t numLayers = mlpData.size();

    // Prepare weight/bias data and layer info for packing
    std::vector<std::span<const Type>> weightDataList;
    std::vector<std::span<const Type>> biasDataList;
    std::vector<std::tuple<size_t, size_t>> layerInfoList;
    for (size_t i = 0; i < numLayers; ++i) {
      weightDataList.push_back(mlpData[i].weightData());
      layerInfoList.emplace_back(mlpData[i].inputDimension(), mlpData[i].outputDimension());
      if (hasBias)
        biasDataList.push_back(mlpData[i].biasData());
      else
        biasDataList.emplace_back(std::span<const Type>{});
    }

    // Pack weights
    std::vector<size_t> matrixStrides(numLayers);
    auto packedWeights = packMatrixData<Type>(
        weightDataList, layerInfoList, MatrixLayout::ROW_MAJOR,
        matrixStrides, MATRIX_ALIGNMENT, MATRIX_VECTOR_STRIDE_ALIGNMENT);
    weightBuf.resize(packedWeights.size() * sizeof(Type));
    std::memcpy(weightBuf.data(), packedWeights.data(), weightBuf.size());

    matrixSizes[0] = static_cast<uint>(matrixStrides[0]);
    matrixSizes[1] = numLayers > 1 ? static_cast<uint>(matrixStrides[1]) : matrixSizes[0];

    // Pack biases
    if (hasBias) {
      auto packedBiases = packVectorData<Type>(biasDataList, VECTOR_ALIGNMENT);
      biasBuf.resize(packedBiases.size() * sizeof(Type));
      std::memcpy(biasBuf.data(), packedBiases.data(), biasBuf.size());
    } else {
      const size_t hiddenDim = (numLayers > 1) ? mlpData[0].outputDimension() : mlpData[0].inputDimension();
      const size_t biasStride = alignBytes(hiddenDim * sizeof(Type), VECTOR_ALIGNMENT);
      biasBuf.assign(biasStride * numLayers, 0);
    }
  }
};

// ============================================================================
// Gradient unpacking/collection utilities
// ============================================================================

// Unpack weight gradients from a flat byte buffer back to per-layer matrices
template <Arithmetic Type>
auto unpackWeightGrads(const std::vector<std::uint8_t>& buf,
                       const std::span<const MlpLayer<Type, Type>> mlpData,
                       const uint2& matrixSizes) -> std::vector<Type>
{
  std::vector<Type> grads;
  for (size_t i = 0; i < mlpData.size(); ++i) {
    const auto& layer = mlpData[i];
    const size_t rows = layer.outputDimension(), cols = layer.inputDimension();
    const size_t vs = alignBytes(cols * sizeof(Type), MATRIX_VECTOR_STRIDE_ALIGNMENT);
    const size_t wOff = calcWeightOffset(matrixSizes, i);
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < cols; ++c) {
        Type v{};
        std::memcpy(&v, buf.data() + wOff + r * vs + c * sizeof(Type), sizeof(Type));
        grads.push_back(v);
      }
  }
  return grads;
}

// Unpack bias gradients from a flat byte buffer
template <Arithmetic Type>
auto unpackBiasGrads(const std::vector<std::uint8_t>& buf,
                     const std::span<const MlpLayer<Type, Type>> mlpData,
                     size_t hiddenDim) -> std::vector<Type>
{
  std::vector<Type> grads;
  const size_t stride = alignBytes(hiddenDim * sizeof(Type), VECTOR_ALIGNMENT);
  for (size_t i = 0; i < mlpData.size(); ++i) {
    const size_t rows = mlpData[i].outputDimension();
    const size_t bOff = stride * i;
    for (size_t r = 0; r < rows; ++r) {
      Type v{};
      std::memcpy(&v, buf.data() + bOff + r * sizeof(Type), sizeof(Type));
      grads.push_back(v);
    }
  }
  return grads;
}

// Collect per-layer weight gradients from CPU reference MLP data
template <Arithmetic Type>
auto collectWeightGrads(const std::span<const MlpLayer<Type, Type>> mlpData) -> std::vector<Type>
{
  std::vector<Type> grads;
  for (const auto& layer : mlpData)
    for (const auto& g : layer.weightGrads())
      grads.push_back(g);
  return grads;
}

// Collect per-layer bias gradients from CPU reference MLP data
template <Arithmetic Type>
auto collectBiasGrads(const std::span<const MlpLayer<Type, Type>> mlpData) -> std::vector<Type>
{
  std::vector<Type> grads;
  for (const auto& layer : mlpData)
    for (const auto& g : layer.biasGrads())
      grads.push_back(g);
  return grads;
}

// ============================================================================
// Pack a single weight matrix into a byte buffer with stride-aligned rows,
// matching the layout expected by mlp.hlsl MatrixRefImpl.
// ============================================================================

template <Arithmetic Type>
auto packSingleMatrix(const std::span<const Type> data,
                      const size_t rowSize,
                      const size_t columnSize,
                      const bool isTransposed) -> std::pair<std::vector<std::uint8_t>, size_t /*vectorStride*/>
{
  const size_t physRows = isTransposed ? columnSize : rowSize;
  const size_t physCols = isTransposed ? rowSize : columnSize;
  const size_t vectorStride = alignBytes(physCols * sizeof(Type), MATRIX_VECTOR_STRIDE_ALIGNMENT);
  const size_t totalBytes = alignBytes(physRows * vectorStride, MATRIX_ALIGNMENT);
  std::vector<std::uint8_t> buf(totalBytes, 0);

  for (size_t r = 0; r < physRows; ++r)
    for (size_t c = 0; c < physCols; ++c) {
      const Type v = data[r * physCols + c];
      std::memcpy(buf.data() + r * vectorStride + c * sizeof(Type), &v, sizeof(Type));
    }

  return {std::move(buf), vectorStride};
}

// Pack a single bias vector into a byte buffer with alignment
template <Arithmetic Type>
auto packSingleBias(const std::span<const Type> data,
                    const size_t rowSize) -> std::vector<std::uint8_t>
{
  const size_t totalBytes = alignBytes(rowSize * sizeof(Type), VECTOR_ALIGNMENT);
  std::vector<std::uint8_t> buf(totalBytes, 0);
  for (size_t i = 0; i < data.size(); ++i)
    std::memcpy(buf.data() + i * sizeof(Type), &data[i], sizeof(Type));
  return buf;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_CPP_FALLBACK_HPP */
