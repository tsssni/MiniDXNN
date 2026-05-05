/*!
  \file gfx_utility.hpp
  \author Sho Ikeda
  \brief GFX utility functions for GPU context, buffer, program, and kernel management
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_GFX_UTILITY_HPP
#define MINIDXNN_EXAMPLE_GFX_UTILITY_HPP 1

// Standard C++ library
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <source_location>
#include <span>
#include <string_view>
#include <vector>
// GFX
#include "gfx.h"
// Example
#include "d3d12_format.hpp"
#include "utility.hpp"

namespace ex {

template <typename Type>
struct BindingData
{
  Type m_value;
  std::string_view m_name;
};

using BufferBindingDataT = BindingData<GfxBuffer>;
using IntBindingDataT = BindingData<std::int32_t>;
using FloatBindingDataT = BindingData<float>;

[[nodiscard]]
inline
auto bind(GfxBuffer buffer, const std::string_view name) -> BufferBindingDataT
{
  return BufferBindingDataT{buffer, name};
}

[[nodiscard]]
inline
auto bind(const std::int32_t value, const std::string_view name) -> IntBindingDataT
{
  return IntBindingDataT{value, name};
}

[[nodiscard]]
inline
auto bind(const float value, const std::string_view name) -> FloatBindingDataT
{
  return FloatBindingDataT{value, name};
}

[[nodiscard]]
auto createGfxContext(const bool enableDebugShader) -> std::shared_ptr<GfxContext>;

template <typename Type>
[[nodiscard]]
auto createGfxBuffer(GfxContext context, const size_t size, const GfxCpuAccess cpuAccess = kGfxCpuAccess_None) -> std::shared_ptr<GfxBuffer>;

template <typename Type>
[[nodiscard]]
auto createGfxBuffer(GfxContext context, std::span<const Type> data, const GfxCpuAccess cpuAccess = kGfxCpuAccess_None) -> std::shared_ptr<GfxBuffer>;

[[nodiscard]]
auto createGfxProgram(GfxContext context, const std::string_view fileName, const std::filesystem::path& fileDir = ".", const std::span<const OptionString> includePathList = {}) -> std::shared_ptr<GfxProgram>;

[[nodiscard]]
auto createGfxComputeKernel(GfxContext context, GfxProgram program, const std::string_view entryPoint = "", const std::span<const OptionString> definitionList = {}) -> std::shared_ptr<GfxKernel>;

inline
auto copyBuffer(GfxContext context, GfxBuffer source, GfxBuffer dest) -> void
{
  gfxCommandCopyBuffer(context, dest, source);
  gfxFinish(context);
}

template <typename Type>
[[nodiscard]]
inline
auto mapToCpu(GfxContext context, GfxBuffer buffer) -> std::span<Type>
{
  const size_t size = buffer.getSize() / sizeof(Type);
  return std::span<Type>{gfxBufferGetData<Type>(context, buffer), size};
}

auto runKernel(GfxContext context, GfxProgram program, GfxKernel kernel, const size_t threadGroupSize = 1, std::initializer_list<BufferBindingDataT> bufferList = {}, std::initializer_list<IntBindingDataT> intList = {}, OptionalRef<float> execTimeInMs = std::nullopt, std::initializer_list<FloatBindingDataT> floatList = {}) -> void;

auto runKernel(GfxContext context, GfxProgram program, GfxKernel kernel, const size_t threadGroupSize, std::span<const BufferBindingDataT> bufferList, std::initializer_list<IntBindingDataT> intList = {}, OptionalRef<float> execTimeInMs = std::nullopt, std::initializer_list<FloatBindingDataT> floatList = {}) -> void;

class GfxAssertTrue
{
 public:
  GfxAssertTrue(const std::source_location& location = std::source_location::current()) : m_location{location} {}

  template <typename ...Args>
  auto operator()(const GfxResult result, std::format_string<Args...> messageFormat, Args&&... args) const -> void
  {
    if (result != kGfxResult_NoError) {
      std::cerr << std::format(messageFormat, std::forward<Args>(args)...)
                << std::format(" in file: {} [line: {}, function={}]\n", m_location.file_name(), m_location.line(), m_location.function_name());
    }
  }

 private:
  std::source_location m_location;
};

template <typename Type> inline
auto createGfxBuffer(GfxContext context, const size_t size, const GfxCpuAccess cpuAccess) -> std::shared_ptr<GfxBuffer>
{
  const size_t sizeInBytes = size * sizeof(Type);
  GfxBuffer buffer = gfxCreateBuffer(context, sizeInBytes, nullptr, cpuAccess);
  std::shared_ptr<GfxBuffer> sharedBuffer(new GfxBuffer{buffer}, [context](GfxBuffer* ptr)
  {
    if (ptr != nullptr) {
      if (*ptr) {
        GfxAssertTrue{}(gfxDestroyBuffer(context, *ptr), "Destroying the buffer failed.");
      }
      delete ptr;
    }
  });
  return sharedBuffer;
}

template <typename Type> inline
auto createGfxBuffer(GfxContext context, std::span<const Type> data, const GfxCpuAccess cpuAccess) -> std::shared_ptr<GfxBuffer>
{
  const size_t sizeInBytes = data.size() * sizeof(Type);
  GfxBuffer buffer = gfxCreateBuffer(context, sizeInBytes, static_cast<void const*>(data.data()), cpuAccess);
  std::shared_ptr<GfxBuffer> sharedBuffer(new GfxBuffer{buffer}, [context](GfxBuffer* ptr)
  {
    if (ptr != nullptr) {
      if (*ptr) {
        GfxAssertTrue{}(gfxDestroyBuffer(context, *ptr), "Destroying the buffer failed.");
      }
      delete ptr;
    }
  });
  return sharedBuffer;
}

// ============================================================================
// D3D12 format buffer creation
// ============================================================================

//! Pack matrices into a GPU buffer.
//! For MUL_OPTIMAL/OUTER_PRODUCT_OPTIMAL layouts, matrices are first packed as ROW_MAJOR
//! then GPU-converted to the optimal layout via gfxConvertMatrix.
//! For ROW_MAJOR/COLUMN_MAJOR, matrices are packed directly.
//! Each entry's m_layout is updated to reflect the effective layout on the GPU buffer.
//! If allowRowMajorFallback is true (default: false), conversion failure silently falls back to ROW_MAJOR.
//! If false, conversion failure returns an empty (null) buffer.
template <Arithmetic Type> inline
auto packAsD3D12MatrixBuffer(GfxContext context,
                             std::span<D3D12MatrixInfo<Type>> infoList,
                             const bool allowRowMajorFallback = false) -> std::shared_ptr<GfxBuffer>
{
  if (infoList.empty())
    return {};

  const MatrixLayout requestedLayout = infoList[0].m_layout;

  if (needsMatrixConversion(requestedLayout)) {
    // Pack as ROW_MAJOR first for GPU conversion
    for (D3D12MatrixInfo<Type>& info : infoList)
      info.m_layout = MatrixLayout::ROW_MAJOR;
    const std::vector bufferData = packAsD3D12Matrix<Type>(infoList);
    std::shared_ptr buffer = createGfxBuffer<Type>(context, bufferData);

    constexpr uint32_t dataType = toD3D12DataType<Type>();
    const uint32_t d3dLayout = toD3D12MatrixLayout(requestedLayout);

    // Query destination sizes for all matrices
    std::vector<uint32_t> destSizes(infoList.size());
    size_t totalDestSize = 0;
    for (size_t i = 0; i < infoList.size(); ++i) {
      const uint32_t rowSize = static_cast<uint32_t>(infoList[i].m_rowSize);
      const uint32_t columnSize = static_cast<uint32_t>(infoList[i].m_columnSize);
      destSizes[i] = static_cast<uint32_t>(gfxGetMatrixMemorySize(context, rowSize, columnSize, d3dLayout, dataType, 0));
      totalDestSize += static_cast<size_t>(destSizes[i]);
    }

    if (totalDestSize > 0) {
      std::shared_ptr destBuffer = createGfxBuffer<uint8_t>(context, totalDestSize);
      gfxFinish(context);

      // Convert each matrix on the GPU
      bool conversionOk = true;
      uint64_t srcOffset = 0;
      uint64_t dstOffset = 0;
      for (size_t i = 0; i < infoList.size(); ++i) {
        const uint32_t rowSize = static_cast<uint32_t>(infoList[i].m_rowSize);
        const uint32_t columnSize = static_cast<uint32_t>(infoList[i].m_columnSize);
        const size_t srcSizeBytes = static_cast<size_t>(rowSize) * infoList[i].m_stride;
        if (srcSizeBytes > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
          conversionOk = false;
          break;
        }
        const uint32_t srcSize = static_cast<uint32_t>(srcSizeBytes);
        gfxFinish(context);
        const GfxResult result = gfxConvertMatrix(context,
            *destBuffer, dstOffset, destSizes[i], d3dLayout, 0, dataType,
            *buffer, srcOffset, srcSize, toD3D12MatrixLayout(MatrixLayout::ROW_MAJOR), static_cast<uint32_t>(infoList[i].m_stride), dataType,
            rowSize, columnSize);
        if (result != kGfxResult_NoError) {
          conversionOk = false;
          break;
        }
        srcOffset += static_cast<uint64_t>(infoList[i].m_dataSize);
        dstOffset += static_cast<uint64_t>(destSizes[i]);
      }
      if (conversionOk) {
        gfxFinish(context);
        // Verify conversion produced non-zero data
        GfxBuffer staging = gfxCreateBuffer(context, totalDestSize, nullptr, kGfxCpuAccess_Read);
        gfxCommandCopyBuffer(context, staging, 0, *destBuffer, 0, totalDestSize);
        gfxFinish(context);
        const void* mapped = gfxBufferGetData(context, staging);
        bool hasNonZero = false;
        if (mapped) {
          const auto* bytes = static_cast<const uint8_t*>(mapped);
          for (size_t b = 0; b < totalDestSize; ++b)
            if (bytes[b] != 0) { hasNonZero = true; break; }
        }
        gfxDestroyBuffer(context, staging);
        if (hasNonZero) {
          // Update info entries to reflect the GPU-converted layout
          for (size_t i = 0; i < infoList.size(); ++i) {
            infoList[i].m_layout = requestedLayout;
            infoList[i].m_dataSize = static_cast<size_t>(destSizes[i]);
          }
          return destBuffer;
        }
      }
    }
    // Conversion not available
    if (allowRowMajorFallback) {
      // info entries already have m_layout = ROW_MAJOR
      return buffer;
    }
    return {};
  }

  // ROW_MAJOR and COLUMN_MAJOR: pack directly
  const std::vector bufferData = packAsD3D12Matrix<Type>(infoList);
  return createGfxBuffer<Type>(context, bufferData);
}

//! Pack vectors into a GPU buffer using D3D12VectorInfo.
template <Arithmetic Type> inline
auto packAsD3D12VectorBuffer(GfxContext context,
                             std::span<D3D12VectorInfo<Type>> infoList) -> std::shared_ptr<GfxBuffer>
{
  const std::vector bufferData = packAsD3D12Vector<Type>(infoList);
  return createGfxBuffer<Type>(context, bufferData);
}

template <Arithmetic Type> inline
auto unpackD3D12MatrixBuffer(GfxContext context,
                             GfxBuffer buffer,
                             std::span<const D3D12MatrixInfo<Type>> infoList) -> std::vector<Type>
{
  if (infoList.empty())
    return {};

  const MatrixLayout layout = infoList[0].m_layout;

  if (needsMatrixConversion(layout)) {
    constexpr uint32_t dataType = toD3D12DataType<Type>();
    const uint32_t srcD3dLayout = toD3D12MatrixLayout(layout);
    constexpr uint32_t dstD3dLayout = toD3D12MatrixLayout(MatrixLayout::ROW_MAJOR);

    std::vector<D3D12MatrixInfo<Type>> rowMajorInfoList(infoList.begin(), infoList.end());
    for (auto& info : rowMajorInfoList)
      info.m_layout = MatrixLayout::ROW_MAJOR;

    size_t totalDstSize = 0;
    for (auto& info : rowMajorInfoList) {
      getD3D12MatrixInfo(info);
      totalDstSize += info.m_dataSize;
    }

    GfxBuffer destBuffer = gfxCreateBuffer(context, totalDstSize, nullptr, kGfxCpuAccess_None);
    gfxFinish(context);

    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    for (size_t i = 0; i < infoList.size(); ++i) {
      const uint32_t rowSize = static_cast<uint32_t>(infoList[i].m_rowSize);
      const uint32_t columnSize = static_cast<uint32_t>(infoList[i].m_columnSize);
      const uint32_t srcSize = static_cast<uint32_t>(infoList[i].m_dataSize);
      const uint32_t dstSize = static_cast<uint32_t>(rowMajorInfoList[i].m_dataSize);
      gfxFinish(context);
      gfxConvertMatrix(context,
          destBuffer, dstOffset, dstSize, dstD3dLayout, static_cast<uint32_t>(rowMajorInfoList[i].m_stride), dataType,
          buffer, srcOffset, srcSize, srcD3dLayout, 0, dataType,
          rowSize, columnSize);
      srcOffset += static_cast<uint64_t>(infoList[i].m_dataSize);
      dstOffset += static_cast<uint64_t>(rowMajorInfoList[i].m_dataSize);
    }
    gfxFinish(context);

    GfxBuffer staging = gfxCreateBuffer(context, totalDstSize, nullptr, kGfxCpuAccess_Read);
    gfxCommandCopyBuffer(context, staging, 0, destBuffer, 0, totalDstSize);
    gfxFinish(context);
    const Type* mapped = gfxBufferGetData<Type>(context, staging);

    std::vector<Type> result;
    size_t readOffset = 0;
    for (size_t i = 0; i < rowMajorInfoList.size(); ++i) {
      const size_t elemCount = rowMajorInfoList[i].m_dataSize / sizeof(Type);
      std::span<const Type> src{mapped + readOffset, elemCount};
      std::vector<Type> mat = convertToRowMatrix(rowMajorInfoList[i], src);
      result.insert(result.end(), mat.begin(), mat.end());
      readOffset += elemCount;
    }

    gfxDestroyBuffer(context, staging);
    gfxDestroyBuffer(context, destBuffer);
    return result;
  }

  GfxBuffer staging = gfxCreateBuffer(context, buffer.getSize(), nullptr, kGfxCpuAccess_Read);
  gfxCommandCopyBuffer(context, staging, buffer);
  gfxFinish(context);
  const Type* mapped = gfxBufferGetData<Type>(context, staging);

  std::vector<Type> result;
  size_t readOffset = 0;
  for (size_t i = 0; i < infoList.size(); ++i) {
    const size_t elemCount = infoList[i].m_dataSize / sizeof(Type);
    std::span<const Type> src{mapped + readOffset, elemCount};
    std::vector<Type> mat = convertToRowMatrix(infoList[i], src);
    result.insert(result.end(), mat.begin(), mat.end());
    readOffset += elemCount;
  }

  gfxDestroyBuffer(context, staging);
  return result;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_GFX_UTILITY_HPP */
