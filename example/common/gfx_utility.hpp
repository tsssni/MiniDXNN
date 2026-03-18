/*!
  \file gfx_utility.hpp
  \author Sho Ikeda
  \brief No brief description
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
#include <memory>
#include <source_location>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>
// GFX
#include "gfx.h"
//
#include "mlp_layer.hpp"
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

auto runKernel(GfxContext context, GfxProgram program, GfxKernel kernel, const size_t threadGroupSize = 1, std::initializer_list<BufferBindingDataT> bufferList = {}, std::initializer_list<IntBindingDataT> intList = {}, OptionalRef<float> execTimeInMs = std::nullopt) -> void;

class GfxAssertTrue
{
 public:
  GfxAssertTrue(const std::source_location& location = std::source_location::current()) : m_location{location} {}

  template <typename ...Args>
  auto operator()(const GfxResult result, std::format_string<Args...> messageFormat, Args&&... args) const -> void
  {
    if (result != kGfxResult_NoError) {
      std::cerr << std::format(messageFormat, std::forward<Args>(args)...)
                << std::format(" in file: {} [line: {}, function={}]", m_location.file_name(), m_location.line(), m_location.function_name())
                << std::endl;
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
  std::shared_ptr<GfxBuffer> sharedBuffer{new GfxBuffer{buffer}, [context](GfxBuffer* buffer)
  {
    if (buffer != nullptr) {
      if (*buffer) {
        GfxAssertTrue{}(gfxDestroyBuffer(context, *buffer), "Destroying the buffer failed.");
      }
      delete buffer;
    }
  }};
  return sharedBuffer;
}

template <typename Type> inline
auto createGfxBuffer(GfxContext context, std::span<const Type> data, const GfxCpuAccess cpuAccess) -> std::shared_ptr<GfxBuffer>
{
  const size_t sizeInBytes = data.size() * sizeof(Type);
  GfxBuffer buffer = gfxCreateBuffer(context, sizeInBytes, static_cast<void const*>(data.data()), cpuAccess);
  std::shared_ptr<GfxBuffer> sharedBuffer{new GfxBuffer{buffer}, [context](GfxBuffer* buffer)
  {
    if (buffer != nullptr) {
      if (*buffer) {
        GfxAssertTrue{}(gfxDestroyBuffer(context, *buffer), "Destroying the buffer failed.");
      }
      delete buffer;
    }
  }};
  return sharedBuffer;
}

template <Arithmetic Type> inline
auto convertToMatrixBuffer(GfxContext context,
                           const size_t inputDim,
                           const size_t outputDim,
                           const std::span<const Type> data,
                           const MatrixLayout layout,
                           const size_t alignment = MATRIX_ALIGNMENT,
                           const size_t strideAlignment = MATRIX_STRIDE_ALIGNMENT) -> std::shared_ptr<GfxBuffer>
{
  std::vector<std::span<const Type>> layerDataList;
  layerDataList.emplace_back(data);
  std::vector<std::tuple<size_t, size_t>> layerInfoList;
  layerInfoList.emplace_back(inputDim, outputDim);
  const std::vector bufferData = ex::packMatrixData<Type>(layerDataList, layerInfoList, layout, alignment, strideAlignment);
  std::shared_ptr buffer = ex::createGfxBuffer<Type>(context, bufferData);
  return buffer;
}

template <Arithmetic Type> inline
auto convertToMatrixBuffer(GfxContext context,
                           const std::span<const MlpLayer<Type, Type>> data,
                           const MatrixLayout layout,
                           const size_t alignment = MATRIX_ALIGNMENT,
                           const size_t strideAlignment = MATRIX_STRIDE_ALIGNMENT) -> std::shared_ptr<GfxBuffer>
{
  using MlpLayerT = MlpLayer<Type, Type>;
  std::vector<std::span<const Type>> layerDataList;
  layerDataList.reserve(data.size());
  std::vector<std::tuple<size_t, size_t>> layerInfoList;
  layerInfoList.reserve(data.size());
  for (const MlpLayerT& layer : data) {
    layerDataList.emplace_back(layer.weightData());
    layerInfoList.emplace_back(layer.inputDimension(), layer.outputDimension());
  }
  const std::vector bufferData = ex::packMatrixData<Type>(layerDataList, layerInfoList, layout, alignment, strideAlignment);
  std::shared_ptr buffer = ex::createGfxBuffer<Type>(context, bufferData);
  return buffer;
}

template <Arithmetic Type> inline
auto convertToVectorBuffer(GfxContext context,
                           const std::span<const Type> data,
                           const size_t alignment = VECTOR_ALIGNMENT) -> std::shared_ptr<GfxBuffer>
{
  std::vector<std::span<const Type>> layerDataList;
  layerDataList.emplace_back(data);
  const std::vector bufferData = ex::packVectorData<Type>(layerDataList, alignment);
  std::shared_ptr buffer = ex::createGfxBuffer<Type>(context, bufferData);
  return buffer;
}

template <Arithmetic Type> inline
auto convertToVectorBuffer(GfxContext context,
                           const std::span<const MlpLayer<Type, Type>> data,
                           const size_t alignment = VECTOR_ALIGNMENT) -> std::shared_ptr<GfxBuffer>
{
  using MlpLayerT = MlpLayer<Type, Type>;
  std::vector<std::span<const Type>> layerDataList;
  layerDataList.reserve(data.size());
  for (const MlpLayerT& layer : data) {
    layerDataList.emplace_back(layer.biasData());
  }
  const std::vector bufferData = ex::packVectorData<Type>(layerDataList, alignment);
  std::shared_ptr buffer = ex::createGfxBuffer<Type>(context, bufferData);
  return buffer;
}

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_GFX_UTILITY_HPP */

