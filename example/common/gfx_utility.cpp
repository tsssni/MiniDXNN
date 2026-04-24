/*!
  \file gfx_utility.cpp
  \author Sho Ikeda
  \brief GFX utility implementations for GPU resource management
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#include "gfx_utility.hpp"
// Standard C++ library
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <span>
#include <string_view>
#include <vector>
// GFX
#include "gfx.h"
#include "gfx_window.h"
// Example
#include "utility.hpp"

namespace ex {

auto createGfxContext(const bool enableDebugShader) -> std::shared_ptr<GfxContext>
{
  // Create a GFX context
  GfxWindow gfxWindow = gfxCreateWindow(1280, 720, "MiniDXNN", kGfxCreateWindowFlag_HideWindow);
  std::shared_ptr<GfxWindow> gfxWindowPtr(new GfxWindow{gfxWindow}, [](GfxWindow* window)
  {
    if (window != nullptr) {
      if (not *window) {
        std::cerr << "[ERROR] Failed to destroy the gfx window.\n";
        std::abort();
      }
      GfxAssertTrue{}(gfxDestroyWindow(*window), "Destroying the window failed.");
      delete window;
    }
  });
  GfxCreateContextFlags flags = kGfxCreateContextFlag_EnableExperimentalShaders;
  if (enableDebugShader) {
    flags |= kGfxCreateContextFlag_EnableShaderDebugging;
  }
  GfxContext gfxContext = gfxCreateContext(*gfxWindowPtr, flags);
  auto* rawGfxContext = new GfxContext{gfxContext};
  std::shared_ptr<GfxContext> gfxContextPtr(rawGfxContext, [gfxWindowPtr](GfxContext* context) mutable
  {
    if (context != nullptr) {
      if (not *context) {
        std::cerr << "[ERROR] Failed to destroy the gfx context.\n";
        std::abort();
      }
      GfxAssertTrue{}(gfxDestroyContext(*context), "Destroying the context failed.");
      delete context;
    }
    gfxWindowPtr.reset();
  });

  return gfxContextPtr;
}

auto createGfxProgram(GfxContext context, const std::string_view fileName, const std::filesystem::path& dirPath, const std::span<const OptionString> includePathList) -> std::shared_ptr<GfxProgram>
{
  const std::string_view shaderMode = "6_9";
  std::vector<const char*> pathList;
  pathList.resize(includePathList.size());
  std::ranges::transform(includePathList, pathList.begin(), [](const OptionString& option) -> const char*
  {
    return option.data();
  });
  GfxProgram program = gfxCreateProgram(context, fileName.data(), dirPath.string().c_str(), shaderMode.data(), pathList.data(), static_cast<std::uint32_t>(pathList.size()));
  std::shared_ptr<GfxProgram> sharedProgram(new GfxProgram{program}, [context](GfxProgram* ptr)
  {
    if (ptr != nullptr) {
      if (*ptr) {
        GfxAssertTrue{}(gfxDestroyProgram(context, *ptr), "Destroying the program failed.");
      }
      delete ptr;
    }
  });
  return sharedProgram;
}

auto createGfxComputeKernel(GfxContext context, GfxProgram program, const std::string_view entryPoint, const std::span<const OptionString> definitionList) -> std::shared_ptr<GfxKernel>
{
  std::vector<const char*> defList;
  defList.resize(definitionList.size());
  std::ranges::transform(definitionList, defList.begin(), [](const OptionString& option)
  {
    return option.data();
  });
  GfxKernel kernel = gfxCreateComputeKernel(context, program, entryPoint.data(), defList.data(), static_cast<std::uint32_t>(defList.size()));
  std::shared_ptr<GfxKernel> sharedKernel(new GfxKernel{kernel}, [context](GfxKernel* ptr)
  {
    if (ptr != nullptr) {
      if (*ptr) {
        GfxAssertTrue{}(gfxDestroyKernel(context, *ptr), "Destroying the kernel failed.");
      }
      delete ptr;
    }
  });
  return sharedKernel;
}

auto runKernel(GfxContext context, GfxProgram program, GfxKernel kernel, const size_t threadGroupSize, std::initializer_list<BufferBindingDataT> bufferList, std::initializer_list<IntBindingDataT> intList, OptionalRef<float> execTimeInMs, std::initializer_list<FloatBindingDataT> floatList) -> void
{
  // Bind parameters
  GfxAssertTrue{}(gfxCommandBindKernel(context, kernel), "Binding the kernel failed.");
  for (const BufferBindingDataT& data : bufferList) {
    GfxAssertTrue{}(gfxProgramSetBuffer(context, program, data.m_name.data(), data.m_value), "");
  }
  for (const IntBindingDataT& data : intList) {
    GfxAssertTrue{}(gfxProgramSetParameter<std::int32_t>(context, program, data.m_name.data(), data.m_value), "");
  }
  for (const FloatBindingDataT& data : floatList) {
    GfxAssertTrue{}(gfxProgramSetParameter<float>(context, program, data.m_name.data(), data.m_value), "");
  }

  // Start measuring the kernel execution time
  std::shared_ptr<GfxTimestampQuery> timestamp;
  if (execTimeInMs.has_value()) {
    GfxTimestampQuery stamp = gfxCreateTimestampQuery(context);
    timestamp.reset(new GfxTimestampQuery{stamp}, [context](GfxTimestampQuery* t)
    {
      if (t != nullptr) {
        if (*t) {
          GfxAssertTrue{}(gfxDestroyTimestampQuery(context, *t), "");
        }
        delete t;
      }
    });
  }
  if (timestamp) GfxAssertTrue{}(gfxCommandBeginTimestampQuery(context, *timestamp), "");

  // Run the kernel
  GfxAssertTrue{}(gfxCommandDispatch(context, static_cast<std::uint32_t>(threadGroupSize), 1, 1), "Dispatching the command failed.");

  // End
  if (timestamp) {
    GfxAssertTrue{}(gfxCommandEndTimestampQuery(context, *timestamp), "");
    GfxAssertTrue{}(gfxCommandResolveTimestamp(context), "");
  }

  // Wait for the kernel completion
  GfxAssertTrue{}(gfxFinish(context), "");

  // Kernel execution time
  if (timestamp) {
    // TODO. Get the duration time
    GfxAssertTrue{}(gfxCommandUpdateTimestamp(context), "");
    const float execTime = gfxTimestampQueryGetDuration(context, *timestamp);
    execTimeInMs->get() = execTime;
  }
  timestamp.reset();
}

auto runKernel(GfxContext context, GfxProgram program, GfxKernel kernel, const size_t threadGroupSize, std::span<const BufferBindingDataT> bufferList, std::initializer_list<IntBindingDataT> intList, OptionalRef<float> execTimeInMs, std::initializer_list<FloatBindingDataT> floatList) -> void
{
  // Bind parameters
  GfxAssertTrue{}(gfxCommandBindKernel(context, kernel), "Binding the kernel failed.");
  for (const BufferBindingDataT& data : bufferList) {
    GfxAssertTrue{}(gfxProgramSetBuffer(context, program, data.m_name.data(), data.m_value), "");
  }
  for (const IntBindingDataT& data : intList) {
    GfxAssertTrue{}(gfxProgramSetParameter<std::int32_t>(context, program, data.m_name.data(), data.m_value), "");
  }
  for (const FloatBindingDataT& data : floatList) {
    GfxAssertTrue{}(gfxProgramSetParameter<float>(context, program, data.m_name.data(), data.m_value), "");
  }

  // Start measuring the kernel execution time
  std::shared_ptr<GfxTimestampQuery> timestamp;
  if (execTimeInMs.has_value()) {
    GfxTimestampQuery stamp = gfxCreateTimestampQuery(context);
    timestamp.reset(new GfxTimestampQuery{stamp}, [context](GfxTimestampQuery* t)
    {
      if (t != nullptr) {
        if (*t) {
          GfxAssertTrue{}(gfxDestroyTimestampQuery(context, *t), "");
        }
        delete t;
      }
    });
  }
  if (timestamp) GfxAssertTrue{}(gfxCommandBeginTimestampQuery(context, *timestamp), "");

  // Run the kernel
  GfxAssertTrue{}(gfxCommandDispatch(context, static_cast<std::uint32_t>(threadGroupSize), 1, 1), "Dispatching the command failed.");

  // End
  if (timestamp) {
    GfxAssertTrue{}(gfxCommandEndTimestampQuery(context, *timestamp), "");
    GfxAssertTrue{}(gfxCommandResolveTimestamp(context), "");
  }

  // Wait for the kernel completion
  GfxAssertTrue{}(gfxFinish(context), "");

  // Kernel execution time
  if (timestamp) {
    GfxAssertTrue{}(gfxCommandUpdateTimestamp(context), "");
    const float execTime = gfxTimestampQueryGetDuration(context, *timestamp);
    execTimeInMs->get() = execTime;
  }
  timestamp.reset();
}

} /* namespace ex */

