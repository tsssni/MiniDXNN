#
# file: options.cmake
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

## Configure project build options
## Sets up cache variables that control compilation and features
function(setProjectOptions)
  # Build configuration options
  option(BUILD_SHARED_LIBS 
    "Build shared libraries (.dll/.so) instead of static (.lib/.a)" 
    OFF)

  # CPU architecture feature level
  # Determines which instruction sets are available (SSE, AVX, AVX2, AVX-512)
  set(MINIDXNN_COMPILER_ARCH_FEATURE_LEVEL "Amd64V3" CACHE STRING
      "CPU architecture feature level: Amd64V1, Amd64V2, Amd64V3, or Amd64V4")
  set_property(CACHE MINIDXNN_COMPILER_ARCH_FEATURE_LEVEL PROPERTY STRINGS
      Amd64V1 Amd64V2 Amd64V3 Amd64V4)
  if(NOT MINIDXNN_COMPILER_ARCH_FEATURE_LEVEL MATCHES "^Amd64V[1-4]$")
    message(FATAL_ERROR
      "MINIDXNN_COMPILER_ARCH_FEATURE_LEVEL must be one of: Amd64V1, Amd64V2, "
      "Amd64V3, Amd64V4 (got '${MINIDXNN_COMPILER_ARCH_FEATURE_LEVEL}')")
  endif()

  # Warning level (modeled on premake's `warnings` API)
  #   off     : disable all warnings
  #   default : compiler default (no extra warning flags)
  #   extra   : a reasonable broad set (-Wall -Wextra -pedantic / -W4)
  #   full    : the maximum set (-Weverything / /Wall) with noise suppressions
  set(MINIDXNN_COMPILER_WARNINGS "default" CACHE STRING
      "Compiler warning level: off, default, extra, full")
  set_property(CACHE MINIDXNN_COMPILER_WARNINGS PROPERTY STRINGS
      off default extra full)
  if(NOT MINIDXNN_COMPILER_WARNINGS MATCHES "^(off|default|extra|full)$")
    message(FATAL_ERROR
      "MINIDXNN_COMPILER_WARNINGS must be one of: off, default, extra, full "
      "(got '${MINIDXNN_COMPILER_WARNINGS}')")
  endif()

  # Sanitizer (modeled on premake's `sanitize` API)
  #   off               : no sanitizer
  #   address           : AddressSanitizer (buffer overflow, use-after-free)
  #   thread            : ThreadSanitizer (data races)
  #   undefinedbehavior : UndefinedBehaviorSanitizer
  # Note: sanitizers significantly slow execution and increase memory use;
  #       the levels are mutually exclusive.
  set(MINIDXNN_COMPILER_SANITIZER "off" CACHE STRING
      "Compiler sanitizer: off, address, thread, undefinedbehavior")
  set_property(CACHE MINIDXNN_COMPILER_SANITIZER PROPERTY STRINGS
      off address thread undefinedbehavior)
  if(NOT MINIDXNN_COMPILER_SANITIZER MATCHES "^(off|address|thread|undefinedbehavior)$")
    message(FATAL_ERROR
      "MINIDXNN_COMPILER_SANITIZER must be one of: off, address, thread, "
      "undefinedbehavior (got '${MINIDXNN_COMPILER_SANITIZER}')")
  endif()

  # MiniDXNN options
  option(MINIDXNN_BUILD_CPP_FALLBACK_ONLY
    "Build without DirectX 12 / gfx dependency; use C++ fallback for MLP computation"
    OFF)
  option(MINIDXNN_BUILD_EXAMPLES 
    "Build example programs demonstrating library usage" 
    ON)
  option(MINIDXNN_BUILD_TESTS 
    "Build unit tests using GoogleTest" 
    OFF)
  option(MINIDXNN_TEST_ENABLE_FP32_TESTS
    "Add unit tests for 32-bit floating point into GoogleTest" 
    OFF)

endfunction(setProjectOptions)
