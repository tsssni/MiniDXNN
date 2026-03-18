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
  set(MINIDXNN_ARCH_FEATURE_LEVEL "Amd64V3" CACHE STRING 
      "CPU architecture feature level: Amd64V1, Amd64V2, Amd64V3, or Amd64V4")

  # Warning options
  option(MINIDXNN_WARNING_EXTRA 
    "Enable comprehensive compiler warnings (-Wall, -Wextra, etc.)" 
    OFF)

  # Sanitizer options for runtime error detection
  # Note: Sanitizers significantly slow down execution and increase memory usage
  # Most sanitizers are mutually exclusive (don't enable address + thread simultaneously)
  option(MINIDXNN_ENABLE_SANITIZER_ADDRESS 
    "AddressSanitizer: Detect memory errors (buffer overflows, use-after-free, etc.)" 
    OFF)

  # TODO: Support other sanitizers
  #option(MINIDXNN_ENABLE_SANITIZER_THREAD 
  #  "ThreadSanitizer: Detect data races and threading issues" 
  #  OFF)
  #option(MINIDXNN_ENABLE_SANITIZER_MEMORY 
  #  "MemorySanitizer: Detect reads of uninitialized memory (Clang/LLVM only)" 
  #  OFF)
  #option(MINIDXNN_ENABLE_SANITIZER_UNDEF_BEHAVIOR 
  #  "UndefinedBehaviorSanitizer: Detect undefined behavior (integer overflow, null deref, etc.)" 
  #  OFF)
  #option(MINIDXNN_ENABLE_SANITIZER_LEAK 
  #  "LeakSanitizer: Detect memory leaks at program exit" 
  #  OFF)
  #option(MINIDXNN_ENABLE_SANITIZER_CFI 
  #  "Control Flow Integrity: Protect against control flow hijacking (Clang with LTO only)" 
  #  OFF)
  #option(MINIDXNN_ENABLE_SANITIZER_SAFE_STACK 
  #  "SafeStack: Protect against stack buffer overflows (Clang only)" 
  #  OFF)

  # MiniDxNN options
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
