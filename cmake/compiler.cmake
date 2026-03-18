#
# file: compiler.cmake
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

## Set C++ compile flags for the target
## Configures C++ standard, compiler-specific optimizations, and architecture features
## Args:
##   target: The CMake target to configure (executable or library)
##   scope: Visibility scope (PRIVATE, PUBLIC, or INTERFACE)
## Options (via cache variables):
##   MINIDXNN_ARCH_FEATURE_LEVEL: CPU architecture level (Amd64V1/V2/V3/V4)
function(setCxxCompileFlags target scope)
  # Detect compiler type using generator expressions
  set(has_msvc $<OR:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>)
  set(has_gcc $<OR:$<C_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:GNU>>)
  set(has_clang $<OR:$<C_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:AppleClang>>)
  set(has_apple_clang $<OR:$<C_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:AppleClang>>)
  
  # Check if using MSVC frontend (e.g., clang-cl)
  set(has_msvc_frontend 0)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(has_msvc_frontend 1)
  endif()

  # Shared compiler definitions
  set(definitions)

  # MSVC compiler flags
  set(msvc_flags /diagnostics:caret  # Enhanced diagnostic formatting
                 /nologo              # Suppress startup banner
                 /fastfail            # Enable fast-fail for security
                 /options:strict      # Strict options checking
                 )
  set(msvc_linker_flags)
  set(msvc_definitions)

  # GCC compiler flags
  set(gcc_flags)
  set(gcc_linker_flags)
  set(gcc_definitions)

  # Clang compiler flags (Unix-style frontend)
  set(clang_flags)
  set(clang_linker_flags)
  set(clang_definitions)

  # Clang-CL compiler flags (MSVC-style frontend)
  set(clang_cl_flags /Qvec              # Enable auto-vectorization
                     /diagnostics:caret  # Enhanced diagnostic formatting
                     -fcolor-diagnostics # Colorized diagnostic output
                     )
  set(clang_cl_linker_flags)
  set(clang_cl_definitions)

  # Configure CPU architecture feature level
  # x86-64 microarchitecture levels define baseline instruction set support:
  #   V1: Basic x86-64 (SSE2)
  #   V2: +SSSE3, SSE4.1, SSE4.2, POPCNT (circa 2009)
  #   V3: +AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE (circa 2015)
  #   V4: +AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL (circa 2017)
  set(feature_level "${MINIDXNN_ARCH_FEATURE_LEVEL}")
  if(feature_level STREQUAL "Amd64V1")
    list(APPEND msvc_flags /favor:AMD64)
    list(APPEND gcc_flags -march=x86-64)
    list(APPEND clang_flags -march=x86-64)
    list(APPEND clang_cl_flags /clang:-march=x86-64)
  elseif(feature_level STREQUAL "Amd64V2")
    list(APPEND msvc_flags /favor:AMD64)
    list(APPEND gcc_flags -fno-math-errno
                          -march=x86-64-v2)
    list(APPEND clang_flags -fno-math-errno
                            -march=x86-64-v2)
    list(APPEND clang_cl_flags /clang:-fno-math-errno
                               /clang:-march=x86-64-v2)
  elseif(feature_level STREQUAL "Amd64V3")
    list(APPEND msvc_flags /favor:AMD64
                           /arch:AVX2)
    list(APPEND gcc_flags -fno-math-errno
                          -march=x86-64-v3)
    list(APPEND clang_flags -fno-math-errno
                            -march=x86-64-v3)
    list(APPEND clang_cl_flags /clang:-fno-math-errno
                               /clang:-march=x86-64-v3)
  elseif(feature_level STREQUAL "Amd64V4")
    list(APPEND msvc_flags /favor:AMD64
                           /arch:AVX512)
    list(APPEND gcc_flags -fno-math-errno
                          -march=x86-64-v4)
    list(APPEND clang_flags -fno-math-errno
                            -march=x86-64-v4)
    list(APPEND clang_cl_flags /clang:-fno-math-errno
                               /clang:-march=x86-64-v4)
  endif()
  list(APPEND definitions ARCH_FEATURE_LEVEL_NAME=\"${feature_level}\")

  # Apply compiler-specific settings to target
  target_compile_features(${target} ${scope} cxx_std_20)
  
  target_compile_options(${target} ${scope}
    $<${has_msvc}:${msvc_flags}>
    $<${has_gcc}:${gcc_flags}>
    $<${has_clang}:$<IF:${has_msvc_frontend},${clang_cl_flags},${clang_flags}>>
  )
  
  target_link_options(${target} ${scope}
    $<${has_msvc}:${msvc_linker_flags}>
    $<${has_gcc}:${gcc_linker_flags}>
    $<${has_clang}:$<IF:${has_msvc_frontend},${clang_cl_linker_flags},${clang_linker_flags}>>
  )
  
  target_compile_definitions(${target} ${scope}
    ${definitions}
    $<${has_msvc}:${msvc_definitions}>
    $<${has_gcc}:${gcc_definitions}>
    $<${has_clang}:$<IF:${has_msvc_frontend},${clang_cl_definitions},${clang_definitions}>>
  )
endfunction(setCxxCompileFlags)

## Set C++ warning flags for the target
## Enables comprehensive compiler warnings to catch potential issues
## Args:
##   target: The CMake target to configure
##   scope: Visibility scope (PRIVATE, PUBLIC, or INTERFACE)
## Options:
##   MINIDXNN_WARNING_EXTRA: Enable maximum warnings (may produce false positives)
function(setCxxWarningFlags target scope)
  # Check if extra warnings are enabled
  set(has_extra $<BOOL:${MINIDXNN_WARNING_EXTRA}>)

  # Detect compiler type
  set(has_msvc $<OR:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>)
  set(has_gcc $<OR:$<C_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:GNU>>)
  set(has_clang $<OR:$<C_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:AppleClang>>)

  # MSVC warning levels
  set(msvc_options /W4)        # Warning level 4 (high but reasonable)
  set(msvc_options_extra /Wall) # All warnings (may be noisy)

  # GCC warning options
  set(gcc_options -Wall
                  -Wextra
                  -pedantic
                  -Wno-attributes  # Suppress warnings on [[maybe_unused]] for member variables
                  )
  set(gcc_options_extra ${gcc_options}
                        -Wcast-align
                        -Wcast-qual
                        -Wctor-dtor-privacy
                        -Wdisabled-optimization
                        -Wformat=2
                        -Winit-self
                        -Wlogical-op
                        -Wmissing-declarations
                        -Wmissing-include-dirs
                        -Wnoexcept
                        -Wold-style-cast
                        -Woverloaded-virtual
                        -Wredundant-decls
                        -Wshadow
                        -Wsign-conversion
                        -Wsign-promo
                        -Wstrict-null-sentinel
                        -Wstrict-overflow=5
                        -Wswitch-default
                        -Wundef
                        )

  # Clang warning options (depends on frontend variant)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # Clang-CL (MSVC-compatible frontend)
    set(clang_options /W4)
    set(clang_options_extra /Wall -Wno-c++-compat -Wno-c++98-compat -Wno-c++98-compat-pedantic)
  elseif(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    # Clang with GNU-compatible frontend
    set(clang_options -Wall -Wextra -pedantic)
    set(clang_options_extra -Weverything -Wno-c++-compat -Wno-c++98-compat -Wno-c++98-compat-pedantic)
  endif()

  # Apply warning flags to target
  target_compile_options(${target} ${scope}
    $<${has_msvc}:$<IF:${has_extra},${msvc_options_extra},${msvc_options}>>
    $<${has_gcc}:$<IF:${has_extra},${gcc_options_extra},${gcc_options}>>
    $<${has_clang}:$<IF:${has_extra},${clang_options_extra},${clang_options}>>
  )
endfunction(setCxxWarningFlags)

## Set sanitizer flags for the target
## Enables runtime sanitizers for detecting bugs and undefined behavior
## Args:
##   target: The CMake target to configure
##   scope: Visibility scope (PRIVATE, PUBLIC, or INTERFACE)
## Note: 
##   - Most sanitizers are mutually exclusive (address, thread, memory)
##   - Requires Clang or GCC compiler support
##   - Sanitizers significantly impact performance and memory usage
function(setSanitizerFlags target scope)
  # Check which sanitizers are enabled via cache options
  set(has_address $<BOOL:${MINIDXNN_ENABLE_SANITIZER_ADDRESS}>)
  #set(has_thread $<BOOL:${MINIDXNN_ENABLE_SANITIZER_THREAD}>)
  #set(has_memory $<BOOL:${MINIDXNN_ENABLE_SANITIZER_MEMORY}>)
  #set(has_undef $<BOOL:${MINIDXNN_ENABLE_SANITIZER_UNDEF_BEHAVIOR}>)
  #set(has_leak $<BOOL:${MINIDXNN_ENABLE_SANITIZER_LEAK}>)
  #set(has_cfi $<BOOL:${MINIDXNN_ENABLE_SANITIZER_CFI}>)
  #set(has_safe_stack $<BOOL:${MINIDXNN_ENABLE_SANITIZER_SAFE_STACK}>)

  # Apply sanitizer compile options
  # TODO: Support other sanitizers
  target_compile_options(${target} ${scope}
    $<${has_address}:-fsanitize=address;-fno-omit-frame-pointer>
    #$<${has_thread}:-fsanitize=thread>
    #$<${has_memory}:-fsanitize=memory;-fno-omit-frame-pointer>
    #$<${has_undef}:-fsanitize=undefined,float-divide-by-zero,unsigned-integer-overflow,implicit-conversion,local-bounds,nullability>
    #$<${has_leak}:-fsanitize=leak>
    #$<${has_cfi}:-fsanitize=cfi>
    #$<${has_safe_stack}:-fsanitize=safe-stack>
  )
  
  # Apply sanitizer link options (must match compile options)
  target_link_options(${target} ${scope}
    $<${has_address}:-fsanitize=address>
    #$<${has_thread}:-fsanitize=thread>
    #$<${has_memory}:-fsanitize=memory>
    #$<${has_undef}:-fsanitize=undefined,float-divide-by-zero,unsigned-integer-overflow,implicit-conversion,local-bounds,nullability>
    #$<${has_leak}:-fsanitize=leak>
    #$<${has_cfi}:-fsanitize=cfi>
    #$<${has_safe_stack}:-fsanitize=safe-stack>
  )
endfunction(setSanitizerFlags)
