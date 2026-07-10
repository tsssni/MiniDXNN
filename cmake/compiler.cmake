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
##   MINIDXNN_COMPILER_ARCH_FEATURE_LEVEL: CPU architecture level (Amd64V1/V2/V3/V4)
function(setCxxCompileFlags target scope)
  # Detect compiler type using generator expressions
  set(has_msvc $<CXX_COMPILER_ID:MSVC>)
  set(has_gcc $<CXX_COMPILER_ID:GNU>)
  set(has_apple_clang $<CXX_COMPILER_ID:AppleClang>)
  set(has_clang $<OR:$<CXX_COMPILER_ID:Clang>,${has_apple_clang}>)
  
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
  set(msvc_definitions _CRT_SECURE_NO_WARNINGS)  # Suppress deprecated CRT warnings from third-party headers (stb_image)

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
  # The level is validated in options.cmake, so exactly one branch matches.
  set(feature_level "${MINIDXNN_COMPILER_ARCH_FEATURE_LEVEL}")
  set(is_v1 $<STREQUAL:${feature_level},Amd64V1>)
  set(is_v2 $<STREQUAL:${feature_level},Amd64V2>)
  set(is_v3 $<STREQUAL:${feature_level},Amd64V3>)
  set(is_v4 $<STREQUAL:${feature_level},Amd64V4>)
  # -fno-math-errno applies to every level except V1.
  set(has_fast_math $<NOT:${is_v1}>)

  list(APPEND msvc_flags /favor:AMD64
                         $<${is_v3}:/arch:AVX2>
                         $<${is_v4}:/arch:AVX512>)
  list(APPEND gcc_flags $<${has_fast_math}:-fno-math-errno>
                        $<${is_v1}:-march=x86-64>
                        $<${is_v2}:-march=x86-64-v2>
                        $<${is_v3}:-march=x86-64-v3>
                        $<${is_v4}:-march=x86-64-v4>)
  list(APPEND clang_flags $<${has_fast_math}:-fno-math-errno>
                          $<${is_v1}:-march=x86-64>
                          $<${is_v2}:-march=x86-64-v2>
                          $<${is_v3}:-march=x86-64-v3>
                          $<${is_v4}:-march=x86-64-v4>)
  list(APPEND clang_cl_flags $<${has_fast_math}:/clang:-fno-math-errno>
                             $<${is_v1}:/clang:-march=x86-64>
                             $<${is_v2}:/clang:-march=x86-64-v2>
                             $<${is_v3}:/clang:-march=x86-64-v3>
                             $<${is_v4}:/clang:-march=x86-64-v4>)
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
## Enables compiler warnings to catch potential issues
## Args:
##   target: The CMake target to configure
##   scope: Visibility scope (PRIVATE, PUBLIC, or INTERFACE)
## Options:
##   MINIDXNN_COMPILER_WARNINGS: Warning level (off, default, extra, full).
##     Modeled on premake's `warnings` API:
##       off     -> suppress all warnings
##       default -> compiler default (no extra flags)
##       extra   -> a broad reasonable set (-Wall -Wextra / /W4)
##       full    -> the maximum set (-Weverything / /Wall) with noise suppressions
function(setCxxWarningFlags target scope)
  set(level "${MINIDXNN_COMPILER_WARNINGS}")

  # Detect compiler type using generator expressions
  set(has_msvc $<CXX_COMPILER_ID:MSVC>)
  set(has_gcc $<CXX_COMPILER_ID:GNU>)
  set(has_apple_clang $<CXX_COMPILER_ID:AppleClang>)
  set(has_clang $<OR:$<CXX_COMPILER_ID:Clang>,${has_apple_clang}>)
  set(is_clang_cl 0)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(is_clang_cl 1)
  endif()

  # MSVC warning sets
  set(msvc_off /W0)
  set(msvc_extra /W4 /wd4244)        # Warning level 4 (high but reasonable)
  set(msvc_full /Wall
                /wd4244
                /wd4514   # unreferenced inline function removed (informational)
                /wd4625   # copy constructor implicitly deleted (GoogleTest)
                /wd4626   # copy assignment implicitly deleted (GoogleTest)
                /wd4702   # unreachable code (can be triggered by compile-time branching)
                /wd4710   # function not inlined (compiler decision, informational)
                /wd4711   # function selected for automatic inline expansion (informational)
                /wd4820   # struct padding added (informational)
                /wd4866   # left-to-right evaluation order (third-party)
                /wd4868   # left-to-right evaluation order in braced init (third-party)
                /wd5026   # move constructor implicitly deleted (GoogleTest)
                /wd5027   # move assignment implicitly deleted (GoogleTest)
                /wd5045   # Spectre mitigation insertion (informational)
                /wd4324   # structure padded due to alignment specifier
                /wd5264   # 'const' variable is not used (used only with GFX)
                ) # All warnings with noise suppressions

  # GCC warning sets
  set(gcc_off -w)
  set(gcc_extra -Wall
                -Wextra
                -pedantic
                -Wno-attributes  # Suppress warnings on [[maybe_unused]] for member variables
                )
  set(gcc_full ${gcc_extra}
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

  # Clang warning sets (depend on frontend variant)
  set(clang_off -w)
  if(is_clang_cl)
    # Clang-CL (MSVC-compatible frontend)
    set(clang_extra /W4)
    set(clang_full /Wall -Wno-c++-compat -Wno-c++98-compat -Wno-c++98-compat-pedantic
                   -Wno-padded                     # Struct padding is intentional
                   -Wno-covered-switch-default      # All enum values handled; default added for safety
                   -Wno-global-constructors         # Google Test macros require global constructors
                   -Wno-unsafe-buffer-usage         # Clang hardening opt-in; valid pointer usage
                   -Wno-unsafe-buffer-usage-in-libc-call
                   -Wno-unsafe-buffer-usage-in-container
                   -Wno-weak-vtables                # vtable placement hint, not a bug
                   -Wno-ctad-maybe-unsupported      # CTAD is intentional C++17/20 usage
                   -Wno-missing-prototypes          # C-only concept, not meaningful for C++
                   -Wno-unused-template             # Templates in headers may not be used in every TU
                   -Wno-undefined-func-template     # Template instantiation across TUs is by design
                   )
  else()
    # Clang with GNU-compatible frontend
    set(clang_extra -Wall -Wextra -pedantic)
    set(clang_full -Weverything -Wno-c++-compat -Wno-c++98-compat -Wno-c++98-compat-pedantic
                   -Wno-padded                     # Struct padding is intentional
                   -Wno-covered-switch-default      # All enum values handled; default added for -Wswitch-default
                   -Wno-global-constructors         # Google Test macros require global constructors
                   -Wno-unsafe-buffer-usage         # Clang hardening opt-in; valid pointer usage
                   -Wno-unsafe-buffer-usage-in-libc-call
                   -Wno-unsafe-buffer-usage-in-container
                   -Wno-weak-vtables                # vtable placement hint, not a bug
                   -Wno-ctad-maybe-unsupported      # CTAD is intentional C++17/20 usage
                   -Wno-missing-prototypes          # C-only concept, not meaningful for C++
                   -Wno-unused-template             # Templates in headers may not be used in every TU
                   -Wno-undefined-func-template     # Template instantiation across TUs is by design
                   )
  endif()

  # "default" adds nothing; the other levels select one set per compiler.
  # The value is validated in options.cmake.
  if(level STREQUAL "default")
    return()
  endif()
  set(is_off $<STREQUAL:${level},off>)
  set(is_extra $<STREQUAL:${level},extra>)
  set(is_full $<STREQUAL:${level},full>)

  set(msvc_options $<${is_off}:${msvc_off}>$<${is_extra}:${msvc_extra}>$<${is_full}:${msvc_full}>)
  set(gcc_options $<${is_off}:${gcc_off}>$<${is_extra}:${gcc_extra}>$<${is_full}:${gcc_full}>)
  set(clang_options $<${is_off}:${clang_off}>$<${is_extra}:${clang_extra}>$<${is_full}:${clang_full}>)

  # Apply warning flags to target
  target_compile_options(${target} ${scope}
    $<${has_msvc}:${msvc_options}>
    $<${has_gcc}:${gcc_options}>
    $<${has_clang}:${clang_options}>
  )
endfunction(setCxxWarningFlags)

## Set sanitizer flags for the target
## Enables runtime sanitizers for detecting bugs and undefined behavior
## Args:
##   target: The CMake target to configure
##   scope: Visibility scope (PRIVATE, PUBLIC, or INTERFACE)
## Options:
##   MINIDXNN_COMPILER_SANITIZER: Sanitizer (off, address, thread, undefinedbehavior).
##     Modeled on premake's `sanitize` API. The levels are mutually exclusive.
## Note:
##   - Requires Clang or GCC compiler support
##   - Sanitizers significantly impact performance and memory usage
function(setSanitizerFlags target scope)
  # "off" adds nothing; the value is validated in options.cmake.
  set(sanitizer "${MINIDXNN_COMPILER_SANITIZER}")
  if(sanitizer STREQUAL "off")
    return()
  endif()

  # Detect compiler type using generator expressions
  set(has_gcc $<CXX_COMPILER_ID:GNU>)
  set(has_apple_clang $<CXX_COMPILER_ID:AppleClang>)
  set(has_clang $<OR:$<CXX_COMPILER_ID:Clang>,${has_apple_clang}>)

  set(is_address $<STREQUAL:${sanitizer},address>)
  set(is_thread $<STREQUAL:${sanitizer},thread>)
  set(is_undef $<STREQUAL:${sanitizer},undefinedbehavior>)

  # UBSan: GCC supports a subset; Clang supports additional checks.
  set(ubsan_gcc -fsanitize=undefined,float-divide-by-zero)
  set(ubsan_clang -fsanitize=undefined,float-divide-by-zero,unsigned-integer-overflow,implicit-conversion,local-bounds,nullability)
  set(ubsan $<${has_gcc}:${ubsan_gcc}>$<${has_clang}:${ubsan_clang}>)

  # MSVC (Unix-style sanitizer flags are unsupported) gets nothing.
  set(supported $<OR:${has_gcc},${has_clang}>)
  set(compile_flags
    $<${is_address}:-fsanitize=address;-fno-omit-frame-pointer>
    $<${is_thread}:-fsanitize=thread>
    $<${is_undef}:${ubsan}>)
  set(link_flags
    $<${is_address}:-fsanitize=address>
    $<${is_thread}:-fsanitize=thread>
    $<${is_undef}:${ubsan}>)

  target_compile_options(${target} ${scope} $<${supported}:${compile_flags}>)
  target_link_options(${target} ${scope} $<${supported}:${link_flags}>)
endfunction(setSanitizerFlags)
