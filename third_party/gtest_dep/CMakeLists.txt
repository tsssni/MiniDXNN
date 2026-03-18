#
# file: CMakeLists.txt.cmake
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.21)

# CMake dependencies
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/utility.cmake)

## Build GoogleTest framework dependency
## Configures GoogleTest with project-specific options
## Disables unused features (gmock, installation, samples) for faster builds
set(project_desc "Google Testing and Mocking Framework")
project(gtest-dep DESCRIPTION "${project_desc}" VERSION 1.17.0 LANGUAGES CXX)
  fixSubmoduleOption(BUILD_GMOCK OFF)
  fixSubmoduleOption(INSTALL_GTEST OFF)
  fixSubmoduleOption(GTEST_HAS_ABSL OFF)
  fixSubmoduleOption(gtest_force_shared_crt ON)
  fixSubmoduleOption(gtest_build_tests OFF)
  fixSubmoduleOption(gtest_build_samples OFF)
  fixSubmoduleOption(gtest_disable_pthreads ON)
  fixSubmoduleOption(gtest_hide_internal_symbols ON)
  add_subdirectory("${PROJECT_SOURCE_DIR}/googletest" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" EXCLUDE_FROM_ALL)

  set_target_properties(gtest
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
      LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
      PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  set_target_properties(gtest_main
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
      LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
      PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

  # Set workaround of warnings
  set(has_clang $<OR:$<C_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:AppleClang>>)
  target_compile_options(gtest PRIVATE $<${has_clang}:-Wno-implicit-int-float-conversion>)
  target_compile_options(gtest_main PRIVATE $<${has_clang}:-Wno-implicit-int-float-conversion>)

  # 
  setCxxCompileFlags(gtest PRIVATE)
  setCxxCompileFlags(gtest_main PRIVATE)
  setSanitizerFlags(gtest PRIVATE)
  setSanitizerFlags(gtest_main PRIVATE)

  #
  add_library(${PROJECT_NAME} ALIAS gtest)
