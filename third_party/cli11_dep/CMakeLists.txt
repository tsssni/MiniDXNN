#
# file: CMakeLists.txt
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.21)

# CMake dependencies
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/utility.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler.cmake)

set(project_desc "")
project(cli11-dep DESCRIPTION "${project_desc}" LANGUAGES CXX)
  fixSubmoduleOption(CLI11_WARNINGS_AS_ERRORS OFF)
  fixSubmoduleOption(CLI11_SINGLE_FILE OFF)
  fixSubmoduleOption(CLI11_SANITIZERS OFF)
  fixSubmoduleOption(CLI11_BUILD_DOCS OFF)
  fixSubmoduleOption(CLI11_BUILD_TESTS OFF)
  fixSubmoduleOption(CLI11_BUILD_EXAMPLES OFF)
  fixSubmoduleOption(CLI11_BUILD_EXAMPLES_JSON OFF)
  fixSubmoduleOption(CLI11_PRECOMPILED OFF)
  fixSubmoduleOption(CLI11_SINGLE_FILE_TESTS OFF)
  fixSubmoduleOption(CLI11_INSTALL OFF)
  fixSubmoduleOption(CLI11_FORCE_LIBCXX OFF)
  fixSubmoduleOption(CLI11_CUDA_TESTS OFF)
  fixSubmoduleOption(CLI11_CLANG_TIDY_OPTIONS "")
  add_subdirectory("${PROJECT_SOURCE_DIR}/CLI11" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" EXCLUDE_FROM_ALL)
  mark_as_advanced(FORCE CLI11_SANITIZERS)

  #
  setCxxCompileFlags(CLI11 INTERFACE)
  setSanitizerFlags(CLI11 INTERFACE)

  # 
  add_library(${PROJECT_NAME} ALIAS CLI11)
