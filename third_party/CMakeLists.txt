#
# file: CMakeLists.txt
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.21)

# Include utility functions and dependency configurations
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/utility.cmake)

set(project_desc "Third-party libraries")
project(ThirdParty DESCRIPTION ${project_desc} LANGUAGES CXX)
  # Build half-precision floating point library
  add_subdirectory("${PROJECT_SOURCE_DIR}/half_float_dep" "${PROJECT_BINARY_DIR}/half_float_dep")
  checkTarget(half-float-dep)

  # Build graphics library (skip in C++ fallback mode)
  if(NOT MINIDXNN_CPP_FALLBACK_ONLY)
    add_subdirectory("${PROJECT_SOURCE_DIR}/gfx_dep" "${PROJECT_BINARY_DIR}/gfx_dep")
    checkTarget(gfx-dep)
  endif()

  # Build CLI11
  add_subdirectory("${PROJECT_SOURCE_DIR}/cli11_dep" "${PROJECT_BINARY_DIR}/cli11_dep")
  checkTarget(cli11-dep)

  # Build stb_image (header-only image loading/writing)
  add_subdirectory("${PROJECT_SOURCE_DIR}/stb_image_dep" "${PROJECT_BINARY_DIR}/stb_image_dep")
  checkTarget(stb-image-dep)

  # Build Google Test framework
  add_subdirectory("${PROJECT_SOURCE_DIR}/gtest_dep" "${PROJECT_BINARY_DIR}/gtest_dep")
  checkTarget(gtest-dep)
