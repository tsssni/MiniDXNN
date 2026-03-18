#
# file: project.cmake
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.21)


## Build half-precision floating-point library dependency
## Extracts and configures the IEEE 754-based half-precision (16-bit) float library
## Creates an INTERFACE library exposing half.hpp header
set(project_desc "Half-precision floating-point library for IEEE 754-based 16-bit floats")
project(half-float-dep DESCRIPTION "${project_desc}" VERSION 2.2.1 LANGUAGES CXX)
  cmake_path(SET half_float_zip_path "${PROJECT_SOURCE_DIR}/half-${PROJECT_VERSION}.zip")
  cmake_path(SET half_float_dest_path "${PROJECT_BINARY_DIR}/half")
  file(ARCHIVE_EXTRACT INPUT ${half_float_zip_path} DESTINATION "${half_float_dest_path}")

  add_library(${PROJECT_NAME} INTERFACE)
  target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE "${half_float_dest_path}/include")
  target_compile_definitions(${PROJECT_NAME} INTERFACE NOMINMAX)
