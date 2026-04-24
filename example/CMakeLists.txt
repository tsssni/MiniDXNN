#
# file: CMakeLists.txt
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.21)

# cmake dependencies
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/utility.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/project.cmake)

#
set(project_desc "MiniDXNN examples")
project(example DESCRIPTION ${project_desc} LANGUAGES CXX)
  setExampleCommon(example-common)
  checkTarget(example-common)
  cmake_path(SET hlsl_include_hpp_dir "${PROJECT_BINARY_DIR}/include")
  createHlslIncludeDirsHpp("${PROJECT_SOURCE_DIR}/kernel" "${PROJECT_BINARY_DIR}" ${hlsl_include_hpp_dir})

  buildExample(01-texture-inference 01_texture_inference ${hlsl_include_hpp_dir})
  checkTarget(01-texture-inference)
  buildExample(02-texture-training 02_texture_training ${hlsl_include_hpp_dir})
  checkTarget(02-texture-training)
