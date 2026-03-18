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
include(${CMAKE_CURRENT_LIST_DIR}/cmake/options.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/project.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utility.cmake)

set(project_desc "A minimal DirectX-based neural network library")
project(MiniDxNN DESCRIPTION ${project_desc} VERSION 0.1.0 LANGUAGES CXX)

  # Configure project options (build flags, sanitizers, etc.)
  setProjectOptions()

  # Top-level project configuration
  # Only applied when MiniDxNN is the main project (not when included as a subproject)
  if(PROJECT_IS_TOP_LEVEL)
    # Configure available build types (multi-config generators like Visual Studio)
    set(CMAKE_CONFIGURATION_TYPES "Debug" "RelWithDebInfo" "Release")

    # Set default build type for single-config generators (Unix Makefiles, Ninja)
    if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE "Debug")
      message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Debug")
    endif()

    # Generate compile_commands.json for IDE integration and tools (clangd, clang-tidy, etc.)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  endif()

  # Build third-party dependencies
  add_subdirectory("${PROJECT_SOURCE_DIR}/third_party" "${PROJECT_BINARY_DIR}/third_party")

  # Configure MiniDxNN core library
  setMiniDxNNCore(minidxnn-core)
  checkTarget(minidxnn-core)

  # Configure and build examples
  if(PROJECT_IS_TOP_LEVEL AND MINIDXNN_BUILD_EXAMPLES)
    message(STATUS "Enable examples")
    add_subdirectory("${PROJECT_SOURCE_DIR}/example" "${PROJECT_BINARY_DIR}/example")
  endif()

  # Configure and build unit tests
  if(PROJECT_IS_TOP_LEVEL AND MINIDXNN_BUILD_TESTS)
    message(STATUS "Enable unittests")
    enable_testing()
    add_subdirectory("${PROJECT_SOURCE_DIR}/unittest" "${PROJECT_BINARY_DIR}/unittest")
  endif()
