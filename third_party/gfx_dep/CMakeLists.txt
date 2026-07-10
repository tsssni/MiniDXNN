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

## Build gfx DirectX 12 framework dependency
## Configures the gfx library for Direct3D 12 rendering
## Applies project-specific compiler and sanitizer flags
set(project_desc "DirectX 12 graphics framework")
project(gfx-dep DESCRIPTION "${project_desc}" LANGUAGES CXX)
  fixSubmoduleOption(GFX_BUILD_EXAMPLES OFF)
  fixSubmoduleOption(GFX_ENABLE_GUI ON)
  add_subdirectory("${PROJECT_SOURCE_DIR}/gfx" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" EXCLUDE_FROM_ALL)
  mark_as_advanced(FORCE D3D12MA_AGILITY_SDK_DIRECTORY)
  mark_as_advanced(FORCE D3D12MA_AGILITY_SDK_PREVIEW)
  mark_as_advanced(FORCE D3D12MA_BUILD_SAMPLE)
  mark_as_advanced(FORCE D3D12_LIB)
  mark_as_advanced(FORCE DXGI_LIB)
  mark_as_advanced(FORCE GFX_ENABLE_GUI)
  mark_as_advanced(FORCE GFX_ENABLE_SCENE)
  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_D3D12MEMORYALLOCATOR)
  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_DIRECTX-DXC)
  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_DIRECTX12-AGILITY)
  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_WINPIXEVENT)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_D3D12MEMORYALLOCATOR)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_DIRECTX-DXC)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_DIRECTX12-AGILITY)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_WINPIXEVENT)

  #
  setCxxCompileFlags(gfx PRIVATE)
  setSanitizerFlags(gfx PRIVATE)

  #
  add_library(${PROJECT_NAME} ALIAS gfx)

  # Treat gfx public include directories as SYSTEM for dependents
  # to suppress third-party warnings when MINIDXNN_WARNING_EXTRA is enabled
  get_target_property(_gfx_includes gfx INTERFACE_INCLUDE_DIRECTORIES)
  if(_gfx_includes)
    target_include_directories(gfx SYSTEM INTERFACE ${_gfx_includes})
  endif()
