#
# file: project.cmake
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

function(setExampleCommon target)
  # cmake dependencies
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cmake/compiler.cmake)

  # 
  add_library(${target} INTERFACE)
  # Add source code
  set(source_files ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/matrix.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/activation.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/mlp_layer.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/cpp_fallback.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/loss.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/loss.cpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/optimizer.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/optimizer.cpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/image.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/image.cpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/pixmap.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/texture.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/utility.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/xoshiro128plus.hpp
                   )
  if(NOT MINIDXNN_CPP_FALLBACK_ONLY)
    list(APPEND source_files
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/gfx_utility.hpp
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/common/gfx_utility.cpp
                   )
  endif()
  source_group(TREE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" PREFIX ${target} FILES ${source_files})
  target_sources(${target} INTERFACE ${source_files})
  # Add properties
  setCxxCompileFlags(${target} INTERFACE)
  setCxxWarningFlags(${target} INTERFACE)
  setSanitizerFlags(${target} INTERFACE)
  target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
  target_link_libraries(${target} INTERFACE minidxnn-core)
endfunction(setExampleCommon)


function(createHlslIncludeDirsHpp compute_shader_dir binary_dir output_dir)
  # Compute shader path
  cmake_path(SET mininn_compute_shader_dir NORMALIZE "${compute_shader_dir}")
  cmake_path(CONVERT "${mininn_compute_shader_dir}" TO_CMAKE_PATH_LIST mininn_compute_shader_dir)

  # DXC header path
  cmake_path(SET mininn_dxc_include_dir NORMALIZE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../third_party/gfx_dep/gfx/third_party/directx-dxc/inc/hlsl")
  cmake_path(CONVERT "${mininn_dxc_include_dir}" TO_CMAKE_PATH_LIST mininn_dxc_include_dir)

  # MLP header path
  cmake_path(SET mininn_mlp_include_dir NORMALIZE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../include/minidxnn/hlsl")
  cmake_path(CONVERT "${mininn_mlp_include_dir}" TO_CMAKE_PATH_LIST mininn_mlp_include_dir)

  #
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/hlsl_include_dirs.hpp.in"
                 "${output_dir}/hlsl_include_dirs.hpp"
                 @ONLY)
endfunction(createHlslIncludeDirsHpp)


function(buildExample target example_dir hlsl_include_hpp_dir)
  # include dependency cmake files
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cmake/utility.cmake)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cmake/compiler.cmake)

  # Create an executable
  source_group(TREE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" PREFIX ${target} FILES ${example_dir}/example.cpp)
  add_executable(${target} ${example_dir}/example.cpp)

  # Copy runtime DLLs to the binary directory (Windows only)
  # This ensures DirectX 12 runtime DLLs are available alongside the executable
  # Skip when building in C++ fallback mode (no GFX DLLs to copy)
  if(WIN32 AND NOT MINIDXNN_CPP_FALLBACK_ONLY)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_newer $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
      COMMAND_EXPAND_LISTS
      COMMENT "Copying runtime DLLs to output directory")
  endif()

  #
  target_link_libraries(${target} PRIVATE example-common cli11-dep)
  target_include_directories(${target} PRIVATE ${hlsl_include_hpp_dir})
endfunction(buildExample)
