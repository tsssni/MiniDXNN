#
# file: project.cmake
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

## Configure MiniDXNN interface library
## Creates an INTERFACE target that links all dependencies
function(setMiniDXNNCore target)
  # Dependencies
  if(NOT TARGET Threads::Threads)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
  endif()

  add_library(${target} INTERFACE)

  #
  target_include_directories(${target} SYSTEM INTERFACE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../include)

  target_link_libraries(${target} INTERFACE Threads::Threads half-float-dep)
  if(MINIDXNN_BUILD_CPP_FALLBACK_ONLY)
    target_compile_definitions(${target} INTERFACE
      MINIDXNN_CPP_FALLBACK_ONLY=1
      #MINIDXNN_NO_INCLUDE_DX_LINALG=1
      #MINIDXNN_USE_SOFTWARE_LINALG_IMPL=1
      #MINIDXNN_CPP_FALLBACK_HALF_TYPE=half_float::half
    )
  else()
    target_link_libraries(${target} INTERFACE gfx-dep)
  endif()
endfunction(setMiniDXNNCore)
