#
# file: project.cmake
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

## Configure MiniDxNN interface library
## Creates an INTERFACE target that links all dependencies
function(setMiniDxNNCore target)
  # Dependencies
  if(NOT TARGET Threads::Threads)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
  endif()

  add_library(${target} INTERFACE)

  #
  target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_link_libraries(${target} INTERFACE Threads::Threads half-float-dep gfx-dep)
endfunction(setMiniDxNNCore)
