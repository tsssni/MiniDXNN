#
# file: CMakeLists.txt
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.21)

# Include utility functions and unit test configurations
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/utility.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/project.cmake)

# Build unit test executable
set(project_desc "MiniDxNN unit tests")
project(unittest DESCRIPTION ${project_desc} LANGUAGES CXX)
  buildUnittest(unittest)
  checkTarget(unittest)
