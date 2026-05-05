#
# file: CMakeLists.txt
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

cmake_minimum_required(VERSION 3.21)

# stb_image interface library (header-only)
add_library(stb-image-dep INTERFACE)
target_include_directories(stb-image-dep SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
