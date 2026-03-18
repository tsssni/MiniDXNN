#
# File: utility.cmake
# Author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

## Override submodule CMake options to prevent them from being exposed in cache
## This is useful when including third-party projects that define their own options
## which would otherwise clutter the project's CMake cache
## Args:
##   variable: The CMake variable name to set
##   value: The value to assign to the variable
## Usage:
##   fixSubmoduleOption(BUILD_TESTING OFF)
macro(fixSubmoduleOption variable value)
  set("${variable}" ${value} CACHE INTERNAL "" FORCE)
endmacro(fixSubmoduleOption)

## Verify that a CMake target exists
## Terminates configuration with an error if the target is not found
## Args:
##   target: The name of the target to check
## Usage:
##   checkTarget(mylib)
function(checkTarget target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Target '${target}' not found.")
  endif()
endfunction(checkTarget)

## Create symbolic link or copy file/directory as fallback
## On Unix systems, creates a symbolic link. On Windows, attempts symlink then falls back to copy
## Args:
##   source_path: Source file or directory path (must exist)
##   dest_path: Destination path for the symlink
## Note: On Windows, developer mode should be enabled for symlink creation without admin privileges
function(createSymlink source_path dest_path)
  file(CREATE_LINK "${source_path}" "${dest_path}" 
    RESULT link_result 
    COPY_ON_ERROR 
    SYMBOLIC)
  if(link_result)
    message(FATAL_ERROR "Failed to create symlink from ${source_path} to ${dest_path}: ${link_result}")
  endif()
endfunction(createSymlink)
