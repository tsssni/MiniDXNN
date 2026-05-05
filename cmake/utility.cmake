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


## Copy runtime DLL/SO overrides from third_party/runtime to the target's output directory
## If third_party/runtime contains DLL or SO files, they overwrite the corresponding
## files already copied by TARGET_RUNTIME_DLLS, allowing custom driver/compiler builds.
## Args:
##   target: The executable target whose output directory receives the overrides
function(copyRuntimeOverrides target)
  cmake_path(SET runtime_dir "${CMAKE_SOURCE_DIR}/third_party/runtime")
  if(WIN32)
    file(GLOB runtime_files "${runtime_dir}/*.dll")
  else()
    file(GLOB runtime_files "${runtime_dir}/*.so" "${runtime_dir}/*.so.*")
  endif()
  if(runtime_files)
    foreach(runtime_file IN LISTS runtime_files)
      cmake_path(GET runtime_file FILENAME filename)
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${runtime_file}" "$<TARGET_FILE_DIR:${target}>/${filename}"
        COMMENT "Overriding ${filename} from third_party/runtime")
    endforeach()
  endif()
endfunction(copyRuntimeOverrides)
