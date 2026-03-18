#
# file: project.cmake
# author: Sho Ikeda
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

## Build unit test executable
## Compiles all test sources and configures test environment
function(buildUnittest target)
  # Include dependency cmake files
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cmake/utility.cmake)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cmake/compiler.cmake)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../example/project.cmake)

  # List source files explicitly for better IDE integration and rebuild detection
  set(source_files
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/unittest.cpp
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/test.cpp
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/test.hpp
  )

  source_group(TREE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" PREFIX ${target} FILES ${source_files})
  add_executable(${target} ${source_files})

  #
  if(NOT TARGET example-common)
    setExampleCommon(example-common)
  endif()
  target_link_libraries(${target} PRIVATE example-common cli11-dep gtest-dep)

  # Copy runtime DLLs to the binary directory (Windows only)
  # This ensures DirectX 12 runtime DLLs are available alongside the executable
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_newer $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
      COMMAND_EXPAND_LISTS
      COMMENT "Copying runtime DLLs to output directory")
  endif()

  # Link test kernel directory
  cmake_path(SET hlsl_include_hpp_dir "${PROJECT_BINARY_DIR}/include")
  createHlslIncludeDirsHpp("${PROJECT_SOURCE_DIR}/kernel" "${PROJECT_BINARY_DIR}" ${hlsl_include_hpp_dir})
  target_include_directories(${target} PRIVATE ${hlsl_include_hpp_dir})

  # Test discovery - automatically discovers and registers individual test cases
  include(GoogleTest)
  gtest_discover_tests(${target}
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    PROPERTIES ENVIRONMENT "GTEST_COLOR=1"
  )
  if(MINIDXNN_TEST_ENABLE_FP32_TESTS)
    target_compile_definitions(${target} PRIVATE MINIDXNN_TEST_ENABLE_FP32_TESTS=1)
  endif()
endfunction()
