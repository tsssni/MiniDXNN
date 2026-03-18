/*!
  \file test.hpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_UNITTEST_TEST_HPP
#define MINIDXNN_UNITTEST_TEST_HPP 1

// Standard C++ library
#include <cassert>
#include <cstdint>
#include <memory>
// GoogleTest
#include "gtest/gtest.h"
// GFX
#include "gfx.h"
#include "gfx_window.h"
// Test
#include "common/mlp_layer.hpp"
#include "common/utility.hpp"

namespace test {

using ex::Arithmetic;

struct TestParameters
{
  //
  double m_similarityThreshold = 0.90;
  std::uint32_t m_seed = 987'654'321;
  std::uint32_t m_numThreadsX = 32;
  size_t m_numTasks = 1024;
  bool m_enableDebugMode = false;
  //
  ex::MatrixLayout m_weightMatrixLayout = ex::MatrixLayout::ROW_MAJOR;
};

class CoopVecTest : public ::testing::TestWithParam<TestParameters>
{
 public:
  template <typename ElemType>
  [[nodiscard]]
  auto calcMatrixStrideAndSize(const size_t rowSize, const size_t columnSize, const ex::MatrixLayout layout) -> std::tuple<size_t, size_t>;

  [[nodiscard]]
  auto context() noexcept -> GfxContext& {return *m_gfxContext;}

  [[nodiscard]]
  auto context() const noexcept -> const GfxContext& {return *m_gfxContext;}

  auto initializeTest() noexcept -> void;

  auto finalizeTest() noexcept -> void;

  [[nodiscard]]
  auto params() const noexcept -> const TestParameters& {return GetParam();}

 protected:
  void SetUp() override {initializeTest();}

  void TearDown() override {finalizeTest();}

 private:
  std::shared_ptr<GfxContext> m_gfxContext;
};

} /* namespace test */

#endif /* MINIDXNN_UNITTEST_TEST_HPP */
