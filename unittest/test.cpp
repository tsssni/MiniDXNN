/*!
  \file test.cpp
  \author Sho Ikeda
  \brief LinearAlgebraMatrixTest fixture implementation for GPU compute shader tests
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#include "test.hpp"

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
// Standard C++ library
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
// GFX
#include "gfx.h"
// Example
#include "common/gfx_utility.hpp"

namespace test {

auto LinearAlgebraMatrixTest::initializeTest() noexcept -> void
{
  finalizeTest();

  m_gfxContext = ex::createGfxContext(params().m_enableDebugMode);
}

auto LinearAlgebraMatrixTest::finalizeTest() noexcept -> void
{
  m_gfxContext.reset();
}

} /* namespace test */
#endif // !MINIDXNN_CPP_FALLBACK_ONLY
