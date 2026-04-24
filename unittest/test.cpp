/*!
  \file test.cpp
  \author Sho Ikeda
  \brief CoopVecTest fixture implementation for GPU compute shader tests
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#include "test.hpp"

#ifndef MINIDXNN_CPP_FALLBACK_ONLY
#include "common/gfx_utility.hpp"

namespace test {

auto CoopVecTest::initializeTest() noexcept -> void
{
  finalizeTest();

  m_gfxContext = ex::createGfxContext(params().m_enableDebugMode);
  // TODO. Check the Cooperative Vector features
}

auto CoopVecTest::finalizeTest() noexcept -> void
{
  // Delete the GFX context
  m_gfxContext.reset();
}

} /* namespace test */
#endif // !MINIDXNN_CPP_FALLBACK_ONLY
