/*!
  \file test.cpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#include "test.hpp"
// Direct3D
#include "d3d12.h"
// GFX
#include "gfx.h"
//
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
