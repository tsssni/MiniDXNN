/*!
  \file optimizer.cpp
  \author Sho Ikeda
  \brief Optimizer non-template implementations
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#include "optimizer.hpp"

// Standard C++ library
#include <string_view>

namespace ex {

auto getOptimizerTypeFromString(const std::string_view name) noexcept -> OptimizerType
{
  using namespace std::string_view_literals;
  const OptimizerType type = (name == "sgd"sv)  ? OptimizerType::SGD :
                             (name == "adam"sv) ? OptimizerType::ADAM :
                             (name == "lion"sv) ? OptimizerType::LION
                                               : OptimizerType::SGD;
  return type;
}

auto getOptimizerTypeString(const OptimizerType type) noexcept -> std::string_view
{
  using namespace std::string_view_literals;
  const std::string_view result = (type == OptimizerType::SGD)  ? "sgd"sv :
                                  (type == OptimizerType::ADAM) ? "adam"sv :
                                  (type == OptimizerType::LION) ? "lion"sv
                                                                : "sgd"sv;
  return result;
}

} /* namespace ex */
