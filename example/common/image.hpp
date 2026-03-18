/*!
  \file image.hpp
  \author Sho Ikeda
  \brief No brief description
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_IMAGE_HPP
#define MINIDXNN_EXAMPLE_IMAGE_HPP 1

// Standard C++ library
#include <ostream>

namespace ex {

// Forward declaration
template <typename> class Pixmap;


template <typename T>
auto writeAsPpm(const Pixmap<T>& pixmap, std::ostream& output) noexcept -> void;

// Impl

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_IMAGE_HPP */
