/*!
  \file image.hpp
  \author Sho Ikeda
  \brief Image file I/O declarations (PPM format)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_EXAMPLE_IMAGE_HPP
#define MINIDXNN_EXAMPLE_IMAGE_HPP 1

// Standard C++ library
#include <ostream>

namespace ex {

// Forward declaration
template <typename, size_t> class Pixmap;


template <typename T, size_t Channel>
auto writeAsPpm(const Pixmap<T, Channel>& pixmap, std::ostream& output) noexcept -> void;

} /* namespace ex */

#endif /* MINIDXNN_EXAMPLE_IMAGE_HPP */
