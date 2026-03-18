#
# file: xoshiro128p.py
# author: Sho Ikeda
# brief: Xoshiro128+ pseudo-random number generator implementation
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

import numpy


class Xoshiro128Plus:
  """Xoshiro128+ pseudo-random number generator"""

  INT_DIGITS = 32
  FLOAT_DIGITS = 23
  DIGIT_OFFSET = INT_DIGITS - FLOAT_DIGITS

  def __init__(self, seed=None):
    """Initialize with a seed"""
    if seed is None:
      seed = 987654321

    # Initialize state using splitmix64-like algorithm
    self.m_state = numpy.zeros(4, dtype=numpy.uint32)
    z = numpy.uint32(seed)
    for i in range(4):
      z = numpy.uint32(z + 0x9e3779b9)
      z = numpy.uint32((z ^ (z >> 16)) * 0x85ebca6b)
      z = numpy.uint32((z ^ (z >> 13)) * 0xc2b2ae35)
      z = numpy.uint32(z ^ (z >> 16))
      self.m_state[i] = z

  def rotateLeft(self, x, k) -> numpy.uint32:
    """Rotate left"""
    return numpy.uint32((x << k) | (x >> (32 - k)))

  def next(self) -> numpy.uint32:
    """Generate next random uint32"""
    result = numpy.uint32(self.m_state[0] + self.m_state[3])

    t = numpy.uint32(self.m_state[1] << 9)

    self.m_state[2] ^= self.m_state[0]
    self.m_state[3] ^= self.m_state[1]
    self.m_state[1] ^= self.m_state[2]
    self.m_state[0] ^= self.m_state[3]

    self.m_state[2] ^= t
    self.m_state[3] = self.rotateLeft(self.m_state[3], 11)

    return result

  def mapToUniFloat(self, x: numpy.uint32) -> float:
    """Map a 32bit integer to a [0, 1) float"""
    u = (x >> self.DIGIT_OFFSET) / (1 << self.FLOAT_DIGITS)
    assert (0.0 <= u) and (u < 1.0), f'uni float mapping failed. {x} to {u}.'
    return u 

  def draw(self) -> float:
    """Generate a random float in [0, 1)"""
    x = self.next()
    u = self.mapToUniFloat(x)
    return u
