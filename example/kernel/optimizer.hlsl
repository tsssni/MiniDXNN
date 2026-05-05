/*!
  \file optimizer.hlsl
  \author Sho Ikeda
  \brief Shared optimizer implementations for packed byte buffers (GPU + C++ fallback)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  Provides SGD, Adam, and Lion optimizers that operate directly on packed byte buffers.
  Requires mlp.hlsl to be included before this file.
*/

#ifndef MINIDXNN_OPTIMIZER_HLSL
#define MINIDXNN_OPTIMIZER_HLSL 1

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace optimizer {

// ============================================================================
// Element-level buffer access helpers
// ============================================================================

namespace detail {

#ifdef MINIDXNN_CPP_HLSL_COMPAT_HPP
// C++ fallback: direct memory access via memcpy

template <typename Type>
float loadParam(ByteAddressBuffer buf, uint byteOffset)
{
  return static_cast<float>(buf.Load<Type>(byteOffset));
}

template <typename Type>
float loadParam(RWByteAddressBuffer buf, uint byteOffset)
{
  return static_cast<float>(buf.Load<Type>(byteOffset));
}

template <typename Type>
void storeParam(RWByteAddressBuffer buf, uint byteOffset, float value)
{
  const Type v = static_cast<Type>(value);
  buf.Store<Type>(byteOffset, v);
}

inline float loadMoment(RWByteAddressBuffer buf, uint byteOffset)
{
  return buf.Load<float>(byteOffset);
}

inline void storeMoment(RWByteAddressBuffer buf, uint byteOffset, float value)
{
  buf.Store<float>(byteOffset, value);
}

inline float sqrtf(float v) { return std::sqrt(v); }

#else
// HLSL GPU: use asfloat / asuint for float32 buffer access.
// float16 accesses are 2-byte aligned; read the enclosing uint and extract.

template <typename Type>
float loadParam(ByteAddressBuffer buf, uint byteOffset)
{
  return (float)buf.Load<Type>(byteOffset);
}

template <typename Type>
float loadParam(RWByteAddressBuffer buf, uint byteOffset)
{
  return (float)buf.Load<Type>(byteOffset);
}

template <typename Type>
void storeParam(RWByteAddressBuffer buf, uint byteOffset, float value)
{
  buf.Store<Type>(byteOffset, (Type)value);
}

float loadMoment(RWByteAddressBuffer buf, uint byteOffset)
{
  return asfloat(buf.Load(byteOffset));
}

void storeMoment(RWByteAddressBuffer buf, uint byteOffset, float value)
{
  buf.Store(byteOffset, asuint(value));
}

float sqrtf(float v) { return sqrt(v); }

#endif

} // namespace detail

//! Clamp value to safe range for the target type to prevent FP16 overflow
template <typename Type>
float clampForType(float v)
{
  if ((uint)(sizeof(Type)) < (uint)(sizeof(float))) {
    const float maxVal = 65504.0f;
    v = (v > maxVal) ? maxVal : ((v < -maxVal) ? -maxVal : v);
  }
  return v;
}

// ============================================================================
// SGD optimizer — per-element update
// ============================================================================

template <typename Type>
void sgdElement(RWByteAddressBuffer params,
                RWByteAddressBuffer grads,
                const float learningRate,
                const float invLossScale,
                const uint elementIndex,
                const uint totalElements)
{
  if (elementIndex >= totalElements) return;
  const uint offset = elementIndex * (uint)(sizeof(Type));
  const float w = detail::loadParam<Type>(params, offset);
  const float g = detail::loadParam<Type>(grads, offset) * invLossScale;
  detail::storeParam<Type>(params, offset, w - learningRate * g);
}

// ============================================================================
// Adam optimizer — per-element update
// Moment buffers store float values (one float per parameter element).
// ============================================================================

template <typename Type>
void adamElement(RWByteAddressBuffer params,
                 RWByteAddressBuffer grads,
                 RWByteAddressBuffer firstMoment,
                 RWByteAddressBuffer secondMoment,
                 const float learningRate,
                 const float beta1,
                 const float beta2,
                 const float epsilon,
                 const float bc1,
                 const float bc2,
                 const float invLossScale,
                 const uint elementIndex,
                 const uint totalElements)
{
  if (elementIndex >= totalElements) return;
  const uint typeOffset = elementIndex * (uint)(sizeof(Type));
  const uint floatOffset = elementIndex * (uint)(sizeof(float));

  const float w = detail::loadParam<Type>(params, typeOffset);
  const float g = detail::loadParam<Type>(grads, typeOffset) * invLossScale;
  float m = detail::loadMoment(firstMoment, floatOffset);
  float v = detail::loadMoment(secondMoment, floatOffset);

  m = beta1 * m + (1.0f - beta1) * g;
  v = beta2 * v + (1.0f - beta2) * g * g;

  detail::storeMoment(firstMoment, floatOffset, m);
  detail::storeMoment(secondMoment, floatOffset, v);

  const float mHat = m / bc1;
  const float vHat = v / bc2;
  detail::storeParam<Type>(params, typeOffset,
      w - learningRate * mHat / (detail::sqrtf(vHat) + epsilon));
}

// ============================================================================
// Lion optimizer — per-element update
// Momentum buffer stores float values (one float per parameter element).
// ============================================================================

template <typename Type>
void lionElement(RWByteAddressBuffer params,
                 RWByteAddressBuffer grads,
                 RWByteAddressBuffer momentum,
                 const float learningRate,
                 const float beta1,
                 const float beta2,
                 const float weightDecay,
                 const float invLossScale,
                 const uint elementIndex,
                 const uint totalElements)
{
  if (elementIndex >= totalElements) return;
  const uint typeOffset = elementIndex * (uint)(sizeof(Type));
  const uint floatOffset = elementIndex * (uint)(sizeof(float));

  const float w = detail::loadParam<Type>(params, typeOffset);
  const float g = detail::loadParam<Type>(grads, typeOffset) * invLossScale;
  float m = detail::loadMoment(momentum, floatOffset);

  // When gradient is near-zero (converged), clear momentum to prevent
  // stale sign updates from accumulating and overflowing FP16 weights.
  if ((g > 0.0f ? g : -g) < 1e-8f) {
    detail::storeParam<Type>(params, typeOffset,
        clampForType<Type>(w - learningRate * weightDecay * w));
    detail::storeMoment(momentum, floatOffset, 0.0f);
    return;
  }

  const float interp = beta1 * m + (1.0f - beta1) * g;
  // Epsilon-thresholded sign: suppress updates from near-zero residual momentum after convergence
  const float signEps = 1e-6f;
  const float update = (interp > signEps) ? 1.0f : ((interp < -signEps) ? -1.0f : 0.0f);
  // Apply weight decay (Chen et al., 2023): θ = θ - η(c + λθ)
  detail::storeParam<Type>(params, typeOffset,
      clampForType<Type>(w - learningRate * (update + weightDecay * w)));

  m = beta2 * m + (1.0f - beta2) * g;
  detail::storeMoment(momentum, floatOffset, m);
}

// ============================================================================
// Convenience: update all elements in a buffer (for C++ fallback loops)
// ============================================================================

#ifdef MINIDXNN_CPP_HLSL_COMPAT_HPP

template <typename Type>
void sgdUpdateAll(RWByteAddressBuffer params,
                  RWByteAddressBuffer grads,
                  const float learningRate,
                  const float invLossScale,
                  const uint bufferSizeInBytes)
{
  const uint totalElements = bufferSizeInBytes / static_cast<uint>(sizeof(Type));
  for (uint i = 0; i < totalElements; ++i)
    sgdElement<Type>(params, grads, learningRate, invLossScale, i, totalElements);
}

template <typename Type>
void adamUpdateAll(RWByteAddressBuffer params,
                   RWByteAddressBuffer grads,
                   RWByteAddressBuffer firstMoment,
                   RWByteAddressBuffer secondMoment,
                   const float learningRate,
                   const float beta1,
                   const float beta2,
                   const float epsilon,
                   const float bc1,
                   const float bc2,
                   const float invLossScale,
                   const uint bufferSizeInBytes)
{
  const uint totalElements = bufferSizeInBytes / static_cast<uint>(sizeof(Type));
  for (uint i = 0; i < totalElements; ++i)
    adamElement<Type>(params, grads, firstMoment, secondMoment,
                      learningRate, beta1, beta2, epsilon, bc1, bc2, invLossScale, i, totalElements);
}

template <typename Type>
void lionUpdateAll(RWByteAddressBuffer params,
                   RWByteAddressBuffer grads,
                   RWByteAddressBuffer momentum,
                   const float learningRate,
                   const float beta1,
                   const float beta2,
                   const float weightDecay,
                   const float invLossScale,
                   const uint bufferSizeInBytes)
{
  const uint totalElements = bufferSizeInBytes / static_cast<uint>(sizeof(Type));
  for (uint i = 0; i < totalElements; ++i)
    lionElement<Type>(params, grads, momentum, learningRate, beta1, beta2, weightDecay, invLossScale, i, totalElements);
}

#endif // MINIDXNN_CPP_HLSL_COMPAT_HPP

} // namespace optimizer

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif // MINIDXNN_OPTIMIZER_HLSL
