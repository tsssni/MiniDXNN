/*!
  \file input_encoding_common.hlsl
  \author Sho Ikeda
  \brief Shared input encoding implementations (GPU + C++ fallback)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  Provides input encoding functions shared between the HLSL compute kernel
  (03_texture_compression_with_input_encoding.comp) and the C++ fallback path.

  Encoding types:
    0 = none (identity, raw UV passed directly)
    1 = positional encoding (sin/cos frequency bands)
    2 = grid encoding (learnable bilinear feature grid)

  Requires mlp.hlsl to be included before this file.
*/

#ifndef MINIDXNN_INPUT_ENCODING_COMMON_HLSL
#define MINIDXNN_INPUT_ENCODING_COMMON_HLSL 1

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace inputenc {

// ============================================================================
// Identity and positional encoding
// ============================================================================

/// Encode a raw (u,v) into identity (passthrough) or positional frequency encoding.
/// INPUT_DIM: output dimension (2 for identity, 4*numFrequencies for positional)
/// ENCODING_TYPE: 0=identity, 1=positional
/// NUM_FREQUENCIES: number of frequency bands (only used when ENCODING_TYPE==1)
template <typename Type, int INPUT_DIM, int ENCODING_TYPE, int NUM_FREQUENCIES>
vector<Type, INPUT_DIM> encodeInput(const vector<Type, 2> uv)
{
  vector<Type, INPUT_DIM> encoded = (vector<Type, INPUT_DIM>)0;
#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
  // HLSL path
  if (ENCODING_TYPE == 0) {
    encoded[0] = uv.x;
    encoded[1] = uv.y;
  } else if (ENCODING_TYPE == 1) {
    static const float PI = 3.14159265358979f;
    [unroll] for (int k = 0; k < NUM_FREQUENCIES; ++k) {
      const float freq = pow(2.0f, (float)k) * PI;
      encoded[k * 4 + 0] = (Type)sin(freq * (float)uv.x);
      encoded[k * 4 + 1] = (Type)cos(freq * (float)uv.x);
      encoded[k * 4 + 2] = (Type)sin(freq * (float)uv.y);
      encoded[k * 4 + 3] = (Type)cos(freq * (float)uv.y);
    }
  }
#else
  // C++ fallback path
  if constexpr (ENCODING_TYPE == 0) {
    encoded[0] = uv[0];
    encoded[1] = uv[1];
  } else if constexpr (ENCODING_TYPE == 1) {
    constexpr float PI = 3.14159265358979f;
    for (int k = 0; k < NUM_FREQUENCIES; ++k) {
      const float freq = std::pow(2.0f, static_cast<float>(k)) * PI;
      encoded[static_cast<size_t>(k * 4 + 0)] = static_cast<Type>(std::sin(freq * static_cast<float>(uv[0])));
      encoded[static_cast<size_t>(k * 4 + 1)] = static_cast<Type>(std::cos(freq * static_cast<float>(uv[0])));
      encoded[static_cast<size_t>(k * 4 + 2)] = static_cast<Type>(std::sin(freq * static_cast<float>(uv[1])));
      encoded[static_cast<size_t>(k * 4 + 3)] = static_cast<Type>(std::cos(freq * static_cast<float>(uv[1])));
    }
  }
#endif
  return encoded;
}

// ============================================================================
// Grid encoding
// ============================================================================

/// Bilinear interpolation of grid features.
/// Returns the encoded feature vector and outputs interpolation weights/offsets
/// for use during backpropagation gradient scattering.
template <typename Type, int INPUT_DIM, int GRID_RESOLUTION>
vector<Type, INPUT_DIM> encodeInputGrid(
    const vector<Type, 2> uv,
    ByteAddressBuffer gridBuffer,
#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
    out float4 interpWeights,
    out uint4 cornerOffsets)
#else
    vector<float, 4>& interpWeights,
    vector<uint32_t, 4>& cornerOffsets)
#endif
{
#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
  // HLSL path
  const float gx = clamp((float)uv.x, 0.0f, 1.0f) * (float)(GRID_RESOLUTION - 1);
  const float gy = clamp((float)uv.y, 0.0f, 1.0f) * (float)(GRID_RESOLUTION - 1);

  const uint ix = min((uint)gx, (uint)(GRID_RESOLUTION) - 2u);
  const uint iy = min((uint)gy, (uint)(GRID_RESOLUTION) - 2u);
#else
  // C++ fallback path
  const float gx = std::clamp(static_cast<float>(uv[0]), 0.0f, 1.0f) * static_cast<float>(GRID_RESOLUTION - 1);
  const float gy = std::clamp(static_cast<float>(uv[1]), 0.0f, 1.0f) * static_cast<float>(GRID_RESOLUTION - 1);

  const uint ix = std::min(static_cast<uint>(gx), static_cast<uint>(GRID_RESOLUTION) - 2u);
  const uint iy = std::min(static_cast<uint>(gy), static_cast<uint>(GRID_RESOLUTION) - 2u);
#endif

#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
  const float fx = gx - (float)(ix);
  const float fy = gy - (float)(iy);
#else
  const float fx = gx - static_cast<float>(ix);
  const float fy = gy - static_cast<float>(iy);
#endif

  interpWeights[0] = (1.0f - fx) * (1.0f - fy);
  interpWeights[1] = fx * (1.0f - fy);
  interpWeights[2] = (1.0f - fx) * fy;
  interpWeights[3] = fx * fy;

#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
  const uint featureBytes = (uint)(INPUT_DIM) * 4u;
  const uint R = (uint)(GRID_RESOLUTION);
#else
  const uint featureBytes = static_cast<uint>(INPUT_DIM) * 4u;
  const uint R = static_cast<uint>(GRID_RESOLUTION);
#endif
  cornerOffsets[0] = (iy * R + ix) * featureBytes;
  cornerOffsets[1] = (iy * R + (ix + 1u)) * featureBytes;
  cornerOffsets[2] = ((iy + 1u) * R + ix) * featureBytes;
  cornerOffsets[3] = ((iy + 1u) * R + (ix + 1u)) * featureBytes;

  using FloatAccessor = mininn::impl::VectorBufferAccessor<float>;
  using FloatVecT = vector<float, INPUT_DIM>;

  FloatVecT result = (FloatVecT)0;
  for (uint c = 0; c < 4; ++c) {
    const FloatVecT corner = FloatAccessor::template load<INPUT_DIM>(gridBuffer, cornerOffsets[c]);
    result = result + interpWeights[c] * corner;
  }

  return (vector<Type, INPUT_DIM>)result;
}

/// Scatter gradient back to grid feature corners during backpropagation.
template <typename Type, int INPUT_DIM>
void scatterGridGradient(
    const vector<Type, INPUT_DIM> inputGrad,
    const vector<float, 4> interpWeights,
    const vector<uint32_t, 4> cornerOffsets,
    RWByteAddressBuffer gridGradBuffer)
{
  for (uint c = 0; c < 4; ++c) {
#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
    for (uint f = 0; f < (uint)(INPUT_DIM); ++f) {
      const float grad = (float)(inputGrad[f]) * interpWeights[c];
#else
    for (uint f = 0; f < static_cast<uint>(INPUT_DIM); ++f) {
      const float grad = static_cast<float>(inputGrad[f]) * interpWeights[c];
#endif
      mininn::impl::atomicFetchAdd<float32_t>(gridGradBuffer, cornerOffsets[c] + f * 4u, grad);
    }
  }
}

} // namespace inputenc

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif // MINIDXNN_INPUT_ENCODING_COMMON_HLSL
