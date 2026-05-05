/*!
  \file texture_inference_with_encoding_common.hlsl
  \author Sho Ikeda
  \brief Shared inference step for texture MLP with input encoding (GPU + C++ fallback)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  Provides the common inference step (encode + forward pass) used by
  03_texture_compression_with_input_encoding.comp and the C++ fallback.
  Requires mlp.hlsl and input_encoding_common.hlsl to be included before this file.
*/

#ifndef MINIDXNN_TEXTURE_INFERENCE_WITH_ENCODING_COMMON_HLSL
#define MINIDXNN_TEXTURE_INFERENCE_WITH_ENCODING_COMMON_HLSL 1

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace texkernel {

/// Inference step with input encoding.
/// ENCODING_TYPE: 0=identity, 1=positional, 2=grid
/// NUM_FREQUENCIES: frequency bands for positional encoding
/// GRID_RESOLUTION: resolution for grid encoding
template <typename Type, uint NUM_LAYERS, int HIDDEN_DIM,
          dx::linalg::ComponentEnum ELEM_TYPE,
          dx::linalg::MatrixLayoutEnum LAYOUT,
          typename ActivationHiddenT, typename ActivationLastT,
          uint W_ALIGN, uint VS_ALIGN, uint B_ALIGN,
          bool HAS_BIAS,
          int INPUT_DIM, int OUTPUT_DIM,
          int ENCODING_TYPE, int NUM_FREQUENCIES, int GRID_RESOLUTION>
void inferenceStepWithEncoding(
    const uint threadId,
    ByteAddressBuffer uvBuffer,
    RWByteAddressBuffer outputBuffer,
    ByteAddressBuffer weightBuffer,
    ByteAddressBuffer biasBuffer,
    ByteAddressBuffer gridBuffer,
    const uint2 weightMatrixSize,
    const uint numTasks)
{
  using LayerDataRefWithBias = mininn::InferenceLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    ELEM_TYPE, LAYOUT, ELEM_TYPE, ELEM_TYPE,
    ActivationHiddenT, ActivationLastT, ELEM_TYPE,
    W_ALIGN, VS_ALIGN, B_ALIGN>;
  using LayerDataRefNoBias = mininn::InferenceLayerDataRefNoBias<
    NUM_LAYERS, HIDDEN_DIM,
    ELEM_TYPE, LAYOUT, ELEM_TYPE,
    ActivationHiddenT, ActivationLastT, ELEM_TYPE,
    W_ALIGN, VS_ALIGN, B_ALIGN>;

  if (numTasks <= threadId)
    return;

  using RawVecT = vector<Type, 2>;
  using InputVecT = vector<Type, INPUT_DIM>;
  using OutputVecT = vector<Type, OUTPUT_DIM>;
  using VecAccessor = mininn::impl::VectorBufferAccessor<Type>;

  const uint rawStride = 2u * (uint)(sizeof(Type));
  const uint outStride = (uint)OUTPUT_DIM * (uint)(sizeof(Type));

  const RawVecT uv = VecAccessor::template load<2>(uvBuffer, threadId * rawStride);

  // Encode input
  InputVecT encoded = (InputVecT)0;
  vector<float, 4> gridInterpWeights = (vector<float, 4>)0;
  vector<uint32_t, 4> gridCornerOffsets = (vector<uint32_t, 4>)0;

#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
  if (ENCODING_TYPE == 2) {
    encoded = inputenc::encodeInputGrid<Type, INPUT_DIM, GRID_RESOLUTION>(
        uv, gridBuffer, gridInterpWeights, gridCornerOffsets);
  } else {
    encoded = inputenc::encodeInput<Type, INPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES>(uv);
  }
#else
  if constexpr (ENCODING_TYPE == 2) {
    encoded = inputenc::encodeInputGrid<Type, INPUT_DIM, GRID_RESOLUTION>(
        uv, gridBuffer, gridInterpWeights, gridCornerOffsets);
  } else {
    encoded = inputenc::encodeInput<Type, INPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES>(uv);
  }
#endif

  // Forward pass
  OutputVecT output = (OutputVecT)0;

  if (HAS_BIAS) {
    LayerDataRefWithBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    layerData.setBiasData(biasBuffer);
    mininn::forward(output, encoded, layerData);
  } else {
    LayerDataRefNoBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    mininn::forward(output, encoded, layerData);
  }

  VecAccessor::template store<OUTPUT_DIM>(outputBuffer, threadId * outStride, output);
}

} // namespace texkernel

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif // MINIDXNN_TEXTURE_INFERENCE_WITH_ENCODING_COMMON_HLSL
