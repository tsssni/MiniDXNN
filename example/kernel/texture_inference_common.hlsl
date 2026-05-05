/*!
  \file texture_inference_common.hlsl
  \author Sho Ikeda
  \brief Shared inference step for texture MLP (GPU + C++ fallback)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  Provides the common inference step (forward pass) used by both
  01_texture_inference.comp and the C++ fallback inference kernel.
  Requires mlp.hlsl to be included before this file.
*/

#ifndef MINIDXNN_TEXTURE_INFERENCE_COMMON_HLSL
#define MINIDXNN_TEXTURE_INFERENCE_COMMON_HLSL 1

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace texkernel {

template <typename Type, uint NUM_LAYERS, int HIDDEN_DIM,
          dx::linalg::ComponentEnum ELEM_TYPE,
          dx::linalg::MatrixLayoutEnum LAYOUT,
          typename ActivationHiddenT, typename ActivationLastT,
          uint W_ALIGN, uint VS_ALIGN, uint B_ALIGN,
          bool HAS_BIAS>
void inferenceStep(
    const uint threadId,
    ByteAddressBuffer uvBuffer,
    RWByteAddressBuffer outputBuffer,
    ByteAddressBuffer weightBuffer,
    ByteAddressBuffer biasBuffer,
    const uint2 weightMatrixSize,
    const uint numTasks)
{
  const int inputDim = 2;
  const int outputDim = 2;
  const uint inputVecStride = inputDim * (uint)(sizeof(Type));
  const uint outputVecStride = outputDim * (uint)(sizeof(Type));
  using InputVecT = vector<Type, inputDim>;
  using OutputVecT = vector<Type, outputDim>;
  using VectorBufferAccessorT = mininn::impl::VectorBufferAccessor<Type>;

  // Select LayerDataRef type based on HAS_BIAS
  // Note: Both with-bias and no-bias variants are always available,
  // but the HAS_BIAS template parameter selects which to use.
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

  const InputVecT uv = VectorBufferAccessorT::template load<inputDim>(uvBuffer, threadId * inputVecStride);

  OutputVecT output = (OutputVecT)0;

  if (HAS_BIAS) {
    LayerDataRefWithBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    layerData.setBiasData(biasBuffer);
    mininn::forward(output, uv, layerData);
  } else {
    LayerDataRefNoBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    mininn::forward(output, uv, layerData);
  }

  VectorBufferAccessorT::template store<outputDim>(outputBuffer, threadId * outputVecStride, output);
}

} // namespace texkernel

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif // MINIDXNN_TEXTURE_INFERENCE_COMMON_HLSL
