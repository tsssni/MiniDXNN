/*!
  \file texture_training_common.hlsl
  \author Sho Ikeda
  \brief Shared training step for texture MLP (GPU + C++ fallback)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  Provides the common training step (forward + MSE loss + backward) used by both
  02_texture_training.comp and the C++ fallback training kernel.
  Requires mlp.hlsl to be included before this file.
*/

#ifndef MINIDXNN_TEXTURE_TRAINING_COMMON_HLSL
#define MINIDXNN_TEXTURE_TRAINING_COMMON_HLSL 1

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace texkernel {

template <typename Type, uint NUM_LAYERS, int HIDDEN_DIM,
          dx::linalg::ComponentEnum ELEM_TYPE,
          dx::linalg::MatrixLayoutEnum LAYOUT,
          typename ActivationHiddenT, typename ActivationLastT,
          uint W_ALIGN, uint VS_ALIGN, uint B_ALIGN>
void trainingStep(
    const uint threadId,
    ByteAddressBuffer uvBuffer,
    ByteAddressBuffer targetBuffer,
    ByteAddressBuffer weightBuffer,
    ByteAddressBuffer biasBuffer,
    RWByteAddressBuffer weightGradBuffer,
    RWByteAddressBuffer biasGradBuffer,
    RWByteAddressBuffer logitsCacheBuffer,
    RWByteAddressBuffer lossBuffer,
    const uint2 weightMatrixSize,
    const uint batchSize,
    const uint batchIndex,
    const uint currentBatchSize,
    const uint logitsStride)
{
  const int inputDim = 2;
  const int outputDim = 2;
  const uint inputVecStride = inputDim * (uint)(sizeof(Type));
  const uint outputVecStride = outputDim * (uint)(sizeof(Type));
  using InputVecT = vector<Type, inputDim>;
  using OutputVecT = vector<Type, outputDim>;
  using VectorBufferAccessorT = mininn::impl::VectorBufferAccessor<Type>;
  using LayerDataRef = mininn::TrainingLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    ELEM_TYPE, LAYOUT, ELEM_TYPE,
    ELEM_TYPE, ELEM_TYPE, ELEM_TYPE, ELEM_TYPE,
    ActivationHiddenT, ActivationLastT, ELEM_TYPE,
    W_ALIGN, VS_ALIGN, B_ALIGN>;

  if (currentBatchSize <= threadId)
    return;

  const uint sampleIndex = batchIndex * batchSize + threadId;
  const InputVecT uv = VectorBufferAccessorT::template load<inputDim>(uvBuffer, sampleIndex * inputVecStride);

  LayerDataRef layerData;
  layerData.setWeightData(weightBuffer, weightMatrixSize);
  layerData.setWeightGradientCache(weightGradBuffer, weightMatrixSize);
  if (LayerDataRef::HAS_BIAS) {
    layerData.setBiasData(biasBuffer);
    layerData.setBiasGradientCache(biasGradBuffer);
  }
  layerData.setLogitsCache(logitsCacheBuffer, threadId * logitsStride);

  OutputVecT output = (OutputVecT)0;
  mininn::forward(output, uv, layerData);

  const OutputVecT target = VectorBufferAccessorT::template load<outputDim>(targetBuffer, sampleIndex * outputVecStride);

  // MSE loss
  const OutputVecT diff = output - target;
  const float loss = (float)(dot(diff, diff)) / (float)(outputDim);
  mininn::impl::atomicFetchAdd(lossBuffer, 0, loss);

  // Loss gradient
  const Type scale = (Type)2 / (Type)((float)(outputDim));
  const Type batchScale = (Type)1 / (Type)((float)(currentBatchSize));
  const OutputVecT lossGrad = (scale * batchScale) * diff;
  mininn::backward(lossGrad, uv, layerData);
}

} // namespace texkernel

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif // MINIDXNN_TEXTURE_TRAINING_COMMON_HLSL
