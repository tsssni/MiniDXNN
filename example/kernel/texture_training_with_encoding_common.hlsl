/*!
  \file texture_training_with_encoding_common.hlsl
  \author Sho Ikeda
  \brief Shared training step for texture MLP with input encoding (GPU + C++ fallback)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  Provides the common training step (encode + forward + MSE loss + backward + grid scatter)
  used by 03_texture_compression_with_input_encoding.comp and the C++ fallback.
  Requires mlp.hlsl and input_encoding_common.hlsl to be included before this file.
*/

#ifndef MINIDXNN_TEXTURE_TRAINING_WITH_ENCODING_COMMON_HLSL
#define MINIDXNN_TEXTURE_TRAINING_WITH_ENCODING_COMMON_HLSL 1

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace texkernel {

/// Training step with input encoding.
/// ENCODING_TYPE: 0=identity, 1=positional, 2=grid
/// NUM_FREQUENCIES: frequency bands for positional encoding
/// GRID_RESOLUTION: resolution for grid encoding
template <typename Type, uint NUM_LAYERS, int HIDDEN_DIM,
          dx::linalg::ComponentEnum ELEM_TYPE,
          dx::linalg::MatrixLayoutEnum LAYOUT,
          typename ActivationHiddenT, typename ActivationLastT,
          uint W_ALIGN, uint VS_ALIGN, uint B_ALIGN,
          int INPUT_DIM, int OUTPUT_DIM,
          int ENCODING_TYPE, int NUM_FREQUENCIES, int GRID_RESOLUTION>
void trainingStepWithEncoding(
    const uint threadId,
    ByteAddressBuffer uvBuffer,
    ByteAddressBuffer targetBuffer,
    ByteAddressBuffer weightBuffer,
    ByteAddressBuffer biasBuffer,
    RWByteAddressBuffer weightGradBuffer,
    RWByteAddressBuffer biasGradBuffer,
    RWByteAddressBuffer logitsCacheBuffer,
    RWByteAddressBuffer lossBuffer,
    ByteAddressBuffer gridBuffer,
    RWByteAddressBuffer gridGradBuffer,
    const uint2 weightMatrixSize,
    const uint batchSize,
    const uint batchIndex,
    const uint currentBatchSize,
    const uint logitsStride,
    const float lossScale)
{
  using LayerDataRef = mininn::TrainingLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    ELEM_TYPE, LAYOUT, ELEM_TYPE,
    ELEM_TYPE, ELEM_TYPE, ELEM_TYPE, ELEM_TYPE,
    ActivationHiddenT, ActivationLastT, ELEM_TYPE,
    W_ALIGN, VS_ALIGN, B_ALIGN>;

  if (currentBatchSize <= threadId)
    return;

  using RawVecT = vector<Type, 2>;
  using InputVecT = vector<Type, INPUT_DIM>;
  using OutputVecT = vector<Type, OUTPUT_DIM>;
  using VecAccessor = mininn::impl::VectorBufferAccessor<Type>;

  const uint rawStride = 2u * (uint)(sizeof(Type));
  const uint outStride = (uint)OUTPUT_DIM * (uint)(sizeof(Type));
  const uint sampleIndex = batchIndex * batchSize + threadId;

  const RawVecT uv = VecAccessor::template load<2>(uvBuffer, sampleIndex * rawStride);

  // Encode input
  vector<float, 4> gridInterpWeights = (vector<float, 4>)0;
  vector<uint32_t, 4> gridCornerOffsets = (vector<uint32_t, 4>)0;
  InputVecT encoded = (InputVecT)0;

#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
  // HLSL path
  if (ENCODING_TYPE == 2) {
    encoded = inputenc::encodeInputGrid<Type, INPUT_DIM, GRID_RESOLUTION>(
        uv, gridBuffer, gridInterpWeights, gridCornerOffsets);
  } else {
    encoded = inputenc::encodeInput<Type, INPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES>(uv);
  }
#else
  // C++ fallback path
  if constexpr (ENCODING_TYPE == 2) {
    encoded = inputenc::encodeInputGrid<Type, INPUT_DIM, GRID_RESOLUTION>(
        uv, gridBuffer, gridInterpWeights, gridCornerOffsets);
  } else {
    encoded = inputenc::encodeInput<Type, INPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES>(uv);
  }
#endif

  // Set up layer data references
  LayerDataRef layerData;
  layerData.setWeightData(weightBuffer, weightMatrixSize);
  layerData.setWeightGradientCache(weightGradBuffer, weightMatrixSize);
  if (LayerDataRef::HAS_BIAS) {
    layerData.setBiasData(biasBuffer);
    layerData.setBiasGradientCache(biasGradBuffer);
  }
  layerData.setLogitsCache(logitsCacheBuffer, threadId * logitsStride);

  // Forward pass
  OutputVecT output = (OutputVecT)0;
  mininn::forward(output, encoded, layerData);

  // MSE loss
  const OutputVecT target = VecAccessor::template load<OUTPUT_DIM>(targetBuffer, sampleIndex * outStride);
  const OutputVecT diff = output - target;
  const float loss = (float)(dot(diff, diff)) / (float)(OUTPUT_DIM);
  mininn::impl::atomicFetchAdd(lossBuffer, 0, loss);

  // Loss gradient (scaled by lossScale for FP16 gradient stability)
  const Type scale = (Type)2 / (Type)((float)(OUTPUT_DIM));
  const Type batchScale = (Type)1 / (Type)((float)(currentBatchSize));
  const OutputVecT lossGrad = ((Type)lossScale * scale * batchScale) * diff;

  // Backward pass
#if !defined(MINIDXNN_CPP_HLSL_COMPAT_HPP)
  if (ENCODING_TYPE == 2) {
    const InputVecT inputGrad = mininn::backward(lossGrad, encoded, layerData);
    inputenc::scatterGridGradient<Type, INPUT_DIM>(inputGrad, gridInterpWeights, gridCornerOffsets, gridGradBuffer);
  } else {
    mininn::backward(lossGrad, encoded, layerData);
  }
#else
  if constexpr (ENCODING_TYPE == 2) {
    const InputVecT inputGrad = mininn::backward(lossGrad, encoded, layerData);
    inputenc::scatterGridGradient<Type, INPUT_DIM>(inputGrad, gridInterpWeights, gridCornerOffsets, gridGradBuffer);
  } else {
    mininn::backward(lossGrad, encoded, layerData);
  }
#endif
}

} // namespace texkernel

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif // MINIDXNN_TEXTURE_TRAINING_WITH_ENCODING_COMMON_HLSL
