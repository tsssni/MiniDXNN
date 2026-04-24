/*!
  \file mlp_test_common.hlsl
  \author Sho Ikeda
  \brief Shared MLP test kernel logic for inference and training (GPU + C++ fallback)
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  Provides the common MLP inference and training test functions used by both
  simple_mlp_inference_test.comp / simple_mlp_training_test.comp and the C++
  fallback kernel dispatch in unittest.cpp.
  Requires mlp.hlsl to be included before this file.
*/

#ifndef MINIDXNN_MLP_TEST_COMMON_HLSL
#define MINIDXNN_MLP_TEST_COMMON_HLSL 1

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace testkernel {

// ============================================================================
// Inference test step
// ============================================================================

template <typename Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM,
          dx::linalg::DataType ELEM_TYPE,
          dx::linalg::MatrixLayout LAYOUT,
          typename ActivationHiddenT, typename ActivationLastT,
          uint W_ALIGN, uint VS_ALIGN, uint B_ALIGN,
          bool HAS_BIAS>
void inferenceStep(
    const uint threadId,
    ByteAddressBuffer inputBuffer,
    RWByteAddressBuffer outputBuffer,
    ByteAddressBuffer weightBuffer,
    ByteAddressBuffer biasBuffer,
    const uint2 weightMatrixSize,
    const uint numTasks)
{
  const uint inputVecStride = INPUT_DIM * (uint)(sizeof(Type));
  const uint outputVecStride = OUTPUT_DIM * (uint)(sizeof(Type));
  using InputVecT = vector<Type, INPUT_DIM>;
  using OutputVecT = vector<Type, OUTPUT_DIM>;
  using VectorBufferAccessorT = mininn::impl::VectorBufferAccessor<Type>;

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

  const InputVecT input = VectorBufferAccessorT::template load<INPUT_DIM>(inputBuffer, threadId * inputVecStride);

  OutputVecT output = (OutputVecT)0;

  if (HAS_BIAS) {
    LayerDataRefWithBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    layerData.setBiasData(biasBuffer);
    mininn::forward(output, input, layerData);
  } else {
    LayerDataRefNoBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    mininn::forward(output, input, layerData);
  }

  VectorBufferAccessorT::template store<OUTPUT_DIM>(outputBuffer, threadId * outputVecStride, output);
}

// ============================================================================
// Training forward step
// ============================================================================

template <typename Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM,
          dx::linalg::DataType ELEM_TYPE,
          dx::linalg::MatrixLayout LAYOUT,
          typename ActivationHiddenT, typename ActivationLastT,
          uint W_ALIGN, uint VS_ALIGN, uint B_ALIGN,
          bool HAS_BIAS>
void trainingForwardStep(
    const uint threadId,
    ByteAddressBuffer inputBuffer,
    RWByteAddressBuffer outputBuffer,
    ByteAddressBuffer weightBuffer,
    ByteAddressBuffer biasBuffer,
    RWByteAddressBuffer logitsCacheBuffer,
    const uint2 weightMatrixSize,
    const uint batchSize,
    const uint logitsStride)
{
  const uint inputVecStride = INPUT_DIM * (uint)(sizeof(Type));
  const uint outputVecStride = OUTPUT_DIM * (uint)(sizeof(Type));
  using InputVecT = vector<Type, INPUT_DIM>;
  using OutputVecT = vector<Type, OUTPUT_DIM>;
  using VectorBufferAccessorT = mininn::impl::VectorBufferAccessor<Type>;

  using LayerDataRefWithBias = mininn::TrainingLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    ELEM_TYPE, LAYOUT, ELEM_TYPE,
    ELEM_TYPE, ELEM_TYPE, ELEM_TYPE, ELEM_TYPE,
    ActivationHiddenT, ActivationLastT, ELEM_TYPE,
    W_ALIGN, VS_ALIGN, B_ALIGN>;
  using LayerDataRefNoBias = mininn::TrainingLayerDataRefNoBias<
    NUM_LAYERS, HIDDEN_DIM,
    ELEM_TYPE, LAYOUT, ELEM_TYPE,
    ELEM_TYPE, ELEM_TYPE,
    ActivationHiddenT, ActivationLastT, ELEM_TYPE,
    W_ALIGN, VS_ALIGN, B_ALIGN>;

  if (batchSize <= threadId)
    return;

  const InputVecT input = VectorBufferAccessorT::template load<INPUT_DIM>(inputBuffer, threadId * inputVecStride);

  OutputVecT output = (OutputVecT)0;

  if (HAS_BIAS) {
    LayerDataRefWithBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    layerData.setBiasData(biasBuffer);
    layerData.setLogitsCache(logitsCacheBuffer, threadId * logitsStride);
    mininn::forward(output, input, layerData);
  } else {
    LayerDataRefNoBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    layerData.setLogitsCache(logitsCacheBuffer, threadId * logitsStride);
    mininn::forward(output, input, layerData);
  }

  VectorBufferAccessorT::template store<OUTPUT_DIM>(outputBuffer, threadId * outputVecStride, output);
}

// ============================================================================
// Training backward step
// ============================================================================

template <typename Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM,
          dx::linalg::DataType ELEM_TYPE,
          dx::linalg::MatrixLayout LAYOUT,
          typename ActivationHiddenT, typename ActivationLastT,
          uint W_ALIGN, uint VS_ALIGN, uint B_ALIGN,
          bool HAS_BIAS>
void trainingBackwardStep(
    const uint threadId,
    ByteAddressBuffer inputBuffer,
    ByteAddressBuffer targetBuffer,
    RWByteAddressBuffer outputBuffer,
    ByteAddressBuffer weightBuffer,
    ByteAddressBuffer biasBuffer,
    RWByteAddressBuffer weightGradBuffer,
    RWByteAddressBuffer biasGradBuffer,
    RWByteAddressBuffer logitsCacheBuffer,
    const uint2 weightMatrixSize,
    const uint batchSize,
    const uint logitsStride)
{
  const uint inputVecStride = INPUT_DIM * (uint)(sizeof(Type));
  const uint outputVecStride = OUTPUT_DIM * (uint)(sizeof(Type));
  using InputVecT = vector<Type, INPUT_DIM>;
  using OutputVecT = vector<Type, OUTPUT_DIM>;
  using VectorBufferAccessorT = mininn::impl::VectorBufferAccessor<Type>;

  using LayerDataRefWithBias = mininn::TrainingLayerDataRef<
    NUM_LAYERS, HIDDEN_DIM,
    ELEM_TYPE, LAYOUT, ELEM_TYPE,
    ELEM_TYPE, ELEM_TYPE, ELEM_TYPE, ELEM_TYPE,
    ActivationHiddenT, ActivationLastT, ELEM_TYPE,
    W_ALIGN, VS_ALIGN, B_ALIGN>;
  using LayerDataRefNoBias = mininn::TrainingLayerDataRefNoBias<
    NUM_LAYERS, HIDDEN_DIM,
    ELEM_TYPE, LAYOUT, ELEM_TYPE,
    ELEM_TYPE, ELEM_TYPE,
    ActivationHiddenT, ActivationLastT, ELEM_TYPE,
    W_ALIGN, VS_ALIGN, B_ALIGN>;

  if (batchSize <= threadId)
    return;

  const InputVecT input = VectorBufferAccessorT::template load<INPUT_DIM>(inputBuffer, threadId * inputVecStride);
  const OutputVecT target = VectorBufferAccessorT::template load<OUTPUT_DIM>(targetBuffer, threadId * outputVecStride);
  const OutputVecT output = VectorBufferAccessorT::template load<OUTPUT_DIM>(outputBuffer, threadId * outputVecStride);

  // L2 Loss gradient: 2/n * (output - target) / batchSize
  const Type batchScale = (Type)1 / (Type)((float)batchSize);
  const OutputVecT diff = output - target;
  const OutputVecT lossGrad = (Type)2 * diff * batchScale / (Type)((float)OUTPUT_DIM);

  if (HAS_BIAS) {
    LayerDataRefWithBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    layerData.setWeightGradientCache(weightGradBuffer, weightMatrixSize);
    layerData.setBiasData(biasBuffer);
    layerData.setBiasGradientCache(biasGradBuffer);
    layerData.setLogitsCache(logitsCacheBuffer, threadId * logitsStride);
    mininn::backward(lossGrad, input, layerData);
  } else {
    LayerDataRefNoBias layerData;
    layerData.setWeightData(weightBuffer, weightMatrixSize);
    layerData.setWeightGradientCache(weightGradBuffer, weightMatrixSize);
    layerData.setLogitsCache(logitsCacheBuffer, threadId * logitsStride);
    mininn::backward(lossGrad, input, layerData);
  }
}

} // namespace testkernel

#if defined(MINIDXNN_CPP_HLSL_COMPAT_HPP) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif // MINIDXNN_MLP_TEST_COMMON_HLSL
