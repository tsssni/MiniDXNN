/*!
  \file mlp.hlsl
  \author Sho Ikeda
  \brief Multi-Layer Perceptron (MLP) implementation in HLSL
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

#ifndef MINIDXNN_MLP_HLSL
#define MINIDXNN_MLP_HLSL

// options
// #define MINIDXNN_NO_INCLUDE_DX_LINALG
// #define MINIDXNN_USE_SOFTWARE_LINALG_IMPL

// DX
#ifndef MINIDXNN_NO_INCLUDE_DX_LINALG
#include <dx/linalg.h>
#endif // MINIDXNN_NO_INCLUDE_DX_LINALG

namespace mininn {

// Activation functions

struct IdentityActivation;
struct SigmoidActivation;
struct ReluActivation;
struct LeakyReluActivation;

// MLP

template <bool IS_MATRIX_TRANSPOSED> struct WeightData;
template <bool IS_BIAS_ENABLED> struct BiasData;

template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          bool __HAS_BIAS,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
struct LayerDataRefImpl
{
  static const bool HAS_BIAS = __HAS_BIAS;


  void setWeightData(WeightBufferT buffer, const uint startOffset = 0)
  {
    m_weight.set(buffer, startOffset);
  }

  void setBiasData(BiasBufferT buffer, const uint startOffset = 0)
  {
    m_bias.set(buffer, startOffset);
  }


  typename WeightData<IS_WEIGHT_MATRIX_TRANSPOSED>::template BufferRef<WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT> m_weight;
  typename BiasData<HAS_BIAS>::template BufferRef<BiasBufferT, BIAS_ELEM_TYPE, BIAS_ALIGNMENT> m_bias;
  ActivationHiddenT m_activationHidden;
  ActivationLastT m_activationLast;
};


// Helper types

template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          bool HAS_BIAS,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
using InferenceLayerDataRefImpl = LayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, HAS_BIAS, BiasBufferT, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT>;

template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
using InferenceLayerDataRef = InferenceLayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, ByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, true, ByteAddressBuffer, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT>;

template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
using InferenceLayerDataRefNoBias = InferenceLayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, ByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, false, ByteAddressBuffer, WEIGHT_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT>;

template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
using RWInferenceLayerDataRef = InferenceLayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, RWByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, true, RWByteAddressBuffer, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT>;

template <uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED = false,
          uint WEIGHT_ALIGNMENT = 128,
          uint WEIGHT_STRIDE_ALIGNMENT = 16,
          uint BIAS_ALIGNMENT = 64>
using RWInferenceLayerDataRefNoBias = InferenceLayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, RWByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, false, RWByteAddressBuffer, WEIGHT_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT>;


//

template <typename OutputElemT,
          int OUTPUT_DIM,
          typename InputElemT,
          int INPUT_DIM,
          uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          bool HAS_BIAS,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
          typename ActivationHiddenT,
          typename ActivationLastT,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED,
          uint WEIGHT_ALIGNMENT,
          uint WEIGHT_STRIDE_ALIGNMENT,
          uint BIAS_ALIGNMENT>
void forward(out vector<OutputElemT, OUTPUT_DIM> output,
             const vector<InputElemT, INPUT_DIM> input,
             const LayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, HAS_BIAS, BiasBufferT, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT> layerData);


// MLP Implementations

namespace impl {

template <typename T> struct TypeTraits {};

#define __MINIDXNN_DEFINE_TYPE_MAPPING(type, value, numElements) \
  template <> struct TypeTraits< type > \
  { \
    static const dx::linalg::DataType COMPONENT_TYPE = ( value ); \
    static const uint NUM_ELEMENTS = ( numElements ); \
  }

__MINIDXNN_DEFINE_TYPE_MAPPING(int16_t, dx::linalg::DATA_TYPE_SINT16, 1);
__MINIDXNN_DEFINE_TYPE_MAPPING(uint16_t, dx::linalg::DATA_TYPE_UINT16, 1);
__MINIDXNN_DEFINE_TYPE_MAPPING(int32_t, dx::linalg::DATA_TYPE_SINT32, 1);
__MINIDXNN_DEFINE_TYPE_MAPPING(uint32_t, dx::linalg::DATA_TYPE_UINT32, 1);
__MINIDXNN_DEFINE_TYPE_MAPPING(float16_t, dx::linalg::DATA_TYPE_FLOAT16, 1);
__MINIDXNN_DEFINE_TYPE_MAPPING(float32_t, dx::linalg::DATA_TYPE_FLOAT32, 1);
__MINIDXNN_DEFINE_TYPE_MAPPING(int8_t4_packed, dx::linalg::DATA_TYPE_SINT8_T4_PACKED, 4);
__MINIDXNN_DEFINE_TYPE_MAPPING(uint8_t4_packed, dx::linalg::DATA_TYPE_UINT8_T4_PACKED, 4);

template <dx::linalg::DataType T> struct ComponentTypeTraits {};

#define __MINIDXNN_DEFINE_COMPONENT_MAPPING(component, type, numElements) \
  template <> struct ComponentTypeTraits< component > \
  { \
    using Type = type; \
    static const uint SIZE = sizeof(Type); \
    static const uint NUM_ELEMENTS = ( numElements ); \
  }

__MINIDXNN_DEFINE_COMPONENT_MAPPING(dx::linalg::DATA_TYPE_SINT16, int16_t, 1);
__MINIDXNN_DEFINE_COMPONENT_MAPPING(dx::linalg::DATA_TYPE_UINT16, uint16_t, 1);
__MINIDXNN_DEFINE_COMPONENT_MAPPING(dx::linalg::DATA_TYPE_SINT32, int32_t, 1);
__MINIDXNN_DEFINE_COMPONENT_MAPPING(dx::linalg::DATA_TYPE_UINT32, uint32_t, 1);
__MINIDXNN_DEFINE_COMPONENT_MAPPING(dx::linalg::DATA_TYPE_FLOAT16, float16_t, 1);
__MINIDXNN_DEFINE_COMPONENT_MAPPING(dx::linalg::DATA_TYPE_FLOAT32, float32_t, 1);
__MINIDXNN_DEFINE_COMPONENT_MAPPING(dx::linalg::DATA_TYPE_SINT8_T4_PACKED, int8_t4_packed, 4);
__MINIDXNN_DEFINE_COMPONENT_MAPPING(dx::linalg::DATA_TYPE_UINT8_T4_PACKED, uint8_t4_packed, 4);

template <typename T>
struct BufferManip
{
  using Type = T;

  template <int N>
  static vector<Type, N> loadVec(ByteAddressBuffer buffer, const uint offset)
  {
    using VecT = vector<Type, N>;
    return buffer.template Load<VecT>(offset);
  }

  template <int N>
  static vector<Type, N> loadVec(RWByteAddressBuffer buffer, const uint offset)
  {
    using VecT = vector<Type, N>;
    return buffer.template Load<VecT>(offset);
  }

  template <int N>
  static void storeVec(RWByteAddressBuffer buffer, const uint offset, const vector<Type, N> value)
  {
    using VecT = vector<Type, N>;
    buffer.template Store<VecT>(offset, value);
  }
};

// https://therealmjp.github.io/posts/shader-fp16/
template <>
struct BufferManip<float16_t>
{
  using Type = float16_t;

  template <int N>
  static vector<Type, N> loadVec(ByteAddressBuffer buffer, const uint offset)
  {
    return __loadVecImpl<N>(buffer, offset);
  }

  template <int N>
  static vector<Type, N> loadVec(RWByteAddressBuffer buffer, const uint offset)
  {
    return __loadVecImpl<N>(buffer, offset);
  }

  template <int N>
  static void storeVec(RWByteAddressBuffer buffer, const uint offset, const vector<Type, N> value)
  {
    __storeVecImpl<N>(buffer, offset, value);
  }

// private:
  template <int N, typename BufferT>
  static vector<Type, N> __loadVecImpl(BufferT buffer, const uint offset)
  {
    const bool isLastOdd = (N & 0x01) == 0x01;
    const int NS = (N + 1) / 2;
    using VecT = vector<half, N>;
    using StagingVecT = vector<uint32_t, NS>;

    VecT output;
    StagingVecT staging = buffer.template Load<StagingVecT>(offset);
    [unroll]
    for (int i = 0; i < (N / 2); ++i) {
      output[2 * i    ] = asfloat16((uint16_t)(staging[i]));
      output[2 * i + 1] = asfloat16((uint16_t)(staging[i] >> 16));
    }
    if (isLastOdd) {
      output[N - 1] = asfloat16((uint16_t)staging[NS - 1]);
    }
    return output;
  }

  template <int N, typename BufferT>
  static void __storeVecImpl(BufferT buffer, const uint offset, const vector<Type, N> value)
  {
    const bool isLastOdd = (N & 0x01) == 0x01;
    const int NS = (N + 1) / 2;
    using VecT = vector<half, N>;
    using StagingVecT = vector<uint32_t, NS>;

    StagingVecT staging = (StagingVecT)0;
    [unroll]
    for (int i = 0; i < (N / 2); ++i) {
      staging[i] = (uint32_t)asuint16(value[2 * i]) | ((uint32_t)asuint16(value[2 * i + 1]) << 16);
    }
    if (isLastOdd) {
      staging[NS - 1] = (uint32_t)asuint16(value[N - 1]);
    }
    buffer.template Store<StagingVecT>(offset, staging);
  }
};

uint align(const uint sizeInBytes, const uint alignmentInBytes)
{
  const uint a = alignmentInBytes - 1;
  return (sizeInBytes + a) & (~a);
}

template <uint ROW_SIZE,
          uint COLUMN_SIZE,
          dx::linalg::MatrixLayout MATRIX_LAYOUT,
          bool IS_MATRIX_TRANSPOSED>
uint getMatrixMajorSize()
{
  const int isColumnMajor = (MATRIX_LAYOUT == dx::linalg::MATRIX_LAYOUT_COLUMN_MAJOR) ? 1 : 0;
  const int isTransposed = IS_MATRIX_TRANSPOSED ? 1 : 0;
  const uint size = (isColumnMajor ^ isTransposed) ? COLUMN_SIZE : ROW_SIZE;
  return size;
}

template <uint ROW_SIZE,
          uint COLUMN_SIZE,
          dx::linalg::MatrixLayout MATRIX_LAYOUT,
          bool IS_MATRIX_TRANSPOSED>
uint getMatrixMinorSize()
{
  const int isColumnMajor = (MATRIX_LAYOUT == dx::linalg::MATRIX_LAYOUT_COLUMN_MAJOR) ? 1 : 0;
  const int isTransposed = IS_MATRIX_TRANSPOSED ? 1 : 0;
  const uint size = (isColumnMajor ^ isTransposed) ? ROW_SIZE : COLUMN_SIZE;
  return size;
}

template <int ROW_SIZE,
          int COLUMN_SIZE,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          uint WEIGHT_STRIDE_ALIGNMENT>
uint getWeightMatrixStride()
{
  using ElemTypeTraits = ComponentTypeTraits<WEIGHT_ELEM_TYPE>;

  const uint minorSize = getMatrixMinorSize<ROW_SIZE, COLUMN_SIZE, WEIGHT_MATRIX_LAYOUT, false>();
  const uint stride = align(minorSize * ElemTypeTraits::SIZE, WEIGHT_STRIDE_ALIGNMENT);
  return stride;
}

template <int ROW_SIZE,
          int COLUMN_SIZE,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          uint WEIGHT_ALIGNMENT,
          uint WEIGHT_STRIDE_ALIGNMENT>
uint getWeightMatrixSize()
{
  const uint majorSize = getMatrixMajorSize<ROW_SIZE, COLUMN_SIZE, WEIGHT_MATRIX_LAYOUT, false>();
  const uint stride = getWeightMatrixStride<ROW_SIZE, COLUMN_SIZE, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_STRIDE_ALIGNMENT>();
  const uint size = align(majorSize * stride, WEIGHT_ALIGNMENT);
  return size;
}

template <int DIM,
          dx::linalg::DataType BIAS_ELEM_TYPE,
          uint BIAS_ALIGNMENT>
uint getBiasVectorSize()
{
  using ElemTypeTraits = ComponentTypeTraits<BIAS_ELEM_TYPE>;

  const uint size = align(DIM * ElemTypeTraits::SIZE, BIAS_ALIGNMENT);
  return size;
}

struct LinearAlgebra
{
  template <typename OutputElemT,
            typename InputElemT,
            int INPUT_ELEM_COUNT,
            typename MatrixBufferT,
            dx::linalg::DataType INPUT_ELEM_T,
            dx::linalg::DataType MATRIX_ELEM_T,
            uint ROW_SIZE,
            uint COLUMN_SIZE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            bool IS_MATRIX_TRANSPOSED>
  static
  vector<OutputElemT, ROW_SIZE>
  mul(const dx::linalg::MatrixRefImpl<MatrixBufferT, MATRIX_ELEM_T, ROW_SIZE, COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED> matrix,
      const dx::linalg::InterpretedVector<InputElemT, INPUT_ELEM_COUNT, INPUT_ELEM_T> input)
  {
    using InputVecT = vector<InputElemT, INPUT_ELEM_COUNT>;
    using OutputVecT = vector<OutputElemT, ROW_SIZE>;
    using MatrixElemT = typename ComponentTypeTraits<MATRIX_ELEM_T>::Type;
    using RowVecT = vector<MatrixElemT, COLUMN_SIZE>;
    using RowAccumVecT = vector<OutputElemT, COLUMN_SIZE>;
    using BufferManipT = BufferManip<MatrixElemT>;

    OutputVecT output = (OutputVecT)0;
#if defined(MINIDXNN_USE_SOFTWARE_LINALG_IMPL) && (MINIDXNN_USE_SOFTWARE_LINALG_IMPL != 0)
    //! \todo Support WEIGHT_MATRIX_LAYOUT and IS_WEIGHT_MATRIX_TRANSPOSED
    [unroll]
    for (uint row = 0, matrixOffset = matrix.StartOffset; row < ROW_SIZE; ++row) {
      const RowVecT rowVec = BufferManipT::template loadVec<COLUMN_SIZE>(matrix.Buffer, matrixOffset);
      output[row] = dot((RowAccumVecT)rowVec, (RowAccumVecT)input.Data);
      matrixOffset += matrix.Stride;
    }
#else // MINIDXNN_USE_SOFTWARE_LINALG_IMPL
    output = dx::linalg::Mul<OutputElemT>(matrix, input);
#endif // MINIDXNN_USE_SOFTWARE_LINALG_IMPL
    return output;
  }

  template <typename OutputElemT,
            typename InputElemT,
            int INPUT_ELEM_COUNT,
            typename MatrixBufferT,
            dx::linalg::DataType INPUT_ELEM_T,
            dx::linalg::DataType MATRIX_ELEM_T,
            uint ROW_SIZE,
            uint COLUMN_SIZE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            bool IS_MATRIX_TRANSPOSED,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_T>
  static
  vector<OutputElemT, ROW_SIZE>
  mulAdd(const dx::linalg::MatrixRefImpl<MatrixBufferT, MATRIX_ELEM_T, ROW_SIZE, COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED> matrix,
         const dx::linalg::InterpretedVector<InputElemT, INPUT_ELEM_COUNT, INPUT_ELEM_T> input,
         const dx::linalg::VectorRefImpl<BiasBufferT, BIAS_ELEM_T> bias)
  {
    using OutputVecT = vector<OutputElemT, ROW_SIZE>;
    using BiasElemT = typename ComponentTypeTraits<BIAS_ELEM_T>::Type;
    using BiasVecT = vector<BiasElemT, ROW_SIZE>;
    using BufferManipT = BufferManip<BiasElemT>;

    OutputVecT output = (OutputVecT)0;
#if defined(MINIDXNN_USE_SOFTWARE_LINALG_IMPL) && (MINIDXNN_USE_SOFTWARE_LINALG_IMPL != 0)
    output = mul<OutputElemT>(matrix, input);
    const BiasVecT biasVec = BufferManipT::template loadVec<ROW_SIZE>(bias.Buffer, bias.StartOffset);
    output += (OutputVecT)biasVec;
#else // MINIDXNN_USE_SOFTWARE_LINALG_IMPL
    output = dx::linalg::MulAdd<OutputElemT>(matrix, input, bias);
#endif // MINIDXNN_USE_SOFTWARE_LINALG_IMPL
    return output;
  }
};

template <bool HAS_BIAS> struct LayerManip;

template <>
struct LayerManip<true>
{
  static const bool HAS_BIAS = true;

  template <typename OutputElemT,
            int OUTPUT_DIM,
            typename InputElemT,
            int INPUT_DIM,
            uint NUM_HIDDEN_LAYERS,
            int HIDDEN_LAYER_DIM,
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            bool IS_WEIGHT_MATRIX_TRANSPOSED,
            uint WEIGHT_ALIGNMENT,
            uint WEIGHT_STRIDE_ALIGNMENT,
            uint BIAS_ALIGNMENT>
  static void forward(out vector<OutputElemT, OUTPUT_DIM> output,
               const vector<InputElemT, INPUT_DIM> input,
               const LayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, HAS_BIAS, BiasBufferT, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT> layerData,
               const uint weightExtraOffset,
               const uint biasExtraOffset)
  {
    using InputTypeTraits = TypeTraits<InputElemT>;

    output = LinearAlgebra::mulAdd<OutputElemT>(
        layerData.m_weight.template makeDxMatrixRef<OUTPUT_DIM, INPUT_DIM>(weightExtraOffset),
        dx::linalg::MakeInterpretedVector<InputTypeTraits::COMPONENT_TYPE>(input),
        layerData.m_bias.makeDxVectorRef(biasExtraOffset));
  }
};

template <>
struct LayerManip<false>
{
  static const bool HAS_BIAS = false;

  template <typename OutputElemT,
            int OUTPUT_DIM,
            typename InputElemT,
            int INPUT_DIM,
            uint NUM_HIDDEN_LAYERS,
            int HIDDEN_LAYER_DIM,
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            bool IS_WEIGHT_MATRIX_TRANSPOSED,
            uint WEIGHT_ALIGNMENT,
            uint WEIGHT_STRIDE_ALIGNMENT,
            uint BIAS_ALIGNMENT>
  static void forward(out vector<OutputElemT, OUTPUT_DIM> output,
                      const vector<InputElemT, INPUT_DIM> input,
                      const LayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, HAS_BIAS, BiasBufferT, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT> layerData,
                      const uint weightExtraOffset,
                      const uint biasExtraOffset)
  {
    using InputTypeTraits = TypeTraits<InputElemT>;

    output = LinearAlgebra::mul<OutputElemT>(
        layerData.m_weight.template makeDxMatrixRef<OUTPUT_DIM, INPUT_DIM>(weightExtraOffset),
        dx::linalg::MakeInterpretedVector<InputTypeTraits::COMPONENT_TYPE>(input));
  }
};

template <uint NUM_HIDDEN_LAYERS>
struct Mlp
{
  template <typename OutputElemT,
            int OUTPUT_DIM,
            typename InputElemT,
            int INPUT_DIM,
            int HIDDEN_LAYER_DIM,
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            bool HAS_BIAS,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            bool IS_WEIGHT_MATRIX_TRANSPOSED,
            uint WEIGHT_ALIGNMENT,
            uint WEIGHT_STRIDE_ALIGNMENT,
            uint BIAS_ALIGNMENT>
  static void forward(out vector<OutputElemT, OUTPUT_DIM> output,
                      const vector<InputElemT, INPUT_DIM> input,
                      const LayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, HAS_BIAS, BiasBufferT, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT> layerData)
  {
    using AccumElemT = typename ComponentTypeTraits<ACCUMULATOR_ELEM_TYPE>::Type;
    using AccumVecT = vector<AccumElemT, HIDDEN_LAYER_DIM>;
    using ActElemT = typename ComponentTypeTraits<ACTIVATION_ELEM_TYPE>::Type;
    using ActVecT = vector<ActElemT, HIDDEN_LAYER_DIM>;
    using LayerManipT = LayerManip<HAS_BIAS>;

    AccumVecT layerOut;
    ActVecT actOut;
    uint weightExtraOffset = 0;
    uint biasExtraOffset = 0;

    // First layer
    {
      LayerManipT::forward(layerOut, input, layerData, weightExtraOffset, biasExtraOffset);
      layerData.m_activationHidden.forward(actOut, (ActVecT)layerOut);
      weightExtraOffset += getWeightMatrixSize<HIDDEN_LAYER_DIM, INPUT_DIM, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT>();
      biasExtraOffset += getBiasVectorSize<HIDDEN_LAYER_DIM, BIAS_ELEM_TYPE, BIAS_ALIGNMENT>();
    }
    // Hidden layers
    [unroll]
    for (int depth = 1; depth < NUM_HIDDEN_LAYERS; ++depth) {
      const ActVecT layerInput = actOut;
      LayerManipT::forward(layerOut, layerInput, layerData, weightExtraOffset, biasExtraOffset);
      layerData.m_activationHidden.forward(actOut, (ActVecT)layerOut);
      weightExtraOffset += getWeightMatrixSize<HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT>();
      biasExtraOffset += getBiasVectorSize<HIDDEN_LAYER_DIM, BIAS_ELEM_TYPE, BIAS_ALIGNMENT>();
    }
    // Last layer
    {
      using AccumVecOutT = vector<AccumElemT, OUTPUT_DIM>;
      using ActVecOutT = vector<ActElemT, OUTPUT_DIM>;

      const ActVecT layerInput = actOut;
      AccumVecOutT lastLayerOut;
      LayerManipT::forward(lastLayerOut, layerInput, layerData, weightExtraOffset, biasExtraOffset);
      layerData.m_activationLast.forward(output, (ActVecOutT)lastLayerOut);
    }
  }
};

template <>
struct Mlp<0>
{
  static const uint NUM_HIDDEN_LAYERS = 0;

  template <typename OutputElemT,
            int OUTPUT_DIM,
            typename InputElemT,
            int INPUT_DIM,
            int HIDDEN_LAYER_DIM,
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            bool HAS_BIAS,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            bool IS_WEIGHT_MATRIX_TRANSPOSED,
            uint WEIGHT_ALIGNMENT,
            uint WEIGHT_STRIDE_ALIGNMENT,
            uint BIAS_ALIGNMENT>
  static void forward(out vector<OutputElemT, OUTPUT_DIM> output,
                      const vector<InputElemT, INPUT_DIM> input,
                      const LayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, HAS_BIAS, BiasBufferT, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT> layerData)
  {
    using AccumElemT = typename ComponentTypeTraits<ACCUMULATOR_ELEM_TYPE>::Type;
    using AccumVecT = vector<AccumElemT, OUTPUT_DIM>;
    using ActElemT = typename ComponentTypeTraits<ACTIVATION_ELEM_TYPE>::Type;
    using ActVecT = vector<ActElemT, OUTPUT_DIM>;
    using LayerManipT = LayerManip<HAS_BIAS>;

    AccumVecT layerOut;
    const uint weightExtraOffset = 0;
    const uint biasExtraOffset = 0;
    LayerManipT::forward(layerOut, input, layerData, weightExtraOffset, biasExtraOffset);
    layerData.m_activationLast.forward(output, (ActVecT)layerOut);
  }
};

} // impl

// Activation function implementation

template <bool IS_MATRIX_TRANSPOSED>
struct WeightData
{
  template <typename BufferT,
            dx::linalg::DataType ELEM_TYPE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            uint ALIGNMENT,
            uint STRIDE_ALIGNMENT>
  struct BufferRef
  {
    template <int ROW_SIZE, int COLUMN_SIZE>
    using DxMatrixRef = dx::linalg::MatrixRefImpl<BufferT, ELEM_TYPE, (uint)ROW_SIZE, (uint)COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED>;


    void set(BufferT buffer, const uint startOffset)
    {
      m_buffer = buffer;
      m_startOffset = startOffset;
    }

    template <int ROW_SIZE, int COLUMN_SIZE>
    DxMatrixRef<ROW_SIZE, COLUMN_SIZE> makeDxMatrixRef(const uint extraOffset)
    {
      const uint stride = impl::getWeightMatrixStride<ROW_SIZE, COLUMN_SIZE, ELEM_TYPE, MATRIX_LAYOUT, STRIDE_ALIGNMENT>();
      DxMatrixRef<ROW_SIZE, COLUMN_SIZE> ref = {m_buffer, m_startOffset + extraOffset, stride};
      return ref;
    }

    BufferT m_buffer;
    uint m_startOffset;
  };
};

template <>
struct BiasData<true>
{
  static const bool HAS_BIAS = true;

  template <typename BufferT,
            dx::linalg::DataType ELEM_TYPE,
            uint ALIGNMENT>
  struct BufferRef
  {
    using DxVectorRef = dx::linalg::VectorRefImpl<BufferT, ELEM_TYPE>;


    void set(BufferT buffer, const uint startOffset)
    {
      m_buffer = buffer;
      m_startOffset = startOffset;
    }

    DxVectorRef makeDxVectorRef(const uint extraOffset)
    {
      DxVectorRef ref = {m_buffer, m_startOffset + extraOffset};
      return ref;
    }

    BufferT m_buffer;
    uint m_startOffset;
  };
};

template <>
struct BiasData<false>
{
  static const bool HAS_BIAS = false;

  template <typename BufferT,
            dx::linalg::DataType ELEM_TYPE,
            uint ALIGNMENT>
  struct BufferRef
  {
    using DxVectorRef = dx::linalg::VectorRefImpl<BufferT, ELEM_TYPE>;


    void set(BufferT, const uint) {}
  };
};


struct IdentityActivation
{
  template <typename OutputElemT, typename InputElemT, int N>
  void forward(out vector<OutputElemT, N> output, const vector<InputElemT, N> input)
  {
    using OutputVecT = vector<OutputElemT, N>;

    output = (OutputVecT)input;
  }
};

struct SigmoidActivation
{
  template <typename OutputElemT, typename InputElemT, int N>
  void forward(out vector<OutputElemT, N> output, const vector<InputElemT, N> input)
  {
    using InputVecT = vector<InputElemT, N>;
    using OutputVecT = vector<OutputElemT, N>;

    const OutputVecT e = exp(-abs(input));
    const OutputElemT one = (InputElemT)1;
    const OutputVecT oneOverEPlusOne = one / (e + one);
    const InputElemT zero = (InputElemT)0;
    output = select(zero > input, one - oneOverEPlusOne, oneOverEPlusOne); 
  }
};

struct ReluActivation
{
  template <typename OutputElemT, typename InputElemT, int N>
  void forward(out vector<OutputElemT, N> output, const vector<InputElemT, N> input)
  {
    using InputVecT = vector<InputElemT, N>;
    using OutputVecT = vector<OutputElemT, N>;

    const InputVecT zero = (InputVecT)0;
    const InputVecT y = max(zero, input);
    output = (OutputVecT)y;
  }
};

struct LeakyReluActivation
{
  template <typename OutputElemT, typename InputElemT, int N>
  void forward(out vector<OutputElemT, N> output, const vector<InputElemT, N> input)
  {
    using InputVecT = vector<InputElemT, N>;
    using OutputVecT = vector<OutputElemT, N>;

    const InputElemT negativeSlope = (InputElemT)0.01; // It must be [0, 1]
    const InputVecT y = max(negativeSlope * input, input);
    output = (OutputVecT)y;
  }
};

//

template <typename OutputElemT,
          int OUTPUT_DIM,
          typename InputElemT,
          int INPUT_DIM,
          uint NUM_HIDDEN_LAYERS,
          int HIDDEN_LAYER_DIM,
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          bool HAS_BIAS,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE,
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
          typename ActivationHiddenT,
          typename ActivationLastT,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE,
          bool IS_WEIGHT_MATRIX_TRANSPOSED,
          uint WEIGHT_ALIGNMENT,
          uint WEIGHT_STRIDE_ALIGNMENT,
          uint BIAS_ALIGNMENT>
void forward(out vector<OutputElemT, OUTPUT_DIM> output,
             const vector<InputElemT, INPUT_DIM> input,
             const LayerDataRefImpl<NUM_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, HAS_BIAS, BiasBufferT, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, IS_WEIGHT_MATRIX_TRANSPOSED, WEIGHT_ALIGNMENT, WEIGHT_STRIDE_ALIGNMENT, BIAS_ALIGNMENT> layerData)
{
  impl::Mlp<NUM_HIDDEN_LAYERS>::forward(output, input, layerData);
}

} // mininn

#endif // MINIDXNN_MLP_HLSL
