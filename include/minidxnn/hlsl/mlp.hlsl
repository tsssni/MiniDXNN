/*!
  \file mlp.hlsl
  \author Sho Ikeda
  \brief Header-only HLSL library for MLP forward & backward passes using DX12 Cooperative Vector
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  MiniDXNN — a single-header HLSL library that provides:
    - Forward pass (inference) with configurable activations
    - Backward pass (training) with gradient accumulation
    - Hardware-accelerated matrix-vector operations via dx::linalg Cooperative Vector
    - Software fallback when MINIDXNN_USE_SOFTWARE_LINALG_IMPL is defined

  Usage:
    #include <minidxnn/hlsl/mlp.hlsl>

  Preprocessor options (define before including this header):
    MINIDXNN_NO_INCLUDE_DX_LINALG     — skip #include <dx/linalg.h>; you provide it yourself
    MINIDXNN_USE_SOFTWARE_LINALG_IMPL — use software mat-vec ops instead of Cooperative Vector

  See docs/mlp_hlsl.md for the full API reference.
*/

#ifndef MINIDXNN_MLP_HLSL
#define MINIDXNN_MLP_HLSL

// C++ reference qualifier for out/inout parameters
#ifdef __cplusplus
#define MINIDXNN_IN(...) const __VA_ARGS__ &
#define MINIDXNN_OUT(...) __VA_ARGS__ &
#define MINIDXNN_INOUT(...) __VA_ARGS__ &
#else
#define MINIDXNN_IN(...) in const __VA_ARGS__
#define MINIDXNN_OUT(...) out __VA_ARGS__
#define MINIDXNN_INOUT(...) inout __VA_ARGS__
#endif

#ifndef MINIDXNN_NO_INCLUDE_DX_LINALG
#include <dx/linalg.h>
#endif // MINIDXNN_NO_INCLUDE_DX_LINALG

namespace mininn {

// ============================================================================
// Forward declarations
// ============================================================================

// Activation functions (implementations at the bottom of this file)
struct IdentityActivation;
struct SigmoidActivation;
struct ReluActivation;
struct LeakyReluActivation;

// Internal buffer-data wrappers (cache vs no-cache specializations at the bottom)
enum CacheMethod
{
  NO_CACHE = 0,
  CACHE,
};

template <CacheMethod CACHE_METHOD> struct MatrixData;
template <CacheMethod CACHE_METHOD> struct VectorData;

// ============================================================================
// LayerDataRefImpl — Core MLP layer-stack descriptor
// ============================================================================
//
// Holds buffer references for weights, biases, gradients, and logits caches.
// Parameterized on element types, cache methods, buffer types, alignments,
// and activation function types. Users typically interact with one of the
// convenience aliases below (InferenceLayerDataRef, TrainingLayerDataRef, etc.).
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          CacheMethod WEIGHT_CACHE_METHOD,
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Weight gradient cache
          CacheMethod WEIGHT_GRADIENT_CACHE_METHOD = CacheMethod::NO_CACHE,
          typename WeightGradientCacheBufferT = WeightBufferT,
          dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Bias
          CacheMethod BIAS_CACHE_METHOD = CacheMethod::NO_CACHE,
          typename BiasBufferT = WeightBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Bias gradient cache
          CacheMethod BIAS_GRADIENT_CACHE_METHOD = CacheMethod::NO_CACHE,
          typename BiasGradientCacheBufferT = BiasBufferT,
          dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE = BIAS_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          CacheMethod LOGITS_CACHE_METHOD = CacheMethod::NO_CACHE,
          typename LogitsCacheBufferT = WeightBufferT,
          dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
struct LayerDataRefImpl
{
  static const bool HAS_BIAS = BIAS_CACHE_METHOD == CacheMethod::CACHE;
  static const uint NUM_BACKBONE_LAYERS = NUM_LAYERS - 1;


  template <CacheMethod CACHE_METHOD, typename BufferT, dx::linalg::DataType ELEM_TYPE, dx::linalg::MatrixLayout LAYOUT>
  using MatrixRefT = typename MatrixData<CACHE_METHOD>::template Ref<BufferT, ELEM_TYPE, LAYOUT, NUM_LAYERS, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT>;
  template <CacheMethod CACHE_METHOD, typename BufferT, dx::linalg::DataType ELEM_TYPE>
  using VectorRefT = typename VectorData<CACHE_METHOD>::template Ref<BufferT, ELEM_TYPE, NUM_LAYERS, HIDDEN_LAYER_DIM, BIAS_VECTOR_ALIGNMENT>;


  void setWeightData(WeightBufferT buffer, const uint2 matrixSize, const uint startOffset = 0)
  {
    m_weight.set(buffer, matrixSize, startOffset);
  }

  void setWeightGradientCache(WeightGradientCacheBufferT buffer, const uint2 matrixSize, const uint startOffset = 0)
  {
    m_weightGradCache.set(buffer, matrixSize, startOffset);
  }

  void setBiasData(BiasBufferT buffer, const uint startOffset = 0)
  {
    m_bias.set(buffer, startOffset);
  }

  void setBiasGradientCache(BiasGradientCacheBufferT buffer, const uint startOffset = 0)
  {
    m_biasGradCache.set(buffer, startOffset);
  }

  void setLogitsCache(LogitsCacheBufferT buffer, const uint startOffset = 0)
  {
    m_logitsCache.set(buffer, startOffset);
  }


  MatrixRefT<WEIGHT_CACHE_METHOD, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT> m_weight;
  VectorRefT<BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE> m_bias;
  MatrixRefT<WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT> m_weightGradCache;
  VectorRefT<BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE> m_biasGradCache;
  VectorRefT<LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE> m_logitsCache;
  ActivationHiddenT m_activationHidden;
  ActivationLastT m_activationLast;
};


// ============================================================================
// Convenience type aliases — Inference
// ============================================================================
//
// These aliases fix buffer types and bias flags so callers only specify
// the architectural parameters (layer count, dimensions, activations, etc.)

// InferenceLayerDataRefImpl: base alias that maps to LayerDataRefImpl with
// inference-appropriate cache settings (weights cached, no gradient caches).
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Bias
          bool HAS_BIAS,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
using InferenceLayerDataRefImpl = LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, CacheMethod::NO_CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, HAS_BIAS ? CacheMethod::CACHE : CacheMethod::NO_CACHE, BiasBufferT, BIAS_ELEM_TYPE, CacheMethod::NO_CACHE, BiasBufferT, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, CacheMethod::NO_CACHE, WeightBufferT, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>;

// InferenceLayerDataRef: read-only inference with bias (ByteAddressBuffer).
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Bias
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
using InferenceLayerDataRef = InferenceLayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, ByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, true, ByteAddressBuffer, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>;

// InferenceLayerDataRefNoBias: read-only inference without bias (ByteAddressBuffer).
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
using InferenceLayerDataRefNoBias = InferenceLayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, ByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, false, ByteAddressBuffer, WEIGHT_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>;

// RWInferenceLayerDataRef: read-write inference with bias (RWByteAddressBuffer).
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Bias
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
using RWInferenceLayerDataRef = InferenceLayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, RWByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, true, RWByteAddressBuffer, BIAS_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>;

// RWInferenceLayerDataRefNoBias: read-write inference without bias (RWByteAddressBuffer).
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
using RWInferenceLayerDataRefNoBias = InferenceLayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, RWByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, false, RWByteAddressBuffer, WEIGHT_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>;

// ============================================================================
// Convenience type aliases — Training
// ============================================================================
//
// Training aliases enable gradient cache buffers (RWByteAddressBuffer) for weight
// and bias gradients, plus a logits cache for storing pre-activation values needed
// by the backward pass.

// TrainingLayerDataRefImpl: base training alias with configurable buffer types.
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Weight gradient cache
          dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
          // Bias
          bool HAS_BIAS,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Bias gradient cache
          dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE = BIAS_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
using TrainingLayerDataRefImpl = LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, CacheMethod::CACHE, RWByteAddressBuffer, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, HAS_BIAS ? CacheMethod::CACHE : CacheMethod::NO_CACHE, BiasBufferT, BIAS_ELEM_TYPE, HAS_BIAS ? CacheMethod::CACHE : CacheMethod::NO_CACHE, RWByteAddressBuffer, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, CacheMethod::CACHE, RWByteAddressBuffer, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>;

// TrainingLayerDataRef: read-only weights with bias, RW gradient/logits caches (ByteAddressBuffer).
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Weight gradient cache
          dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
          // Bias
          dx::linalg::DataType BIAS_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Bias gradient cache
          dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE = BIAS_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
using TrainingLayerDataRef = TrainingLayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, ByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, true, ByteAddressBuffer, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>;

// TrainingLayerDataRefNoBias: training without bias.
template <// Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Weight gradient cache
          dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT = IdentityActivation,
          typename ActivationLastT = IdentityActivation,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE = WEIGHT_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT = 128,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT = 16,
          uint BIAS_VECTOR_ALIGNMENT = 64>
using TrainingLayerDataRefNoBias = TrainingLayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, ByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, false, ByteAddressBuffer, WEIGHT_ELEM_TYPE, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>;


// ============================================================================
// Public API — forward() and backward() entry points
// ============================================================================
//
// forward(): Runs a complete MLP forward pass (input → hidden layers → output).
// backward(): Runs a complete backward pass, accumulating weight/bias gradients.
//             Requires a preceding forward() call to populate the logits cache.

template <// Output
          typename OutputElemT,
          int OUTPUT_DIM,
          // Input
          typename InputElemT,
          int INPUT_DIM,
          // Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Weight gradient cache
          CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
          typename WeightGradientCacheBufferT,
          dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
          // Bias
          CacheMethod BIAS_CACHE_METHOD,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE,
          // Bias gradient cache
          CacheMethod BIAS_GRADIENT_CACHE_METHOD,
          typename BiasGradientCacheBufferT,
          dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
          CacheMethod LOGITS_CACHE_METHOD,
          typename LogitsCacheBufferT,
          dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT,
          typename ActivationLastT,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
          uint BIAS_VECTOR_ALIGNMENT>
void forward(MINIDXNN_OUT(vector<OutputElemT, OUTPUT_DIM>) output,
             MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
             MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData);

template <// Output
          typename OutputElemT,
          int OUTPUT_DIM,
          // Input
          typename InputElemT,
          int INPUT_DIM,
          // Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Weight gradient cache
          CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
          typename WeightGradientCacheBufferT,
          dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
          // Bias
          CacheMethod BIAS_CACHE_METHOD,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE,
          // Bias gradient cache
          CacheMethod BIAS_GRADIENT_CACHE_METHOD,
          typename BiasGradientCacheBufferT,
          dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
          CacheMethod LOGITS_CACHE_METHOD,
          typename LogitsCacheBufferT,
          dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT,
          typename ActivationLastT,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
          uint BIAS_VECTOR_ALIGNMENT>
vector<OutputElemT, INPUT_DIM>
backward(MINIDXNN_IN(vector<OutputElemT, OUTPUT_DIM>) downstreamGrad,
         MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
         MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData);


// ============================================================================
// Implementation details (mininn::impl)
// ============================================================================
//
// Everything below is internal machinery. Users should not depend on these
// types or functions directly; use the public API above.

namespace impl {

// ----------------------------------------------------------------------------
// TypeTraits / ComponentTypeTraits — HLSL scalar ↔ dx::linalg::DataType mapping
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// VectorBufferAccessor — typed load/store for ByteAddressBuffer / RWByteAddressBuffer
// ----------------------------------------------------------------------------
// The float16_t specialization handles manual uint32 packing/unpacking for correct FP16 I/O.
// Reference: https://therealmjp.github.io/posts/shader-fp16/
template <typename T>
struct VectorBufferAccessor
{
  using Type = T;

  template <int N>
  static vector<Type, N> load(ByteAddressBuffer buffer, const uint offset)
  {
    using VecT = vector<Type, N>;
    return buffer.template Load<VecT>(offset);
  }

  template <int N>
  static vector<Type, N> load(RWByteAddressBuffer buffer, const uint offset)
  {
    using VecT = vector<Type, N>;
    return buffer.template Load<VecT>(offset);
  }

  template <int N>
  static void store(RWByteAddressBuffer buffer, const uint offset, MINIDXNN_IN(vector<Type, N>) value)
  {
    using VecT = vector<Type, N>;
    buffer.template Store<VecT>(offset, value);
  }
};

// VectorBufferAccessor<float16_t>: Packs/unpacks two float16 values per uint32
// for correct alignment-safe I/O. Reference: https://therealmjp.github.io/posts/shader-fp16/
template <>
struct VectorBufferAccessor<float16_t>
{
  using Type = float16_t;

  template <int N>
  static vector<Type, N> load(ByteAddressBuffer buffer, const uint offset)
  {
    return __loadImpl<N>(buffer, offset);
  }

  template <int N>
  static vector<Type, N> load(RWByteAddressBuffer buffer, const uint offset)
  {
    return __loadImpl<N>(buffer, offset);
  }

  template <int N>
  static void store(RWByteAddressBuffer buffer, const uint offset, MINIDXNN_IN(vector<Type, N>) value)
  {
    __storeImpl<N>(buffer, offset, value);
  }

  // Internal helpers for packing/unpacking float16 via uint32
  template <int N, typename BufferT>
  static vector<Type, N> __loadImpl(BufferT buffer, const uint offset)
  {
    const bool isLastOdd = (N & 0x01) == 0x01;
    const int NS = (N + 1) / 2;
    using VecT = vector<half, N>;
    using StagingVecT = vector<uint32_t, NS>;

    VecT output;
    StagingVecT staging = buffer.template Load<StagingVecT>(offset);
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
  static void __storeImpl(BufferT buffer, const uint offset, const vector<Type, N> value)
  {
    const bool isLastOdd = (N & 0x01) == 0x01;
    const int NS = (N + 1) / 2;
    using StagingVecT = vector<uint32_t, NS>;

    StagingVecT staging = (StagingVecT)0;
    for (int i = 0; i < (N / 2); ++i) {
      staging[i] = (uint32_t)asuint16(value[2 * i]) | ((uint32_t)asuint16(value[2 * i + 1]) << 16);
    }
    if (isLastOdd) {
      staging[NS - 1] = (uint32_t)asuint16(value[N - 1]);
    }
    buffer.template Store<StagingVecT>(offset, staging);
  }
};

// ----------------------------------------------------------------------------
// Atomic helpers — CAS-loop float/float16 atomic add for gradient accumulation
// ----------------------------------------------------------------------------

// AtomicManipF16: Helper for CAS-based atomic operations on float16 values
// packed two-per-uint32 in a ByteAddressBuffer.
struct AtomicManipF16
{
  static float16_t get(const uint32_t bits, const bool isHigh16Bits)
  {
    const uint32_t f16Mask = 0xffff;
    const uint32_t f16Bits = isHigh16Bits ? (bits >> 16) : (bits & f16Mask);
    const float16_t f16 = (float16_t)f16tof32(f16Bits);
    return f16;
  }

  static void set(MINIDXNN_INOUT(uint32_t) bits, const float16_t value, const bool isHigh16Bits)
  {
    const uint32_t f16Mask = 0xffff;
    const uint32_t f16Bits = (uint32_t)f32tof16((float)value);
    bits = isHigh16Bits ? ((uint32_t)(f16Bits << 16) | (uint32_t)(bits & f16Mask))
                        : ((uint32_t)(bits & ~f16Mask) | f16Bits);
  }

  static uint32_t add(const uint32_t lhs, const float16_t rhs, const bool isHigh16Bits)
  {
    const float16_t lhsF16 = get(lhs, isHigh16Bits);
    const float16_t resultF16 = lhsF16 + rhs;
    uint32_t result = lhs;
    set(result, resultF16, isHigh16Bits);
    return result;
  }
};

// CAS-loop atomic fetch-add for floating-point types
template <typename Type>
Type atomicFetchAdd(RWByteAddressBuffer object, const uint offset, const Type value);

template <>
float32_t atomicFetchAdd(RWByteAddressBuffer object, const uint offset, const float32_t value)
{
  uint32_t oldValue = 0;
#ifndef __cplusplus
  [allow_uav_condition]
#endif // __cplusplus
  while (true) {
    oldValue = object.Load<uint32_t>(offset);
    const uint32_t newValue = asuint(asfloat(oldValue) + value);
    uint32_t originalValue = 0;
    object.InterlockedCompareExchange(offset, oldValue, newValue, originalValue);
    if (oldValue == originalValue) {
      break;
    }
    oldValue = originalValue;
  }
  return asfloat(oldValue);
}

template <>
float16_t atomicFetchAdd(RWByteAddressBuffer object, const uint offset, const float16_t value)
{
  using ManipT = AtomicManipF16;
  const uint f16Index = offset / 2;
  const uint u32Offset = sizeof(uint32_t) * (f16Index / 2);
  const bool isHigh16Bits = (f16Index & 0x1) == 0x1;

  uint32_t oldValue = 0;
#ifndef __cplusplus
  [allow_uav_condition]
#endif // __cplusplus
  while (true) {
    oldValue = object.Load<uint32_t>(u32Offset);
    const uint32_t newValue = ManipT::add(oldValue, value, isHigh16Bits);
    uint32_t originalValue = 0;
    object.InterlockedCompareExchange(u32Offset, oldValue, newValue, originalValue);
    if (oldValue == originalValue) {
      break;
    }
    oldValue = originalValue;
  }
  return ManipT::get(oldValue, isHigh16Bits);
}

// ----------------------------------------------------------------------------
// Utility — alignment, matrix layout, linear algebra operations
// ----------------------------------------------------------------------------

// Align size up to the nearest multiple of alignmentInBytes (must be a power of 2)
uint align(const uint sizeInBytes, const uint alignmentInBytes)
{
  const uint a = alignmentInBytes - 1;
  return (sizeInBytes + a) & (~a);
}

// MatrixDataLayout: Computes byte offsets and provides row/column access
// for matrices stored in row-major, column-major, or optimal layouts.
template <dx::linalg::DataType ELEM_TYPE, dx::linalg::MatrixLayout LAYOUT>
struct MatrixDataLayout
{
  using Type = typename ComponentTypeTraits<ELEM_TYPE>::Type;


  static const bool IS_ROW_MAJOR = LAYOUT == dx::linalg::MATRIX_LAYOUT_ROW_MAJOR;
  static const bool IS_COLUMN_MAJOR = LAYOUT == dx::linalg::MATRIX_LAYOUT_COLUMN_MAJOR;
  static const bool IS_MUL_OPTIMAL = LAYOUT == dx::linalg::MATRIX_LAYOUT_MUL_OPTIMAL;
  static const bool IS_OUTER_PRODUCT_OPTIMAL = LAYOUT == dx::linalg::MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL;


  template <uint ROW_SIZE, uint COLUMN_SIZE, bool IS_TRANSPOSED>
  static uint getMinorSize()
  {
    const int isColumnMajor = IS_COLUMN_MAJOR ? 1 : 0;
    const int isTransposed = IS_TRANSPOSED ? 1 : 0;
    const uint size = (isColumnMajor ^ isTransposed) ? ROW_SIZE : COLUMN_SIZE;
    return size;
  }

  template <uint ROW_SIZE,
            uint COLUMN_SIZE,
            bool IS_TRANSPOSED>
  static uint getElementOffset(const uint row, const uint column, const uint vectorStride)
  {
    const int isColumnMajor = IS_COLUMN_MAJOR ? 1 : 0;
    const int isTransposed = IS_TRANSPOSED ? 1 : 0;
    const uint major = (isColumnMajor ^ isTransposed) ? column : row;
    const uint minor = (isColumnMajor ^ isTransposed) ? row : column;
    const uint offset = major * vectorStride + minor * sizeof(Type);
    return offset;
  }

  template <uint ROW_SIZE,
            uint COLUMN_SIZE,
            bool IS_TRANSPOSED,
            typename BufferT>
  static vector<Type, COLUMN_SIZE> getRowVector(BufferT buffer, const uint row, const uint startOffset, const uint vectorStride)
  {
    const int isColumnMajor = IS_COLUMN_MAJOR ? 1 : 0;
    const int isTransposed = IS_TRANSPOSED ? 1 : 0;
    const bool isFastPath = (isColumnMajor ^ isTransposed) == 0;

    vector<Type, COLUMN_SIZE> rowVec;
    if (isFastPath) {
      using VectorBufferAccessorT = VectorBufferAccessor<Type>;
      const uint offset = startOffset + row * vectorStride;
      rowVec = VectorBufferAccessorT::template load<COLUMN_SIZE>(buffer, offset);
    }
    else { // slow path
      for (uint column = 0; column < COLUMN_SIZE; ++column) {
        const uint offset = startOffset + getElementOffset<ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED>(row, column, vectorStride);
        rowVec[column] = buffer.template Load<Type>(offset);
      }
    }
    return rowVec;
  }
};

// LinearAlgebra: Matrix-vector operations with SW (software) and HW (dx::linalg) paths.
// mul / mulAdd: used in the forward pass
// outerProductAcc / vectorAcc: used in the backward pass for gradient accumulation
struct LinearAlgebra
{
  template <typename OutputElemT,
            typename InputElemT,
            int INPUT_ELEM_COUNT,
            typename MatrixBufferT,
            dx::linalg::DataType INPUT_ELEM_TYPE,
            dx::linalg::DataType MATRIX_ELEM_TYPE,
            uint ROW_SIZE,
            uint COLUMN_SIZE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            bool IS_MATRIX_TRANSPOSED>
  static
  vector<OutputElemT, ROW_SIZE>
  mulSW(MINIDXNN_IN(dx::linalg::MatrixRefImpl<MatrixBufferT, MATRIX_ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED>) matrix,
        MINIDXNN_IN(dx::linalg::InterpretedVector<InputElemT, INPUT_ELEM_COUNT, INPUT_ELEM_TYPE>) input)
  {
    using OutputVecT = vector<OutputElemT, ROW_SIZE>;
    using LayoutT = MatrixDataLayout<MATRIX_ELEM_TYPE, MATRIX_LAYOUT>;
    using MatrixElemT = typename LayoutT::Type;
    using RowVecT = vector<MatrixElemT, COLUMN_SIZE>;
    using RowAccumVecT = vector<OutputElemT, COLUMN_SIZE>;

    OutputVecT output = (OutputVecT)0;
    for (uint row = 0; row < ROW_SIZE; ++row) {
      const RowVecT rowVec = LayoutT::template getRowVector<ROW_SIZE, COLUMN_SIZE, IS_MATRIX_TRANSPOSED>(matrix.Buffer, row, matrix.StartOffset, matrix.Stride);
      output[row] = dot((RowAccumVecT)rowVec, (RowAccumVecT)input.Data);
    }
    return output;
  }

  template <typename OutputElemT,
            typename InputElemT,
            int INPUT_ELEM_COUNT,
            typename MatrixBufferT,
            dx::linalg::DataType INPUT_ELEM_TYPE,
            dx::linalg::DataType MATRIX_ELEM_TYPE,
            uint ROW_SIZE,
            uint COLUMN_SIZE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            bool IS_MATRIX_TRANSPOSED>
  static
  vector<OutputElemT, ROW_SIZE>
  mulHW(MINIDXNN_IN(dx::linalg::MatrixRefImpl<MatrixBufferT, MATRIX_ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED>) matrix,
        MINIDXNN_IN(dx::linalg::InterpretedVector<InputElemT, INPUT_ELEM_COUNT, INPUT_ELEM_TYPE>) input)
  {
    return dx::linalg::Mul<OutputElemT>(matrix, input);
  }

  template <typename OutputElemT,
            typename InputElemT,
            int INPUT_ELEM_COUNT,
            typename MatrixBufferT,
            dx::linalg::DataType INPUT_ELEM_TYPE,
            dx::linalg::DataType MATRIX_ELEM_TYPE,
            uint ROW_SIZE,
            uint COLUMN_SIZE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            bool IS_MATRIX_TRANSPOSED>
  static
  vector<OutputElemT, ROW_SIZE>
  mul(MINIDXNN_IN(dx::linalg::MatrixRefImpl<MatrixBufferT, MATRIX_ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED>) matrix,
      MINIDXNN_IN(dx::linalg::InterpretedVector<InputElemT, INPUT_ELEM_COUNT, INPUT_ELEM_TYPE>) input)
  {
    using OutputVecT = vector<OutputElemT, ROW_SIZE>;

    const OutputVecT output = 
#if defined(MINIDXNN_USE_SOFTWARE_LINALG_IMPL) && (MINIDXNN_USE_SOFTWARE_LINALG_IMPL != 0)
        mulSW<OutputElemT>(matrix, input);
#else // MINIDXNN_USE_SOFTWARE_LINALG_IMPL
        mulHW<OutputElemT>(matrix, input);
#endif // MINIDXNN_USE_SOFTWARE_LINALG_IMPL
    return output;
  }

  template <typename OutputElemT,
            typename InputElemT,
            int INPUT_ELEM_COUNT,
            typename MatrixBufferT,
            dx::linalg::DataType INPUT_ELEM_TYPE,
            dx::linalg::DataType MATRIX_ELEM_TYPE,
            uint ROW_SIZE,
            uint COLUMN_SIZE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            bool IS_MATRIX_TRANSPOSED,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE>
  static
  vector<OutputElemT, ROW_SIZE>
  mulAddSW(MINIDXNN_IN(dx::linalg::MatrixRefImpl<MatrixBufferT, MATRIX_ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED>) matrix,
           MINIDXNN_IN(dx::linalg::InterpretedVector<InputElemT, INPUT_ELEM_COUNT, INPUT_ELEM_TYPE>) input,
           MINIDXNN_IN(dx::linalg::VectorRefImpl<BiasBufferT, BIAS_ELEM_TYPE>) bias)
  {
    using OutputVecT = vector<OutputElemT, ROW_SIZE>;
    using BiasElemT = typename ComponentTypeTraits<BIAS_ELEM_TYPE>::Type;
    using BiasVecT = vector<BiasElemT, ROW_SIZE>;
    using VectorBufferAccessorT = VectorBufferAccessor<BiasElemT>;

    const BiasVecT biasVec = VectorBufferAccessorT::template load<ROW_SIZE>(bias.Buffer, bias.StartOffset);
    const OutputVecT output = mul<OutputElemT>(matrix, input) + (OutputVecT)biasVec;
    return output;
  }

  template <typename OutputElemT,
            typename InputElemT,
            int INPUT_ELEM_COUNT,
            typename MatrixBufferT,
            dx::linalg::DataType INPUT_ELEM_TYPE,
            dx::linalg::DataType MATRIX_ELEM_TYPE,
            uint ROW_SIZE,
            uint COLUMN_SIZE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            bool IS_MATRIX_TRANSPOSED,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE>
  static
  vector<OutputElemT, ROW_SIZE>
  mulAddHW(MINIDXNN_IN(dx::linalg::MatrixRefImpl<MatrixBufferT, MATRIX_ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED>) matrix,
           MINIDXNN_IN(dx::linalg::InterpretedVector<InputElemT, INPUT_ELEM_COUNT, INPUT_ELEM_TYPE>) input,
           MINIDXNN_IN(dx::linalg::VectorRefImpl<BiasBufferT, BIAS_ELEM_TYPE>) bias)
  {
    return dx::linalg::MulAdd<OutputElemT>(matrix, input, bias);
  }

  template <typename OutputElemT,
            typename InputElemT,
            int INPUT_ELEM_COUNT,
            typename MatrixBufferT,
            dx::linalg::DataType INPUT_ELEM_TYPE,
            dx::linalg::DataType MATRIX_ELEM_TYPE,
            uint ROW_SIZE,
            uint COLUMN_SIZE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT,
            bool IS_MATRIX_TRANSPOSED,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE>
  static
  vector<OutputElemT, ROW_SIZE>
  mulAdd(MINIDXNN_IN(dx::linalg::MatrixRefImpl<MatrixBufferT, MATRIX_ELEM_TYPE, ROW_SIZE, COLUMN_SIZE, MATRIX_LAYOUT, IS_MATRIX_TRANSPOSED>) matrix,
         MINIDXNN_IN(dx::linalg::InterpretedVector<InputElemT, INPUT_ELEM_COUNT, INPUT_ELEM_TYPE>) input,
         MINIDXNN_IN(dx::linalg::VectorRefImpl<BiasBufferT, BIAS_ELEM_TYPE>) bias)
  {
    using OutputVecT = vector<OutputElemT, ROW_SIZE>;

    const OutputVecT output =
#if defined(MINIDXNN_USE_SOFTWARE_LINALG_IMPL) && (MINIDXNN_USE_SOFTWARE_LINALG_IMPL != 0)
        mulAddSW<OutputElemT>(matrix, input, bias);
#else // MINIDXNN_USE_SOFTWARE_LINALG_IMPL
        mulAddHW<OutputElemT>(matrix, input, bias);
#endif // MINIDXNN_USE_SOFTWARE_LINALG_IMPL
    return output;
  }

  // outerProductAccSW: outer product gradient accumulation
  // Uses int dimensions for vector parameters (HLSL vector<T, int>) and
  // uint dimensions for RWMatrixRef (dx::linalg uses uint).
  template <typename InputElemT,
            int ROW_SIZE,
            int COLUMN_SIZE,
            dx::linalg::DataType MATRIX_ELEM_TYPE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT>
  static
  void outerProductAccSW(MINIDXNN_IN(vector<InputElemT, ROW_SIZE>) vecLhs,
                         MINIDXNN_IN(vector<InputElemT, COLUMN_SIZE>) vecRhs,
                         dx::linalg::RWMatrixRef<MATRIX_ELEM_TYPE, (uint)ROW_SIZE, (uint)COLUMN_SIZE, MATRIX_LAYOUT, false> matrix)
  {
    using LayoutT = MatrixDataLayout<MATRIX_ELEM_TYPE, MATRIX_LAYOUT>;
    using MatrixElemT = typename LayoutT::Type;
    const bool isMatrixTransposed = false;

    if (LayoutT::IS_COLUMN_MAJOR) { // For column-major memory access
      for (uint row = 0; row < (uint)ROW_SIZE; ++row) {
        const vector<InputElemT, COLUMN_SIZE> rowVec = vecLhs[row] * vecRhs;
        for (uint column = 0; column < (uint)COLUMN_SIZE; ++column) {
          const uint offset = LayoutT::template getElementOffset<(uint)ROW_SIZE, (uint)COLUMN_SIZE, isMatrixTransposed>(row, column, matrix.Stride);
          const MatrixElemT value = (MatrixElemT)rowVec[column];
          atomicFetchAdd(matrix.Buffer, matrix.StartOffset + offset, value);
        }
      }
    }
    else { // For row-major memory access
      for (uint column = 0; column < (uint)COLUMN_SIZE; ++column) {
        const vector<InputElemT, ROW_SIZE> columnVec = vecLhs * vecRhs[column];
        for (uint row = 0; row < (uint)ROW_SIZE; ++row) {
          const uint offset = LayoutT::template getElementOffset<(uint)ROW_SIZE, (uint)COLUMN_SIZE, isMatrixTransposed>(row, column, matrix.Stride);
          const MatrixElemT value = (MatrixElemT)columnVec[row];
          atomicFetchAdd(matrix.Buffer, matrix.StartOffset + offset, value);
        }
      }
    }
  }

  template <typename InputElemT,
            int ROW_SIZE,
            int COLUMN_SIZE,
            dx::linalg::DataType MATRIX_ELEM_TYPE,
            dx::linalg::MatrixLayout MATRIX_LAYOUT>
  static
  void outerProductAcc(MINIDXNN_IN(vector<InputElemT, ROW_SIZE>) vecLhs,
                       MINIDXNN_IN(vector<InputElemT, COLUMN_SIZE>) vecRhs,
                       dx::linalg::RWMatrixRef<MATRIX_ELEM_TYPE, (uint)ROW_SIZE, (uint)COLUMN_SIZE, MATRIX_LAYOUT, false> matrix)
  {
    outerProductAccSW(vecLhs, vecRhs, matrix);
  }

  template <typename ElemT, int SIZE>
  static
  void vectorAccSW(MINIDXNN_IN(vector<ElemT, SIZE>) input,
                   dx::linalg::RWVectorRef<TypeTraits<ElemT>::COMPONENT_TYPE> vec)
  {
    for (uint i = 0; i < SIZE; ++i) {
      atomicFetchAdd(vec.Buffer, vec.StartOffset + i * sizeof(ElemT), input[i]);
    }
  }

  template <typename ElemT, int SIZE>
  static
  void vectorAcc(MINIDXNN_IN(vector<ElemT, SIZE>) input,
                 dx::linalg::RWVectorRef<TypeTraits<ElemT>::COMPONENT_TYPE> vec)
  {
    vectorAccSW(input, vec);
  }
};

// ----------------------------------------------------------------------------
// LinearLayer — single-layer forward/backward, with/without bias
// ----------------------------------------------------------------------------

template <bool HAS_BIAS> struct LinearLayer;

template <>
struct LinearLayer<true>
{
  static const bool HAS_BIAS = true;


  template <// Output
            typename OutputElemT,
            int OUTPUT_DIM,
            // Input
            typename InputElemT,
            int INPUT_DIM,
            // Layer configurations
            uint NUM_LAYERS,
            int HIDDEN_LAYER_DIM,
            // Weight
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            // Weight gradient cache
            CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
            typename WeightGradientCacheBufferT,
            dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
            // Bias
            CacheMethod BIAS_CACHE_METHOD,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            // Bias gradient cache
            CacheMethod BIAS_GRADIENT_CACHE_METHOD,
            typename BiasGradientCacheBufferT,
            dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
            // Pre-activation
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            CacheMethod LOGITS_CACHE_METHOD,
            typename LogitsCacheBufferT,
            dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
            // Activation functions
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            // Alignments
            uint WEIGHT_MATRIX_ALIGNMENT,
            uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
            uint BIAS_VECTOR_ALIGNMENT>
  static
  void forward(MINIDXNN_OUT(vector<OutputElemT, OUTPUT_DIM>) output,
               MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
               MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData,
               const uint layerIndex)
  {
    using InputTypeTraits = TypeTraits<InputElemT>;
    using OutputVecT = vector<OutputElemT, OUTPUT_DIM>;
    using LogitsElemType = typename ComponentTypeTraits<LOGITS_CACHE_ELEM_TYPE>::Type;
    using LogitsVecT = vector<LogitsElemType, OUTPUT_DIM>;

    const OutputVecT result = LinearAlgebra::mulAdd<OutputElemT>(
        layerData.m_weight.template makeDxMatrixRef<OUTPUT_DIM, INPUT_DIM>(layerIndex),
        dx::linalg::MakeInterpretedVector<InputTypeTraits::COMPONENT_TYPE>(input),
        layerData.m_bias.makeDxVectorRef(layerIndex));

    // Pre-activation cache
    layerData.m_logitsCache.setValue((LogitsVecT)result, layerIndex);

    output = result;
  }

  template <// upstream grad
            typename UpstreamElemT,
            int INPUT_DIM,
            // downstream grad
            typename DownstreamElemT,
            int OUTPUT_DIM,
            // Input
            typename InputElemT,
            // Layer configurations
            uint NUM_LAYERS,
            int HIDDEN_LAYER_DIM,
            // Weight
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            // Weight gradient cache
            CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
            typename WeightGradientCacheBufferT,
            dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
            // Bias
            CacheMethod BIAS_CACHE_METHOD,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            // Bias gradient cache
            CacheMethod BIAS_GRADIENT_CACHE_METHOD,
            typename BiasGradientCacheBufferT,
            dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
            // Pre-activation
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            CacheMethod LOGITS_CACHE_METHOD,
            typename LogitsCacheBufferT,
            dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
            // Activation functions
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            // Alignments
            uint WEIGHT_MATRIX_ALIGNMENT,
            uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
            uint BIAS_VECTOR_ALIGNMENT>
  static
  void backward(MINIDXNN_OUT(vector<UpstreamElemT, INPUT_DIM>) upstreamGrad,
                MINIDXNN_IN(vector<DownstreamElemT, OUTPUT_DIM>) downstreamGrad,
                MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
                MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData,
                const uint layerIndex)
  {
    LinearAlgebra::outerProductAcc(downstreamGrad, input, layerData.m_weightGradCache.template makeDxMatrixRef<OUTPUT_DIM, INPUT_DIM>(layerIndex));
    LinearAlgebra::vectorAcc(downstreamGrad, layerData.m_biasGradCache.makeDxVectorRef(layerIndex));
    const dx::linalg::DataType DOWNSTREAM_ELEM_T = TypeTraits<DownstreamElemT>::COMPONENT_TYPE;
    upstreamGrad = LinearAlgebra::mul<UpstreamElemT>(
        layerData.m_weight.template makeDxMatrixRef<INPUT_DIM, OUTPUT_DIM, true>(layerIndex),
        dx::linalg::MakeInterpretedVector<DOWNSTREAM_ELEM_T>(downstreamGrad));
  }
};

template <>
struct LinearLayer<false>
{
  static const bool HAS_BIAS = false;

  template <// Output
            typename OutputElemT,
            int OUTPUT_DIM,
            // Input
            typename InputElemT,
            int INPUT_DIM,
            // Layer configurations
            uint NUM_LAYERS,
            int HIDDEN_LAYER_DIM,
            // Weight
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            // Weight gradient cache
            CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
            typename WeightGradientCacheBufferT,
            dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
            // Bias
            CacheMethod BIAS_CACHE_METHOD,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            // Bias gradient cache
            CacheMethod BIAS_GRADIENT_CACHE_METHOD,
            typename BiasGradientCacheBufferT,
            dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
            // Pre-activation
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            CacheMethod LOGITS_CACHE_METHOD,
            typename LogitsCacheBufferT,
            dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
            // Activation functions
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            // Alignments
            uint WEIGHT_MATRIX_ALIGNMENT,
            uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
            uint BIAS_VECTOR_ALIGNMENT>
  static
  void forward(MINIDXNN_OUT(vector<OutputElemT, OUTPUT_DIM>) output,
               MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
               MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData,
               const uint layerIndex)
  {
    using InputTypeTraits = TypeTraits<InputElemT>;
    using OutputVecT = vector<OutputElemT, OUTPUT_DIM>;
    using LogitsElemType = typename ComponentTypeTraits<LOGITS_CACHE_ELEM_TYPE>::Type;
    using LogitsVecT = vector<LogitsElemType, OUTPUT_DIM>;

    const OutputVecT result = LinearAlgebra::mul<OutputElemT>(
        layerData.m_weight.template makeDxMatrixRef<OUTPUT_DIM, INPUT_DIM>(layerIndex),
        dx::linalg::MakeInterpretedVector<InputTypeTraits::COMPONENT_TYPE>(input));

    // Pre-activation cache
    layerData.m_logitsCache.setValue((LogitsVecT)result, layerIndex);

    output = result;
  }

  template <// upstream grad
            typename UpstreamElemT,
            int INPUT_DIM,
            // downstream grad
            typename DownstreamElemT,
            int OUTPUT_DIM,
            // Input
            typename InputElemT,
            // Layer configurations
            uint NUM_LAYERS,
            int HIDDEN_LAYER_DIM,
            // Weight
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            // Weight gradient cache
            CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
            typename WeightGradientCacheBufferT,
            dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
            // Bias
            CacheMethod BIAS_CACHE_METHOD,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            // Bias gradient cache
            CacheMethod BIAS_GRADIENT_CACHE_METHOD,
            typename BiasGradientCacheBufferT,
            dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
            // Pre-activation
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            CacheMethod LOGITS_CACHE_METHOD,
            typename LogitsCacheBufferT,
            dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
            // Activation functions
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            // Alignments
            uint WEIGHT_MATRIX_ALIGNMENT,
            uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
            uint BIAS_VECTOR_ALIGNMENT>
  static
  void backward(MINIDXNN_OUT(vector<UpstreamElemT, INPUT_DIM>) upstreamGrad,
                MINIDXNN_IN(vector<DownstreamElemT, OUTPUT_DIM>) downstreamGrad,
                MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
                MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData,
                const uint layerIndex)
  {
    LinearAlgebra::outerProductAcc(downstreamGrad, input, layerData.m_weightGradCache.template makeDxMatrixRef<OUTPUT_DIM, INPUT_DIM>(layerIndex));
    const dx::linalg::DataType DOWNSTREAM_ELEM_T = TypeTraits<DownstreamElemT>::COMPONENT_TYPE;
    upstreamGrad = LinearAlgebra::mul<UpstreamElemT>(
        layerData.m_weight.template makeDxMatrixRef<INPUT_DIM, OUTPUT_DIM, true>(layerIndex),
        dx::linalg::MakeInterpretedVector<DOWNSTREAM_ELEM_T>(downstreamGrad));
  }
};

// ----------------------------------------------------------------------------
// Mlp — multi-layer orchestrator (dispatches LinearLayer per depth)
// ----------------------------------------------------------------------------
// Specialized for NUM_LAYERS > 1 (multi-layer) and NUM_LAYERS == 1 (single-layer).
template <uint NUM_LAYERS>
struct Mlp
{
  template <// Output
            typename OutputElemT,
            int OUTPUT_DIM,
            // Input
            typename InputElemT,
            int INPUT_DIM,
            // Layer configurations
            int HIDDEN_LAYER_DIM,
            // Weight
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            // Weight gradient cache
            CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
            typename WeightGradientCacheBufferT,
            dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
            // Bias
            CacheMethod BIAS_CACHE_METHOD,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            // Bias gradient cache
            CacheMethod BIAS_GRADIENT_CACHE_METHOD,
            typename BiasGradientCacheBufferT,
            dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
            // Pre-activation
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            CacheMethod LOGITS_CACHE_METHOD,
            typename LogitsCacheBufferT,
            dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
            // Activation functions
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            // Alignments
            uint WEIGHT_MATRIX_ALIGNMENT,
            uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
            uint BIAS_VECTOR_ALIGNMENT>
  static
  void forward(MINIDXNN_OUT(vector<OutputElemT, OUTPUT_DIM>) output,
               MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
               MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData)
  {
    using AccumElemT = typename ComponentTypeTraits<ACCUMULATOR_ELEM_TYPE>::Type;
    using AccumVecT = vector<AccumElemT, HIDDEN_LAYER_DIM>;
    using ActElemT = typename ComponentTypeTraits<ACTIVATION_ELEM_TYPE>::Type;
    using ActVecT = vector<ActElemT, HIDDEN_LAYER_DIM>;
    const bool hasBias = BIAS_CACHE_METHOD == CacheMethod::CACHE;
    using LinearLayerT = LinearLayer<hasBias>;

    ActVecT actOut;
    // First layer
    {
      const uint depth = 0;
      AccumVecT layerOut;
      LinearLayerT::forward(layerOut, input, layerData, depth);
      layerData.m_activationHidden.forward(actOut, (ActVecT)layerOut);
    }
    // Hidden layers
    for (uint depth = 1; depth < layerData.NUM_BACKBONE_LAYERS; ++depth) {
      const ActVecT layerInput = actOut;
      AccumVecT layerOut;
      LinearLayerT::forward(layerOut, layerInput, layerData, depth);
      layerData.m_activationHidden.forward(actOut, (ActVecT)layerOut);
    }
    // Last layer
    {
      using AccumVecOutT = vector<AccumElemT, OUTPUT_DIM>;
      using ActVecOutT = vector<ActElemT, OUTPUT_DIM>;

      const uint depth = layerData.NUM_BACKBONE_LAYERS; 
      const ActVecT layerInput = actOut;
      AccumVecOutT lastLayerOut;
      LinearLayerT::forward(lastLayerOut, layerInput, layerData, depth);
      layerData.m_activationLast.forward(output, (ActVecOutT)lastLayerOut);
    }
  }

  template <// Output
            typename OutputElemT,
            int OUTPUT_DIM,
            // Input
            typename InputElemT,
            int INPUT_DIM,
            // Layer configurations
            int HIDDEN_LAYER_DIM,
            // Weight
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            // Weight gradient cache
            CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
            typename WeightGradientCacheBufferT,
            dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
            // Bias
            CacheMethod BIAS_CACHE_METHOD,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            // Bias gradient cache
            CacheMethod BIAS_GRADIENT_CACHE_METHOD,
            typename BiasGradientCacheBufferT,
            dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
            // Pre-activation
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            CacheMethod LOGITS_CACHE_METHOD,
            typename LogitsCacheBufferT,
            dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
            // Activation functions
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            // Alignments
            uint WEIGHT_MATRIX_ALIGNMENT,
            uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
            uint BIAS_VECTOR_ALIGNMENT>
  static
  vector<OutputElemT, INPUT_DIM>
  backward(MINIDXNN_IN(vector<OutputElemT, OUTPUT_DIM>) downstreamGrad,
           MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
           MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData)
  {
    using AccumElemT = typename ComponentTypeTraits<ACCUMULATOR_ELEM_TYPE>::Type;
    using AccumVecT = vector<AccumElemT, HIDDEN_LAYER_DIM>;
    using ActElemT = typename ComponentTypeTraits<ACTIVATION_ELEM_TYPE>::Type;
    using ActVecT = vector<ActElemT, HIDDEN_LAYER_DIM>;
    const bool hasBias = BIAS_CACHE_METHOD == CacheMethod::CACHE;
    using LinearLayerT = LinearLayer<hasBias>;

    AccumVecT upstreamLayerGrad;
    ActVecT prevLogits;
    // Last layer
    {
      using ActLastVecT = vector<ActElemT, OUTPUT_DIM>;
      const uint depth = layerData.NUM_BACKBONE_LAYERS; 
      // Calculate downstream gradient of the layer
      const ActLastVecT logits = (ActLastVecT)layerData.m_logitsCache.template getValue<OUTPUT_DIM>(depth);
      ActLastVecT activationError;
      layerData.m_activationLast.backward(activationError, (ActLastVecT)logits);
      const ActLastVecT downstreamLayerGrad = activationError * (ActLastVecT)downstreamGrad;
      // Reproduce the input to the layer from the cache
      prevLogits = (ActVecT)layerData.m_logitsCache.template getValue<HIDDEN_LAYER_DIM>(depth - 1);
      ActVecT layerInput;
      layerData.m_activationHidden.forward(layerInput, (ActVecT)prevLogits);
      // Linear layer backward and calculate upstream gradient of the layer
      LinearLayerT::backward(upstreamLayerGrad, downstreamLayerGrad, layerInput, layerData, depth);
    }
    // Hidden layers
    for (uint depth = layerData.NUM_BACKBONE_LAYERS - 1; depth > 0; --depth) {
      // Calculate downstream gradient of the layer
      const ActVecT logits = prevLogits;
      ActVecT activationError;
      layerData.m_activationHidden.backward(activationError, logits);
      const ActVecT downstreamLayerGrad = activationError * (ActVecT)upstreamLayerGrad;
      // Reproduce the input to the layer from the cache
      prevLogits = (ActVecT)layerData.m_logitsCache.template getValue<HIDDEN_LAYER_DIM>(depth - 1);
      ActVecT layerInput;
      layerData.m_activationHidden.forward(layerInput, (ActVecT)prevLogits);
      // Linear layer backward and calculate upstream gradient of the layer
      LinearLayerT::backward(upstreamLayerGrad, downstreamLayerGrad, layerInput, layerData, depth);
    }
    // First layer
    vector<AccumElemT, INPUT_DIM> upstreamLayerGradIn;
    {
      const uint depth = 0; 
      // Calculate downstream gradient of the layer
      const ActVecT logits = prevLogits;
      ActVecT activationError;
      layerData.m_activationHidden.backward(activationError, logits);
      const ActVecT downstreamLayerGrad = activationError * (ActVecT)upstreamLayerGrad;
      // Linear layer backward and calculate upstream gradient of the layer
      LinearLayerT::backward(upstreamLayerGradIn, downstreamLayerGrad, input, layerData, depth);
    }

    using OutputVecT = vector<OutputElemT, INPUT_DIM>;
    return (OutputVecT)upstreamLayerGradIn;
  }
};

template <>
struct Mlp<1>
{
  static const uint NUM_LAYERS = 1;


  template <// Output
            typename OutputElemT,
            int OUTPUT_DIM,
            // Input
            typename InputElemT,
            int INPUT_DIM,
            // Layer configurations
            int HIDDEN_LAYER_DIM,
            // Weight
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            // Weight gradient cache
            CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
            typename WeightGradientCacheBufferT,
            dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
            // Bias
            CacheMethod BIAS_CACHE_METHOD,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            // Bias gradient cache
            CacheMethod BIAS_GRADIENT_CACHE_METHOD,
            typename BiasGradientCacheBufferT,
            dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
            // Pre-activation
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            CacheMethod LOGITS_CACHE_METHOD,
            typename LogitsCacheBufferT,
            dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
            // Activation functions
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            // Alignments
            uint WEIGHT_MATRIX_ALIGNMENT,
            uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
            uint BIAS_VECTOR_ALIGNMENT>
  static
  void forward(MINIDXNN_OUT(vector<OutputElemT, OUTPUT_DIM>) output,
               MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
               MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData)
  {
    using AccumElemT = typename ComponentTypeTraits<ACCUMULATOR_ELEM_TYPE>::Type;
    using AccumVecT = vector<AccumElemT, OUTPUT_DIM>;
    using ActElemT = typename ComponentTypeTraits<ACTIVATION_ELEM_TYPE>::Type;
    using ActVecT = vector<ActElemT, OUTPUT_DIM>;
    const bool hasBias = BIAS_CACHE_METHOD == CacheMethod::CACHE;
    using LinearLayerT = LinearLayer<hasBias>;

    // Last layer
    {
      const uint depth = 0;
      AccumVecT layerOut;
      LinearLayerT::forward(layerOut, input, layerData, depth);
      layerData.m_activationLast.forward(output, (ActVecT)layerOut);
    }
  }

  template <// Output
            typename OutputElemT,
            int OUTPUT_DIM,
            // Input
            typename InputElemT,
            int INPUT_DIM,
            // Layer configurations
            int HIDDEN_LAYER_DIM,
            // Weight
            typename WeightBufferT,
            dx::linalg::DataType WEIGHT_ELEM_TYPE,
            dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
            // Weight gradient cache
            CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
            typename WeightGradientCacheBufferT,
            dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
            // Bias
            CacheMethod BIAS_CACHE_METHOD,
            typename BiasBufferT,
            dx::linalg::DataType BIAS_ELEM_TYPE,
            // Bias gradient cache
            CacheMethod BIAS_GRADIENT_CACHE_METHOD,
            typename BiasGradientCacheBufferT,
            dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
            // Pre-activation
            dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
            CacheMethod LOGITS_CACHE_METHOD,
            typename LogitsCacheBufferT,
            dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
            // Activation functions
            typename ActivationHiddenT,
            typename ActivationLastT,
            dx::linalg::DataType ACTIVATION_ELEM_TYPE,
            // Alignments
            uint WEIGHT_MATRIX_ALIGNMENT,
            uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
            uint BIAS_VECTOR_ALIGNMENT>
  static
  vector<OutputElemT, INPUT_DIM>
  backward(MINIDXNN_IN(vector<OutputElemT, OUTPUT_DIM>) downstreamGrad,
           MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
           MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData)
  {
    using AccumElemT = typename ComponentTypeTraits<ACCUMULATOR_ELEM_TYPE>::Type;
    using AccumVecT = vector<AccumElemT, INPUT_DIM>;
    using ActElemT = typename ComponentTypeTraits<ACTIVATION_ELEM_TYPE>::Type;
    const bool hasBias = BIAS_CACHE_METHOD == CacheMethod::CACHE;
    using LinearLayerT = LinearLayer<hasBias>;

    AccumVecT upstreamLayerGrad;
    // Last layer
    {
      using ActLastVecT = vector<ActElemT, OUTPUT_DIM>;
      const uint depth = 0; 
      // Calculate downstream gradient of the layer
      const ActLastVecT logits = (ActLastVecT)layerData.m_logitsCache.template getValue<OUTPUT_DIM>(depth);
      ActLastVecT activationError;
      layerData.m_activationLast.backward(activationError, (ActLastVecT)logits);
      const ActLastVecT downstreamLayerGrad = activationError * (ActLastVecT)downstreamGrad;
      // Linear layer backward and calculate upstream gradient of the layer
      LinearLayerT::backward(upstreamLayerGrad, downstreamLayerGrad, input, layerData, depth);
    }

    using OutputVecT = vector<OutputElemT, INPUT_DIM>;
    return (OutputVecT)upstreamLayerGrad;
  }
};

template <>
struct Mlp<0>
{
  // MLP with 0 layers is not valid
};

} // impl

// ============================================================================
// Activation functions
// ============================================================================
//
// Each activation provides:
//   forward(out output, input)  — element-wise activation
//   backward(out gradient, input) — element-wise derivative (for training)
//
// Custom activations: any struct with matching forward/backward signatures
// can be used as the ActivationHiddenT or ActivationLastT template argument.

struct IdentityActivation
{
  template <typename OutputElemT, typename InputElemT, int N>
  void forward(MINIDXNN_OUT(vector<OutputElemT, N>) output,
               MINIDXNN_IN(vector<InputElemT, N>) input)
  {
    using OutputVecT = vector<OutputElemT, N>;
    output = (OutputVecT)input;
  }

  template <typename OutputElemT, typename InputElemT, int N>
  void backward(MINIDXNN_OUT(vector<OutputElemT, N>) gradient,
                MINIDXNN_IN(vector<InputElemT, N>) /* input */)
  {
    using OutputVecT = vector<OutputElemT, N>;
    gradient = (OutputVecT)1;
  }
};

struct SigmoidActivation
{
  template <typename OutputElemT, typename InputElemT, int N>
  void forward(MINIDXNN_OUT(vector<OutputElemT, N>) output,
               MINIDXNN_IN(vector<InputElemT, N>) input)
  {
    using OutputVecT = vector<OutputElemT, N>;

    const OutputVecT e = exp(-abs((OutputVecT)input));
    const OutputElemT one = (OutputElemT)1;
    const OutputVecT oneOverEPlusOne = one / (e + one);
    const OutputElemT zero = (OutputElemT)0;
    output = select(zero > (OutputVecT)input, one - oneOverEPlusOne, oneOverEPlusOne);
  }

  template <typename OutputElemT, typename InputElemT, int N>
  void backward(MINIDXNN_OUT(vector<OutputElemT, N>) gradient,
                MINIDXNN_IN(vector<InputElemT, N>) input)
  {
    using OutputVecT = vector<OutputElemT, N>;

    const OutputVecT e = exp(-abs((OutputVecT)input));
    const OutputElemT one = (OutputElemT)1;
    const OutputVecT oneOverEPlusOne = one / (e + one);
    gradient = (one - oneOverEPlusOne) * oneOverEPlusOne;
  }
};

struct ReluActivation
{
  template <typename OutputElemT, typename InputElemT, int N>
  void forward(MINIDXNN_OUT(vector<OutputElemT, N>) output,
               MINIDXNN_IN(vector<InputElemT, N>) input)
  {
    using InputVecT = vector<InputElemT, N>;
    using OutputVecT = vector<OutputElemT, N>;

    const InputVecT zero = (InputVecT)0;
    const InputVecT y = max(zero, input);
    output = (OutputVecT)y;
  }

  template <typename OutputElemT, typename InputElemT, int N>
  void backward(MINIDXNN_OUT(vector<OutputElemT, N>) gradient,
                MINIDXNN_IN(vector<InputElemT, N>) input)
  {
    using OutputVecT = vector<OutputElemT, N>;
    const InputElemT zero = (InputElemT)0;
    const InputElemT one = (InputElemT)1;
    gradient = select(input < zero, (OutputVecT)zero, (OutputVecT)one);
  }
};

struct LeakyReluActivation
{
  template <typename OutputElemT, typename InputElemT, int N>
  void forward(MINIDXNN_OUT(vector<OutputElemT, N>) output,
               MINIDXNN_IN(vector<InputElemT, N>) input)
  {
    using InputVecT = vector<InputElemT, N>;
    using OutputVecT = vector<OutputElemT, N>;

    const InputElemT negativeSlope = (InputElemT)0.01; // Must be in [0, 1]
    const InputVecT y = max(negativeSlope * input, input);
    output = (OutputVecT)y;
  }

  template <typename OutputElemT, typename InputElemT, int N>
  void backward(MINIDXNN_OUT(vector<OutputElemT, N>) gradient,
                MINIDXNN_IN(vector<InputElemT, N>) input)
  {
    using OutputVecT = vector<OutputElemT, N>;

    const InputElemT zero = (InputElemT)0;
    const InputElemT one = (InputElemT)1;
    const InputElemT negativeSlope = (InputElemT)0.01f;
    gradient = select(input < zero, (OutputVecT)negativeSlope, (OutputVecT)one);
  }
};

// ============================================================================
// Buffer storage — MatrixData and VectorData (cache / no-cache specializations)
// ============================================================================
//
// CACHE:    holds a buffer reference and computes per-layer offsets with alignment
// NO_CACHE: no-op stubs used when a particular buffer kind is not needed

// TransposeResolver: For RowMajor/ColumnMajor layouts,
// since transpose is not supported in cooperative vector, eliminates IS_TRANSPOSED
// by swapping the layout instead (transposing row-major ≡ column-major and vice versa).
// For other layouts (MulOptimal, OuterProductOptimal), LAYOUT and IS_TRANSPOSED
// are passed through unchanged.
template <dx::linalg::MatrixLayout LAYOUT, bool IS_TRANSPOSED>
struct TransposeResolver
{
  static const dx::linalg::MatrixLayout EFFECTIVE_LAYOUT = LAYOUT;
  static const bool EFFECTIVE_TRANSPOSED = IS_TRANSPOSED;
};

template <>
struct TransposeResolver<dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, true>
{
  static const dx::linalg::MatrixLayout EFFECTIVE_LAYOUT = dx::linalg::MATRIX_LAYOUT_COLUMN_MAJOR;
  static const bool EFFECTIVE_TRANSPOSED = false;
};

template <>
struct TransposeResolver<dx::linalg::MATRIX_LAYOUT_COLUMN_MAJOR, true>
{
  static const dx::linalg::MatrixLayout EFFECTIVE_LAYOUT = dx::linalg::MATRIX_LAYOUT_ROW_MAJOR;
  static const bool EFFECTIVE_TRANSPOSED = false;
};

template <>
struct MatrixData<CacheMethod::NO_CACHE>
{
  template <typename BufferT,
            dx::linalg::DataType ELEM_TYPE,
            dx::linalg::MatrixLayout LAYOUT,
            uint NUM_LAYERS,
            uint ALIGNMENT,
            uint VECTOR_STRIDE_ALIGNMENT>
  struct Ref
  {
    using LayoutT = impl::MatrixDataLayout<ELEM_TYPE, LAYOUT>;
    using ElemType = typename LayoutT::Type;
    template <uint ROW_SIZE, uint COLUMN_SIZE, bool IS_TRANSPOSED>
    using DxMatrixRef = dx::linalg::MatrixRefImpl<BufferT, ELEM_TYPE, ROW_SIZE, COLUMN_SIZE,
        TransposeResolver<LAYOUT, IS_TRANSPOSED>::EFFECTIVE_LAYOUT,
        TransposeResolver<LAYOUT, IS_TRANSPOSED>::EFFECTIVE_TRANSPOSED>;


    void set(BufferT /* buffer */, const uint2 /* matrixStride */, const uint /* startOffset */)
    {
      // No cache
    }

    template <uint ROW_SIZE, uint COLUMN_SIZE, bool IS_TRANSPOSED = false>
    DxMatrixRef<ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED>
    makeDxMatrixRef(const uint /* layerIndex */)
    {
      DxMatrixRef<ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED> ref;
      return ref;
    }
  };
};

template <>
struct MatrixData<CacheMethod::CACHE>
{
  template <typename BufferT,
            dx::linalg::DataType ELEM_TYPE,
            dx::linalg::MatrixLayout LAYOUT,
            uint NUM_LAYERS,
            uint ALIGNMENT,
            uint VECTOR_STRIDE_ALIGNMENT>
  struct Ref
  {
    using __BaseT = typename MatrixData<CacheMethod::NO_CACHE>::template Ref<BufferT, ELEM_TYPE, LAYOUT, NUM_LAYERS, ALIGNMENT, VECTOR_STRIDE_ALIGNMENT>;
    using LayoutT = typename __BaseT::LayoutT;
    using ElemType = typename __BaseT::ElemType;
    template <uint ROW_SIZE, uint COLUMN_SIZE, bool IS_TRANSPOSED>
    using DxMatrixRef = typename __BaseT::template DxMatrixRef<ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED>;


    template <uint ROW_SIZE, uint COLUMN_SIZE, bool IS_TRANSPOSED = false>
    static uint getVectorStride()
    {
      uint stride = 0;
      if (LayoutT::IS_ROW_MAJOR || LayoutT::IS_COLUMN_MAJOR) {
        const uint s = LayoutT::template getMinorSize<ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED>();
        stride = impl::align(s * sizeof(ElemType), VECTOR_STRIDE_ALIGNMENT);
      }
      return stride;
    }

    uint getOffset(const uint layerIndex)
    {
      const bool isBehindFirstLayer = layerIndex > 0;
      const bool isBehindSecondLayer = layerIndex > 1;
      const uint offset = m_startOffset +
          (isBehindFirstLayer ? impl::align(m_size[0], ALIGNMENT) : 0u) +
          (isBehindSecondLayer ? impl::align(m_size[1], ALIGNMENT) * (layerIndex - 1) : 0u);
      return offset;
    }

    void set(BufferT buffer, const uint2 matrixSize, const uint startOffset)
    {
      m_buffer = buffer;
      m_size = matrixSize;
      m_startOffset = startOffset;
    }

    template <uint ROW_SIZE, uint COLUMN_SIZE, bool IS_TRANSPOSED = false>
    DxMatrixRef<ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED>
    makeDxMatrixRef(const uint layerIndex)
    {
      const uint vectorStride = getVectorStride<ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED>();
      const uint offset = getOffset(layerIndex);
      DxMatrixRef<ROW_SIZE, COLUMN_SIZE, IS_TRANSPOSED> ref = {m_buffer, offset, vectorStride};
      return ref;
    }


    BufferT m_buffer;
    uint32_t m_startOffset;
    uint32_t2 m_size; // The list of matrix size of first and hidden layers
  };
};

template <>
struct VectorData<CacheMethod::NO_CACHE>
{
  template <typename BufferT,
            dx::linalg::DataType ELEM_TYPE,
            uint NUM_LAYERS,
            uint HIDDEN_LAYER_DIM,
            uint ALIGNMENT>
  struct Ref
  {
    using ElemType = typename impl::ComponentTypeTraits<ELEM_TYPE>::Type;
    using DxVectorRef = dx::linalg::VectorRefImpl<BufferT, ELEM_TYPE>;


    template <int DIM>
    void setValue(MINIDXNN_IN(vector<ElemType, DIM>) /* value */, const uint /* layerIndex */)
    {
    }

    void set(BufferT /* buffer */, const uint /* startOffset */)
    {
    }

    DxVectorRef makeDxVectorRef(const uint /* layerIndex */)
    {
      DxVectorRef ref;
      return ref;
    }
  };
};

template <>
struct VectorData<CacheMethod::CACHE>
{
  template <typename BufferT,
            dx::linalg::DataType ELEM_TYPE,
            uint NUM_LAYERS,
            uint HIDDEN_LAYER_DIM,
            uint ALIGNMENT>
  struct Ref
  {
    using __BaseT = typename VectorData<CacheMethod::NO_CACHE>::template Ref<BufferT, ELEM_TYPE, NUM_LAYERS, HIDDEN_LAYER_DIM, ALIGNMENT>;
    using ElemType = typename __BaseT::ElemType;
    using VectorBufferAccessorT = impl::VectorBufferAccessor<ElemType>;
    using DxVectorRef = typename __BaseT::DxVectorRef;


    static uint getStride()
    {
      return impl::align(HIDDEN_LAYER_DIM * sizeof(ElemType), ALIGNMENT);
    }

    uint getOffset(const uint layerIndex)
    {
      const uint stride = getStride();
      const uint extraOffset = layerIndex * stride;
      const uint offset = m_startOffset + extraOffset;
      return offset;
    }

    template <int DIM>
    vector<ElemType, DIM> getValue(const uint layerIndex)
    {
      const uint offset = getOffset(layerIndex);
      return VectorBufferAccessorT::template load<DIM>(m_buffer, offset);
    }

    template <int DIM>
    void setValue(MINIDXNN_IN(vector<ElemType, DIM>) value, const uint layerIndex)
    {
      const uint offset = getOffset(layerIndex);
      VectorBufferAccessorT::template store<DIM>(m_buffer, offset, value);
    }

    void set(BufferT buffer, const uint startOffset)
    {
      m_buffer = buffer;
      m_startOffset = startOffset;
    }

    DxVectorRef makeDxVectorRef(const uint layerIndex)
    {
      const uint offset = getOffset(layerIndex);
      DxVectorRef ref = {m_buffer, offset};
      return ref;
    }

    BufferT m_buffer;
    uint32_t m_startOffset;
  };
};

// ============================================================================
// forward() / backward() — implementation (delegates to impl::Mlp)
// ============================================================================

template <// Output
          typename OutputElemT,
          int OUTPUT_DIM,
          // Input
          typename InputElemT,
          int INPUT_DIM,
          // Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Weight gradient cache
          CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
          typename WeightGradientCacheBufferT,
          dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
          // Bias
          CacheMethod BIAS_CACHE_METHOD,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE,
          // Bias gradient cache
          CacheMethod BIAS_GRADIENT_CACHE_METHOD,
          typename BiasGradientCacheBufferT,
          dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
          CacheMethod LOGITS_CACHE_METHOD,
          typename LogitsCacheBufferT,
          dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT,
          typename ActivationLastT,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
          uint BIAS_VECTOR_ALIGNMENT>
void forward(MINIDXNN_OUT(vector<OutputElemT, OUTPUT_DIM>) output,
             MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
             MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData)
{
  impl::Mlp<NUM_LAYERS>::forward(output, input, layerData);
}

template <// Output
          typename OutputElemT,
          int OUTPUT_DIM,
          // Input
          typename InputElemT,
          int INPUT_DIM,
          // Layer configurations
          uint NUM_LAYERS,
          int HIDDEN_LAYER_DIM,
          // Weight
          typename WeightBufferT,
          dx::linalg::DataType WEIGHT_ELEM_TYPE,
          dx::linalg::MatrixLayout WEIGHT_MATRIX_LAYOUT,
          // Weight gradient cache
          CacheMethod WEIGHT_GRADIENT_CACHE_METHOD,
          typename WeightGradientCacheBufferT,
          dx::linalg::DataType WEIGHT_GRADIENT_CACHE_ELEM_TYPE,
          // Bias
          CacheMethod BIAS_CACHE_METHOD,
          typename BiasBufferT,
          dx::linalg::DataType BIAS_ELEM_TYPE,
          // Bias gradient cache
          CacheMethod BIAS_GRADIENT_CACHE_METHOD,
          typename BiasGradientCacheBufferT,
          dx::linalg::DataType BIAS_GRADIENT_CACHE_ELEM_TYPE,
          // Pre-activation
          dx::linalg::DataType ACCUMULATOR_ELEM_TYPE,
          CacheMethod LOGITS_CACHE_METHOD,
          typename LogitsCacheBufferT,
          dx::linalg::DataType LOGITS_CACHE_ELEM_TYPE,
          // Activation functions
          typename ActivationHiddenT,
          typename ActivationLastT,
          dx::linalg::DataType ACTIVATION_ELEM_TYPE,
          // Alignments
          uint WEIGHT_MATRIX_ALIGNMENT,
          uint WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
          uint BIAS_VECTOR_ALIGNMENT>
vector<OutputElemT, INPUT_DIM>
backward(MINIDXNN_IN(vector<OutputElemT, OUTPUT_DIM>) downstreamGrad,
         MINIDXNN_IN(vector<InputElemT, INPUT_DIM>) input,
         MINIDXNN_INOUT(LayerDataRefImpl<NUM_LAYERS, HIDDEN_LAYER_DIM, CacheMethod::CACHE, WeightBufferT, WEIGHT_ELEM_TYPE, WEIGHT_MATRIX_LAYOUT, WEIGHT_GRADIENT_CACHE_METHOD, WeightGradientCacheBufferT, WEIGHT_GRADIENT_CACHE_ELEM_TYPE, BIAS_CACHE_METHOD, BiasBufferT, BIAS_ELEM_TYPE, BIAS_GRADIENT_CACHE_METHOD, BiasGradientCacheBufferT, BIAS_GRADIENT_CACHE_ELEM_TYPE, ACCUMULATOR_ELEM_TYPE, LOGITS_CACHE_METHOD, LogitsCacheBufferT, LOGITS_CACHE_ELEM_TYPE, ActivationHiddenT, ActivationLastT, ACTIVATION_ELEM_TYPE, WEIGHT_MATRIX_ALIGNMENT, WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT, BIAS_VECTOR_ALIGNMENT>) layerData)
{
  return impl::Mlp<NUM_LAYERS>::backward(downstreamGrad, input, layerData);
}

} // mininn

#endif // MINIDXNN_MLP_HLSL
