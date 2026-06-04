/*!
  \file cpp_fallback_path.hpp
  \author Sho Ikeda
  \brief C++ fallback kernel dispatch functions for unit tests
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This header is included from unittest.cpp inside the anonymous namespace.
  It depends on types and headers already included by unittest.cpp:
    - common/cpp_fallback.hpp, kernel/mlp_test_common.hlsl
    - test::CppFallbackTest from test.hpp
*/

#ifndef MINIDXNN_UNITTEST_CPP_FALLBACK_PATH_HPP
#define MINIDXNN_UNITTEST_CPP_FALLBACK_PATH_HPP 1

// TODO. Fix me
constexpr size_t BIAS_VECTOR_ALIGNMENT = 128;

// ============================================================================
// C++ fallback helper functions
// These implement the kernel-equivalent operations using the mlp.hlsl C++ path.
// ============================================================================

template <ex::Arithmetic Type, int ROW_SIZE, int COLUMN_SIZE, bool IS_TRANSPOSED, bool HAS_BIAS>
auto cppFallbackLinearAlgebraMul(const std::vector<std::uint8_t>& weightBuf,
                                 const size_t vectorStride,
                                 const std::vector<std::uint8_t>& biasBuf,
                                 const std::vector<Type>& inputVec,
                                 std::vector<Type>& outputVec) -> void
{
  constexpr dx::linalg::DataType DT = ex::DxLinalgDataTypeOf<Type>::value;
  using Resolver = mininn::TransposeResolver<dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, IS_TRANSPOSED>;
  using MatrixRefT = mininn::impl::MatrixRefImpl<ByteAddressBuffer, DT, ROW_SIZE, COLUMN_SIZE, Resolver::EFFECTIVE_LAYOUT, Resolver::EFFECTIVE_TRANSPOSED>;

  ByteAddressBuffer wBuf{weightBuf};
  MatrixRefT matrix = {wBuf, 0, static_cast<uint>(vectorStride)};

  ::vector<Type, COLUMN_SIZE> input{};
  for (size_t d = 0; d < static_cast<size_t>(COLUMN_SIZE); ++d)
    input[d] = inputVec[d];

  ::vector<Type, ROW_SIZE> result{};

  if constexpr (HAS_BIAS) {
    ByteAddressBuffer bBuf{biasBuf};
    mininn::impl::VectorRefImpl<ByteAddressBuffer, DT, BIAS_VECTOR_ALIGNMENT> biasRef = {bBuf, 0};
    result = mininn::impl::LinearAlgebra::mulAdd<Type>(matrix, input, biasRef);
  } else {
    result = mininn::impl::LinearAlgebra::mul<Type>(matrix, input);
  }

  outputVec.resize(ROW_SIZE);
  for (size_t d = 0; d < static_cast<size_t>(ROW_SIZE); ++d)
    outputVec[d] = result[d];
}

template <ex::Arithmetic Type, int SIZE>
auto cppFallbackVectorAcc(RWByteAddressBuffer& outputBuf, const size_t numTasks) -> void
{
  constexpr dx::linalg::DataType DT = mininn::impl::TypeTraits<Type>::COMPONENT_TYPE;

  const uint numThreads = std::max(1u, std::thread::hardware_concurrency());
  const uint totalTasks = static_cast<uint>(numTasks);
  const uint tasksPerThread = totalTasks / numThreads;
  const uint remainder = totalTasks % numThreads;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  uint taskStart = 0;
  for (uint t = 0; t < numThreads; ++t) {
    const uint taskEnd = taskStart + tasksPerThread + (t < remainder ? 1 : 0);
    threads.emplace_back([&, taskStart, taskEnd]() {
      for (uint task = taskStart; task < taskEnd; ++task) {
        ::vector<Type, SIZE> input{};
        for (size_t i = 0; i < static_cast<size_t>(SIZE); ++i)
          input[i] = static_cast<Type>(static_cast<float>(1u << static_cast<unsigned>(i)));

        mininn::impl::RWVectorRef<DT, BIAS_VECTOR_ALIGNMENT> output = {outputBuf, 0};
        mininn::impl::LinearAlgebra::vectorAcc(input, output);
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

template <ex::Arithmetic Type>
auto cppFallbackAtomicFetchAdd(RWByteAddressBuffer& outputBuf, const size_t numTasks, const size_t numIterations) -> void
{
  const uint numThreads = std::max(1u, std::thread::hardware_concurrency());
  const uint totalTasks = static_cast<uint>(numTasks);
  const uint tasksPerThread = totalTasks / numThreads;
  const uint remainder = totalTasks % numThreads;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  uint taskStart = 0;
  for (uint t = 0; t < numThreads; ++t) {
    const uint taskEnd = taskStart + tasksPerThread + (t < remainder ? 1 : 0);
    threads.emplace_back([&, taskStart, taskEnd]() {
      for (uint task = taskStart; task < taskEnd; ++task) {
        for (size_t iter = 0; iter < numIterations; ++iter) {
          // Slot 0-3: constant accumulation (basic correctness)
          mininn::impl::atomicFetchAdd(outputBuf, 0 * sizeof(Type), static_cast<Type>(1));
          mininn::impl::atomicFetchAdd(outputBuf, 1 * sizeof(Type), static_cast<Type>(2));
          mininn::impl::atomicFetchAdd(outputBuf, 2 * sizeof(Type), static_cast<Type>(4));
          mininn::impl::atomicFetchAdd(outputBuf, 3 * sizeof(Type), static_cast<Type>(8));
          // Slot 4-5: small fractional values
          mininn::impl::atomicFetchAdd(outputBuf, 4 * sizeof(Type), static_cast<Type>(0.125));
          mininn::impl::atomicFetchAdd(outputBuf, 5 * sizeof(Type), static_cast<Type>(0.0625));
          // Slot 6: alternating sign by thread ID
          const Type sign = (task & 1u) ? static_cast<Type>(-1) : static_cast<Type>(1);
          mininn::impl::atomicFetchAdd(outputBuf, 6 * sizeof(Type), sign);
          // Slot 7: non-power-of-two value
          mininn::impl::atomicFetchAdd(outputBuf, 7 * sizeof(Type), static_cast<Type>(3));
        }
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

template <ex::Arithmetic Type, int ROW_SIZE, int COLUMN_SIZE>
auto cppFallbackOuterProductAcc(RWByteAddressBuffer& outputBuf,
                                const size_t vectorStride,
                                const size_t numTasks) -> void
{
  constexpr dx::linalg::DataType DT = mininn::impl::TypeTraits<Type>::COMPONENT_TYPE;

  const uint numThreads = std::max(1u, std::thread::hardware_concurrency());
  const uint totalTasks = static_cast<uint>(numTasks);
  const uint tasksPerThread = totalTasks / numThreads;
  const uint remainder = totalTasks % numThreads;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  uint taskStart = 0;
  for (uint t = 0; t < numThreads; ++t) {
    const uint taskEnd = taskStart + tasksPerThread + (t < remainder ? 1 : 0);
    threads.emplace_back([&, taskStart, taskEnd]() {
      for (uint task = taskStart; task < taskEnd; ++task) {
        ::vector<Type, ROW_SIZE> lhs{};
        ::vector<Type, COLUMN_SIZE> rhs{};
        for (size_t i = 0; i < static_cast<size_t>(ROW_SIZE); ++i)
          lhs[i] = static_cast<Type>(static_cast<float>(i + 1));
        for (size_t j = 0; j < static_cast<size_t>(COLUMN_SIZE); ++j)
          rhs[j] = static_cast<Type>(1.0f);

        mininn::impl::RWMatrixRef<DT, static_cast<uint>(ROW_SIZE), static_cast<uint>(COLUMN_SIZE),
            dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, false> matrix = {outputBuf, 0, static_cast<uint>(vectorStride)};
        mininn::impl::LinearAlgebra::outerProductAcc(lhs, rhs, matrix);
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

// ============================================================================
// C++ fallback kernel dispatch (MLP forward/backward)
// ============================================================================

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM,
          typename ActivationHiddenT, typename ActivationLastT, bool HAS_BIAS>
auto cppFallbackForwardKernel(ex::PackedMlpBuffers<Type>& packed,
                              const std::vector<Type>& inputs,
                              std::vector<Type>& outputs,
                              size_t numTasks) -> void
{
  constexpr dx::linalg::DataType DT = ex::DxLinalgDataTypeOf<Type>::value;

  ByteAddressBuffer inputBuf{inputs};
  RWByteAddressBuffer outputBuf{outputs};

  for (uint task = 0; task < static_cast<uint>(numTasks); ++task) {
    testkernel::inferenceStep<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM,
        DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
        128, 16, 128, HAS_BIAS>(
        task, inputBuf, outputBuf, packed.weightBAB(), packed.biasBAB(),
        packed.matrixSizes, static_cast<uint>(numTasks));
  }
}

// Dispatch activation types at runtime
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM, bool HAS_BIAS>
auto dispatchForwardActivation(ex::ActivationType hiddenAct,
                               ex::ActivationType lastAct,
                               ex::PackedMlpBuffers<Type>& packed,
                               const std::vector<Type>& inputs,
                               std::vector<Type>& outputs,
                               size_t numTasks) -> bool
{
  #define DISPATCH_ACT(HiddenT, LastT, hiddenE, lastE) \
    if (hiddenAct == (hiddenE) && lastAct == (lastE)) { \
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM, HiddenT, LastT, HAS_BIAS>(packed, inputs, outputs, numTasks); \
      return true; \
    }

  using namespace mininn;
  DISPATCH_ACT(IdentityActivation,  IdentityActivation,  ex::ActivationType::IDENTITY, ex::ActivationType::IDENTITY)
  DISPATCH_ACT(IdentityActivation,  SigmoidActivation,   ex::ActivationType::IDENTITY, ex::ActivationType::SIGMOID)
  DISPATCH_ACT(ReluActivation,      IdentityActivation,  ex::ActivationType::RELU,     ex::ActivationType::IDENTITY)
  DISPATCH_ACT(ReluActivation,      SigmoidActivation,   ex::ActivationType::RELU,     ex::ActivationType::SIGMOID)
  DISPATCH_ACT(SigmoidActivation,   SigmoidActivation,   ex::ActivationType::SIGMOID,  ex::ActivationType::SIGMOID)
  DISPATCH_ACT(LeakyReluActivation, SigmoidActivation,   ex::ActivationType::LEAKY_RELU, ex::ActivationType::SIGMOID)
  DISPATCH_ACT(LeakyReluActivation, IdentityActivation,  ex::ActivationType::LEAKY_RELU, ex::ActivationType::IDENTITY)

  #undef DISPATCH_ACT
  return false;
}

// Dispatch NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM at runtime
template <ex::Arithmetic Type>
auto dispatchForward(size_t numLayers, size_t hiddenDim, size_t inputDim, size_t outputDim,
                     ex::ActivationType hiddenAct, ex::ActivationType lastAct,
                     ex::PackedMlpBuffers<Type>& packed,
                     const std::vector<Type>& inputs,
                     std::vector<Type>& outputs,
                     size_t numTasks,
                     bool hasBias = true) -> bool
{
  #define DISPATCH_FWD(NL, HD, ID, OD) \
    if (numLayers == (NL) && hiddenDim == (HD) && inputDim == (ID) && outputDim == (OD)) { \
      if (hasBias) \
        return dispatchForwardActivation<Type, (NL), (HD), (ID), (OD), true>(hiddenAct, lastAct, packed, inputs, outputs, numTasks); \
      else \
        return dispatchForwardActivation<Type, (NL), (HD), (ID), (OD), false>(hiddenAct, lastAct, packed, inputs, outputs, numTasks); \
    }

  // Single layer (numBackboneLayers=0)
  DISPATCH_FWD(1, 2, 2, 2)
  DISPATCH_FWD(1, 4, 2, 4)
  DISPATCH_FWD(1, 16, 16, 4)
  DISPATCH_FWD(1, 16, 16, 16)
  // 2 layers (1 backbone)
  DISPATCH_FWD(2, 2, 2, 2)
  DISPATCH_FWD(2, 8, 2, 4)
  // 3 layers (2 backbone)
  DISPATCH_FWD(3, 8, 2, 4)
  DISPATCH_FWD(3, 16, 8, 4)
  // 4 layers (3 backbone)
  DISPATCH_FWD(4, 6, 2, 4)
  DISPATCH_FWD(4, 8, 2, 4)

  #undef DISPATCH_FWD
  return false;
}

// ============================================================================
// Backward test dispatch
// ============================================================================

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM,
          typename ActivationHiddenT, typename ActivationLastT, bool HAS_BIAS>
auto cppFallbackBackwardKernel(ex::PackedMlpBuffers<Type>& packed,
                               const std::vector<Type>& inputs,
                               const std::vector<Type>& targets,
                               std::vector<Type>& outputs,
                               std::vector<std::uint8_t>& weightGradBuf,
                               std::vector<std::uint8_t>& biasGradBuf,
                               size_t batchSize) -> void
{
  constexpr dx::linalg::DataType DT = ex::DxLinalgDataTypeOf<Type>::value;

  const size_t logitsStride = ex::alignBytes(static_cast<size_t>(HIDDEN_DIM) * sizeof(Type), ex::VECTOR_ALIGNMENT);
  const size_t logitsPerSample = logitsStride * NUM_LAYERS;
  const size_t logitsBufSize = logitsPerSample * batchSize;

  weightGradBuf.assign(packed.weightBuf.size(), 0);
  biasGradBuf.assign(packed.biasBuf.size(), 0);
  std::vector<std::uint8_t> logitsBuf(logitsBufSize, 0);

  ByteAddressBuffer inputBuf{inputs};
  ByteAddressBuffer targetBuf{targets};
  RWByteAddressBuffer outputBuf{outputs};

  RWByteAddressBuffer wGradBAB{weightGradBuf};
  RWByteAddressBuffer bGradBAB{biasGradBuf};
  RWByteAddressBuffer logitsBAB{logitsBuf};

  // Forward pass for all samples
  for (uint s = 0; s < static_cast<uint>(batchSize); ++s) {
    testkernel::trainingForwardStep<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM,
        DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
        128, 16, 128, HAS_BIAS>(
        s, inputBuf, outputBuf, packed.weightBAB(), packed.biasBAB(),
        logitsBAB, packed.matrixSizes,
        static_cast<uint>(batchSize), static_cast<uint>(logitsPerSample));
  }

  // Backward pass for all samples
  RWByteAddressBuffer outputReadBuf{outputs};

  for (uint s = 0; s < static_cast<uint>(batchSize); ++s) {
    testkernel::trainingBackwardStep<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM,
        DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
        128, 16, 128, HAS_BIAS>(
        s, inputBuf, targetBuf, outputReadBuf, packed.weightBAB(), packed.biasBAB(),
        wGradBAB, bGradBAB, logitsBAB, packed.matrixSizes,
        static_cast<uint>(batchSize), static_cast<uint>(logitsPerSample));
  }
}

// Dispatch activation types for backward
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM, int INPUT_DIM, int OUTPUT_DIM, bool HAS_BIAS>
auto dispatchBackwardActivation(ex::ActivationType hiddenAct,
                                ex::ActivationType lastAct,
                                ex::PackedMlpBuffers<Type>& packed,
                                const std::vector<Type>& inputs,
                                const std::vector<Type>& targets,
                                std::vector<Type>& outputs,
                                std::vector<std::uint8_t>& weightGradBuf,
                                std::vector<std::uint8_t>& biasGradBuf,
                                size_t batchSize) -> bool
{
  #define DISPATCH_BACK_ACT(HiddenT, LastT, hiddenE, lastE) \
    if (hiddenAct == (hiddenE) && lastAct == (lastE)) { \
      cppFallbackBackwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM, HiddenT, LastT, HAS_BIAS>( \
          packed, inputs, targets, outputs, weightGradBuf, biasGradBuf, batchSize); \
      return true; \
    }

  using namespace mininn;
  DISPATCH_BACK_ACT(LeakyReluActivation, SigmoidActivation, ex::ActivationType::LEAKY_RELU, ex::ActivationType::SIGMOID)
  DISPATCH_BACK_ACT(IdentityActivation,  IdentityActivation, ex::ActivationType::IDENTITY,  ex::ActivationType::IDENTITY)
  DISPATCH_BACK_ACT(ReluActivation,      SigmoidActivation,  ex::ActivationType::RELU,      ex::ActivationType::SIGMOID)

  #undef DISPATCH_BACK_ACT
  return false;
}

// Dispatch NUM_LAYERS, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM for backward
template <ex::Arithmetic Type>
auto dispatchBackward(size_t numLayers, size_t hiddenDim, size_t inputDim, size_t outputDim,
                      ex::ActivationType hiddenAct, ex::ActivationType lastAct,
                      ex::PackedMlpBuffers<Type>& packed,
                      const std::vector<Type>& inputs,
                      const std::vector<Type>& targets,
                      std::vector<Type>& outputs,
                      std::vector<std::uint8_t>& weightGradBuf,
                      std::vector<std::uint8_t>& biasGradBuf,
                      size_t batchSize,
                      bool hasBias = true) -> bool
{
  #define DISPATCH_BACK(NL, HD, ID, OD) \
    if (numLayers == (NL) && hiddenDim == (HD) && inputDim == (ID) && outputDim == (OD)) { \
      if (hasBias) \
        return dispatchBackwardActivation<Type, (NL), (HD), (ID), (OD), true>(hiddenAct, lastAct, packed, inputs, targets, outputs, weightGradBuf, biasGradBuf, batchSize); \
      else \
        return dispatchBackwardActivation<Type, (NL), (HD), (ID), (OD), false>(hiddenAct, lastAct, packed, inputs, targets, outputs, weightGradBuf, biasGradBuf, batchSize); \
    }

  // Single layer (numBackboneLayers=0): inputDim=2, outputDim=4
  DISPATCH_BACK(1, 4, 2, 4)
  // 2 layers (1 backbone)
  DISPATCH_BACK(2, 8, 2, 4)
  // 3 layers (2 backbone)
  DISPATCH_BACK(3, 8, 2, 4)
  // 4 layers (3 backbone)
  DISPATCH_BACK(4, 6, 2, 4)
  DISPATCH_BACK(4, 8, 2, 4)

  #undef DISPATCH_BACK
  return false;
}

// ============================================================================
// Smoke test
// ============================================================================

// Smoke test: identity weights with sigmoid activation (verifies basic mlp.hlsl C++ path)
template <typename Type>
auto testCppFallbackForwardSmoke(const CppFallbackTest& /* test */) -> void
{
  constexpr unsigned int NUM_LAYERS = 1;
  constexpr int DIM = 2;

  std::vector<std::uint8_t> weightBuf(256, 0);
  std::vector<std::uint8_t> biasBuf(128, 0);

  {
    Type* w0 = reinterpret_cast<Type*>(weightBuf.data());
    w0[0] = Type(1.0f);
    w0[1] = Type(0.0f);
    Type* w1 = reinterpret_cast<Type*>(weightBuf.data() + 16);
    w1[0] = Type(0.0f);
    w1[1] = Type(1.0f);
  }
  {
    Type* b = reinterpret_cast<Type*>(biasBuf.data());
    b[0] = Type(0.1f);
    b[1] = Type(0.2f);
  }

  ByteAddressBuffer wBuf{weightBuf};
  ByteAddressBuffer bBuf{biasBuf};

  constexpr dx::linalg::DataType DT = ex::DxLinalgDataTypeOf<Type>::value;
  using LayerDataRefT = mininn::InferenceLayerDataRef<
    NUM_LAYERS, DIM,
    DT,
    dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
    DT,
    DT,
    mininn::IdentityActivation,
    mininn::SigmoidActivation,
    DT,
    128, 16, 128>;

  LayerDataRefT layerData;
  layerData.setWeightData(wBuf, uint2{128u, 128u}, 0);
  layerData.setBiasData(bBuf, 0);

  vector<Type, DIM> input;
  input[0] = Type(0.5f);
  input[1] = Type(0.3f);
  vector<Type, DIM> output;
  mininn::forward(output, input, layerData);

  const float expected0 = 1.0f / (1.0f + std::exp(-0.6f));
  const float expected1 = 1.0f / (1.0f + std::exp(-0.5f));
  EXPECT_NEAR(static_cast<float>(output[0]), expected0, 0.01f);
  EXPECT_NEAR(static_cast<float>(output[1]), expected1, 0.01f);
}

#endif /* MINIDXNN_UNITTEST_CPP_FALLBACK_PATH_HPP */
