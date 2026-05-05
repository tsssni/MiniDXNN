/*!
  \file cpp_fallback_path.hpp
  \author Sho Ikeda
  \brief C++ fallback training and inference paths for texture MLP with input encoding
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This header is included from example.cpp inside the anonymous namespace.
  It depends on types and functions defined earlier in example.cpp:
    - CliOptions, MlpConfig<Type>, InputEncoding, encodedInputDim,
      inputEncodingFromString, mapToLdr<Type>
  and on headers already included by example.cpp:
    - common/cpp_fallback.hpp, common/mlp_layer.hpp, common/loss.hpp,
      common/optimizer.hpp, common/texture.hpp,
      kernel/input_encoding_common.hlsl,
      kernel/texture_training_with_encoding_common.hlsl,
      kernel/texture_inference_with_encoding_common.hlsl,
      kernel/optimizer.hlsl
*/

#ifndef MINIDXNN_EXAMPLE_03_CPP_FALLBACK_PATH_HPP
#define MINIDXNN_EXAMPLE_03_CPP_FALLBACK_PATH_HPP 1

// ============================================================================
// Packed training buffers (extends PackedMlpBuffers with gradient + optimizer state)
// ============================================================================

template <ex::Arithmetic Type>
struct PackedTrainingBuffers03
{
  std::vector<std::uint8_t> weightBuf;
  std::vector<std::uint8_t> biasBuf;
  std::vector<std::uint8_t> weightGradBuf;
  std::vector<std::uint8_t> biasGradBuf;
  std::vector<std::uint8_t> logitsCacheBuf;
  float lossValue = 0.0f;
  size_t biasStride = 0;
  uint2 matrixSizes{};

  // Grid feature buffers (float32)
  std::vector<std::uint8_t> gridBuf;
  std::vector<std::uint8_t> gridGradBuf;
  std::vector<std::uint8_t> gridFirstMomentBuf;
  std::vector<std::uint8_t> gridSecondMomentBuf;

  // Optimizer state buffers
  std::vector<std::uint8_t> weightFirstMomentBuf;
  std::vector<std::uint8_t> weightSecondMomentBuf;
  std::vector<std::uint8_t> biasFirstMomentBuf;
  std::vector<std::uint8_t> biasSecondMomentBuf;
  size_t timestep = 0;

  ByteAddressBuffer weightBAB() const { return ByteAddressBuffer{weightBuf}; }
  ByteAddressBuffer biasBAB() const { return ByteAddressBuffer{biasBuf}; }
  RWByteAddressBuffer weightRWBAB() { return RWByteAddressBuffer{weightBuf}; }
  RWByteAddressBuffer biasRWBAB() { return RWByteAddressBuffer{biasBuf}; }
  RWByteAddressBuffer weightGradRWBAB() { return RWByteAddressBuffer{weightGradBuf}; }
  ByteAddressBuffer weightGradBAB() const { return ByteAddressBuffer{weightGradBuf}; }
  RWByteAddressBuffer biasGradRWBAB() { return RWByteAddressBuffer{biasGradBuf}; }
  ByteAddressBuffer biasGradBAB() const { return ByteAddressBuffer{biasGradBuf}; }
  RWByteAddressBuffer logitsCacheRWBAB() { return RWByteAddressBuffer{logitsCacheBuf}; }
  RWByteAddressBuffer lossRWBAB()
  {
    return RWByteAddressBuffer{reinterpret_cast<std::uint8_t*>(&lossValue), sizeof(float)};
  }
  ByteAddressBuffer gridBAB() const { return ByteAddressBuffer{gridBuf}; }
  RWByteAddressBuffer gridRWBAB() { return RWByteAddressBuffer{gridBuf}; }
  RWByteAddressBuffer gridGradRWBAB() { return RWByteAddressBuffer{gridGradBuf}; }
  ByteAddressBuffer gridGradBAB() const { return ByteAddressBuffer{gridGradBuf}; }
  RWByteAddressBuffer weightFirstMomentRWBAB() { return RWByteAddressBuffer{weightFirstMomentBuf}; }
  RWByteAddressBuffer weightSecondMomentRWBAB() { return RWByteAddressBuffer{weightSecondMomentBuf}; }
  RWByteAddressBuffer biasFirstMomentRWBAB() { return RWByteAddressBuffer{biasFirstMomentBuf}; }
  RWByteAddressBuffer biasSecondMomentRWBAB() { return RWByteAddressBuffer{biasSecondMomentBuf}; }
  RWByteAddressBuffer gridFirstMomentRWBAB() { return RWByteAddressBuffer{gridFirstMomentBuf}; }
  RWByteAddressBuffer gridSecondMomentRWBAB() { return RWByteAddressBuffer{gridSecondMomentBuf}; }

  auto pack(const std::span<const ex::MlpLayer<Type, Type, Type, Type>> mlpData, const bool hasBias) -> void
  {
    const size_t numLayers = mlpData.size();
    const size_t hiddenDim = (numLayers > 1) ? mlpData[0].outputDimension()
                                              : mlpData[0].inputDimension();
    ex::PackedMlpBuffers<Type> base;
    base.pack(mlpData, hasBias);
    weightBuf = std::move(base.weightBuf);
    biasBuf = std::move(base.biasBuf);
    matrixSizes = base.matrixSizes;
    biasStride = ex::alignBytes(hiddenDim * sizeof(Type), ex::VECTOR_ALIGNMENT);
    weightGradBuf.assign(weightBuf.size(), 0);
    biasGradBuf.assign(biasBuf.size(), 0);
  }

  auto allocateLogitsCache(const size_t batchSize, const size_t numLayers) -> void
  {
    const size_t perSample = biasStride * numLayers;
    logitsCacheBuf.assign(perSample * batchSize, 0);
  }

  auto initializeGrid(const size_t gridElements, const std::vector<float>& gridFeatures) -> void
  {
    gridBuf.resize(gridElements * sizeof(float));
    std::memcpy(gridBuf.data(), gridFeatures.data(), gridBuf.size());
    gridGradBuf.assign(gridBuf.size(), 0);
  }

  auto allocateOptimizerState(const ex::OptimizerType optType, const size_t gridElements) -> void
  {
    const size_t weightElements = weightBuf.size() / sizeof(Type);
    const size_t biasElements = biasBuf.size() / sizeof(Type);
    if (optType == ex::OptimizerType::ADAM) {
      weightFirstMomentBuf.assign(weightElements * sizeof(float), 0);
      weightSecondMomentBuf.assign(weightElements * sizeof(float), 0);
      biasFirstMomentBuf.assign(biasElements * sizeof(float), 0);
      biasSecondMomentBuf.assign(biasElements * sizeof(float), 0);
      if (gridElements > 0) {
        gridFirstMomentBuf.assign(gridElements * sizeof(float), 0);
        gridSecondMomentBuf.assign(gridElements * sizeof(float), 0);
      }
    } else if (optType == ex::OptimizerType::LION) {
      weightFirstMomentBuf.assign(weightElements * sizeof(float), 0);
      biasFirstMomentBuf.assign(biasElements * sizeof(float), 0);
      if (gridElements > 0) {
        gridFirstMomentBuf.assign(gridElements * sizeof(float), 0);
      }
    }
    timestep = 0;
  }

  auto zeroGradients() -> void
  {
    std::ranges::fill(weightGradBuf, std::uint8_t{0});
    std::ranges::fill(biasGradBuf, std::uint8_t{0});
    if (!gridGradBuf.empty())
      std::ranges::fill(gridGradBuf, std::uint8_t{0});
    lossValue = 0.0f;
  }

  auto applyOptimizer(const ex::OptimizerType optType, const float lr, const size_t gridElements,
                      const float adamBeta1, const float adamBeta2, const float adamEpsilon,
                      const float lionBeta1, const float lionBeta2, const float lionWeightDecay,
                      const float invLossScale) -> void
  {
    const uint wSize = static_cast<uint>(weightBuf.size());
    const uint bSize = static_cast<uint>(biasBuf.size());
    const uint gSize = static_cast<uint>(gridElements * sizeof(float));

    switch (optType) {
      case ex::OptimizerType::SGD:
        optimizer::sgdUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(), lr, invLossScale, wSize);
        optimizer::sgdUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(), lr, invLossScale, bSize);
        if (gridElements > 0)
          optimizer::sgdUpdateAll<float>(gridRWBAB(), gridGradRWBAB(), lr, invLossScale, gSize);
        break;
      case ex::OptimizerType::ADAM: {
        ++timestep;
        const float bc1 = 1.0f - std::pow(adamBeta1, static_cast<float>(timestep));
        const float bc2 = 1.0f - std::pow(adamBeta2, static_cast<float>(timestep));
        optimizer::adamUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(),
            weightFirstMomentRWBAB(), weightSecondMomentRWBAB(),
            lr, adamBeta1, adamBeta2, adamEpsilon, bc1, bc2, invLossScale, wSize);
        optimizer::adamUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(),
            biasFirstMomentRWBAB(), biasSecondMomentRWBAB(),
            lr, adamBeta1, adamBeta2, adamEpsilon, bc1, bc2, invLossScale, bSize);
        if (gridElements > 0)
          optimizer::adamUpdateAll<float>(gridRWBAB(), gridGradRWBAB(),
              gridFirstMomentRWBAB(), gridSecondMomentRWBAB(),
              lr, adamBeta1, adamBeta2, adamEpsilon, bc1, bc2, invLossScale, gSize);
        break;
      }
      case ex::OptimizerType::LION: {
        optimizer::lionUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(),
            weightFirstMomentRWBAB(), lr, lionBeta1, lionBeta2, lionWeightDecay, invLossScale, wSize);
        optimizer::lionUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(),
            biasFirstMomentRWBAB(), lr, lionBeta1, lionBeta2, lionWeightDecay, invLossScale, bSize);
        if (gridElements > 0)
          optimizer::lionUpdateAll<float>(gridRWBAB(), gridGradRWBAB(),
              gridFirstMomentRWBAB(), lr, lionBeta1, lionBeta2, lionWeightDecay, invLossScale, gSize);
        break;
      }
      default:
        break;
    }
  }
};

// ============================================================================
// Multi-threaded training kernel dispatch
// ============================================================================

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          typename ActivationHiddenT, typename ActivationLastT,
          int INPUT_DIM, int OUTPUT_DIM, int ENCODING_TYPE, int NUM_FREQUENCIES, int GRID_RESOLUTION>
auto cppFallbackTrainingKernel(PackedTrainingBuffers03<Type>& packed,
                               const std::vector<Type>& uvData,
                               const std::vector<Type>& texelData,
                               const size_t batchSize,
                               const size_t batchIndex,
                               const size_t currentBatchSize,
                               const float lossScale) -> void
{
  constexpr dx::linalg::DataType DT = ex::DxLinalgDataTypeOf<Type>::value;

  ByteAddressBuffer uvBuf{uvData};
  ByteAddressBuffer targetBuf{texelData};

  const uint totalTasks = static_cast<uint>(currentBatchSize);
  const uint numThreads = std::max(1u, std::thread::hardware_concurrency());
  const uint tasksPerThread = totalTasks / numThreads;
  const uint remainder = totalTasks % numThreads;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  uint taskStart = 0;
  for (uint t = 0; t < numThreads; ++t) {
    const uint taskEnd = taskStart + tasksPerThread + (t < remainder ? 1 : 0);
    threads.emplace_back([&, taskStart, taskEnd]() {
      for (uint threadId = taskStart; threadId < taskEnd; ++threadId) {
        texkernel::trainingStepWithEncoding<Type, NUM_LAYERS, HIDDEN_DIM,
            DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
            128, 16, 64, INPUT_DIM, OUTPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES, GRID_RESOLUTION>(
            threadId, uvBuf, targetBuf, packed.weightBAB(), packed.biasBAB(),
            packed.weightGradRWBAB(), packed.biasGradRWBAB(),
            packed.logitsCacheRWBAB(), packed.lossRWBAB(),
            packed.gridBAB(), packed.gridGradRWBAB(),
            packed.matrixSizes, static_cast<uint>(batchSize),
            static_cast<uint>(batchIndex), static_cast<uint>(currentBatchSize),
            static_cast<uint>(packed.biasStride), lossScale);
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Dispatch activation types for training
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          int INPUT_DIM, int OUTPUT_DIM, int ENCODING_TYPE, int NUM_FREQUENCIES, int GRID_RESOLUTION>
auto dispatchTrainingActivation(const ex::ActivationType hiddenAct,
                                PackedTrainingBuffers03<Type>& packed,
                                const std::vector<Type>& uvData,
                                const std::vector<Type>& texelData,
                                const size_t batchSize,
                                const size_t batchIndex,
                                const size_t currentBatchSize,
                                const float lossScale) -> bool
{
  using Sigmoid = mininn::SigmoidActivation;

  #define DISPATCH_TRAIN_ACT_03(HiddenT, hiddenE) \
    if (hiddenAct == (hiddenE)) { \
      cppFallbackTrainingKernel<Type, NUM_LAYERS, HIDDEN_DIM, HiddenT, Sigmoid, \
          INPUT_DIM, OUTPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES, GRID_RESOLUTION>( \
          packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale); \
      return true; \
    }

  DISPATCH_TRAIN_ACT_03(mininn::IdentityActivation,  ex::ActivationType::IDENTITY)
  DISPATCH_TRAIN_ACT_03(mininn::SigmoidActivation,   ex::ActivationType::SIGMOID)
  DISPATCH_TRAIN_ACT_03(mininn::ReluActivation,       ex::ActivationType::RELU)
  DISPATCH_TRAIN_ACT_03(mininn::LeakyReluActivation,  ex::ActivationType::LEAKY_RELU)

  #undef DISPATCH_TRAIN_ACT_03
  return false;
}

// Dispatch NUM_LAYERS and HIDDEN_DIM for training
template <ex::Arithmetic Type, int INPUT_DIM, int OUTPUT_DIM, int ENCODING_TYPE, int NUM_FREQUENCIES, int GRID_RESOLUTION>
auto dispatchTraining03(const size_t numLayers, const size_t hiddenDim,
                        const ex::ActivationType hiddenAct,
                        PackedTrainingBuffers03<Type>& packed,
                        const std::vector<Type>& uvData,
                        const std::vector<Type>& texelData,
                        const size_t batchSize,
                        const size_t batchIndex,
                        const size_t currentBatchSize,
                        const float lossScale) -> bool
{
  #define DISPATCH_TRAIN_03(NL, HD) \
    if (numLayers == (NL) && hiddenDim == (HD)) \
      return dispatchTrainingActivation<Type, (NL), (HD), INPUT_DIM, OUTPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES, GRID_RESOLUTION>( \
          hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);

  DISPATCH_TRAIN_03(2, 8)   DISPATCH_TRAIN_03(2, 16)
  DISPATCH_TRAIN_03(2, 32)  DISPATCH_TRAIN_03(2, 64)
  DISPATCH_TRAIN_03(3, 8)   DISPATCH_TRAIN_03(3, 16)
  DISPATCH_TRAIN_03(3, 32)  DISPATCH_TRAIN_03(3, 64)
  DISPATCH_TRAIN_03(4, 8)   DISPATCH_TRAIN_03(4, 16)
  DISPATCH_TRAIN_03(4, 32)  DISPATCH_TRAIN_03(4, 64)
  DISPATCH_TRAIN_03(5, 8)   DISPATCH_TRAIN_03(5, 16)
  DISPATCH_TRAIN_03(5, 32)  DISPATCH_TRAIN_03(5, 64)

  #undef DISPATCH_TRAIN_03
  return false;
}

// ============================================================================
// Multi-threaded inference kernel dispatch
// ============================================================================

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          typename ActivationHiddenT, typename ActivationLastT,
          int INPUT_DIM, int OUTPUT_DIM, int ENCODING_TYPE, int NUM_FREQUENCIES, int GRID_RESOLUTION>
auto cppFallbackInferenceKernel(const PackedTrainingBuffers03<Type>& packed,
                                const std::vector<Type>& uvData,
                                std::vector<Type>& output,
                                const size_t numTasks) -> void
{
  constexpr dx::linalg::DataType DT = ex::DxLinalgDataTypeOf<Type>::value;

  ByteAddressBuffer uvBuf{uvData};
  RWByteAddressBuffer outBuf{output};

  const uint totalTasks = static_cast<uint>(numTasks);
  const uint numThreads = std::max(1u, std::thread::hardware_concurrency());
  const uint tasksPerThread = totalTasks / numThreads;
  const uint remainder = totalTasks % numThreads;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  uint taskStart = 0;
  for (uint t = 0; t < numThreads; ++t) {
    const uint taskEnd = taskStart + tasksPerThread + (t < remainder ? 1 : 0);
    threads.emplace_back([&, taskStart, taskEnd]() {
      for (uint task = taskStart; task < taskEnd; ++task) {
        texkernel::inferenceStepWithEncoding<Type, NUM_LAYERS, HIDDEN_DIM,
            DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
            128, 16, 64, true,
            INPUT_DIM, OUTPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES, GRID_RESOLUTION>(
            task, uvBuf, outBuf, packed.weightBAB(), packed.biasBAB(),
            packed.gridBAB(), packed.matrixSizes, totalTasks);
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Dispatch activation types for inference
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          int INPUT_DIM, int OUTPUT_DIM, int ENCODING_TYPE, int NUM_FREQUENCIES, int GRID_RESOLUTION>
auto dispatchInferenceActivation(const ex::ActivationType hiddenAct,
                                 const PackedTrainingBuffers03<Type>& packed,
                                 const std::vector<Type>& uvData,
                                 std::vector<Type>& output,
                                 const size_t numTasks) -> bool
{
  using Sigmoid = mininn::SigmoidActivation;

  #define DISPATCH_INF_ACT_03(HiddenT, hiddenE) \
    if (hiddenAct == (hiddenE)) { \
      cppFallbackInferenceKernel<Type, NUM_LAYERS, HIDDEN_DIM, HiddenT, Sigmoid, \
          INPUT_DIM, OUTPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES, GRID_RESOLUTION>( \
          packed, uvData, output, numTasks); \
      return true; \
    }

  DISPATCH_INF_ACT_03(mininn::IdentityActivation,  ex::ActivationType::IDENTITY)
  DISPATCH_INF_ACT_03(mininn::SigmoidActivation,   ex::ActivationType::SIGMOID)
  DISPATCH_INF_ACT_03(mininn::ReluActivation,       ex::ActivationType::RELU)
  DISPATCH_INF_ACT_03(mininn::LeakyReluActivation,  ex::ActivationType::LEAKY_RELU)

  #undef DISPATCH_INF_ACT_03
  return false;
}

// Dispatch NUM_LAYERS and HIDDEN_DIM for inference
template <ex::Arithmetic Type, int INPUT_DIM, int OUTPUT_DIM, int ENCODING_TYPE, int NUM_FREQUENCIES, int GRID_RESOLUTION>
auto dispatchInference03(const size_t numLayers, const size_t hiddenDim,
                         const ex::ActivationType hiddenAct,
                         const PackedTrainingBuffers03<Type>& packed,
                         const std::vector<Type>& uvData,
                         std::vector<Type>& output,
                         const size_t numTasks) -> bool
{
  #define DISPATCH_INF_03(NL, HD) \
    if (numLayers == (NL) && hiddenDim == (HD)) \
      return dispatchInferenceActivation<Type, (NL), (HD), INPUT_DIM, OUTPUT_DIM, ENCODING_TYPE, NUM_FREQUENCIES, GRID_RESOLUTION>( \
          hiddenAct, packed, uvData, output, numTasks);

  DISPATCH_INF_03(2, 8)   DISPATCH_INF_03(2, 16)
  DISPATCH_INF_03(2, 32)  DISPATCH_INF_03(2, 64)
  DISPATCH_INF_03(3, 8)   DISPATCH_INF_03(3, 16)
  DISPATCH_INF_03(3, 32)  DISPATCH_INF_03(3, 64)
  DISPATCH_INF_03(4, 8)   DISPATCH_INF_03(4, 16)
  DISPATCH_INF_03(4, 32)  DISPATCH_INF_03(4, 64)
  DISPATCH_INF_03(5, 8)   DISPATCH_INF_03(5, 16)
  DISPATCH_INF_03(5, 32)  DISPATCH_INF_03(5, 64)

  #undef DISPATCH_INF_03
  return false;
}

// ============================================================================
// Runtime encoding dispatch
// ============================================================================

template <ex::Arithmetic Type>
auto dispatchTrainingWithEncoding(const size_t numLayers, const size_t hiddenDim,
                                  const ex::ActivationType hiddenAct,
                                  const InputEncoding encoding,
                                  const size_t positionalFrequencies,
                                  const size_t gridResolution,
                                  const size_t inputDim,
                                  PackedTrainingBuffers03<Type>& packed,
                                  const std::vector<Type>& uvData,
                                  const std::vector<Type>& texelData,
                                  const size_t batchSize,
                                  const size_t batchIndex,
                                  const size_t currentBatchSize,
                                  const float lossScale) -> bool
{
  // Dispatch based on encoding type, input dim, and frequency count
  if (encoding == InputEncoding::NONE) {
    return dispatchTraining03<Type, 2, 4, 0, 1, 1>(
        numLayers, hiddenDim, hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);
  } else if (encoding == InputEncoding::POSITIONAL) {
    // Dispatch based on positional frequencies (common values: 4, 8)
    if (positionalFrequencies == 4 && inputDim == 16) {
      return dispatchTraining03<Type, 16, 4, 1, 4, 1>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);
    } else if (positionalFrequencies == 8 && inputDim == 32) {
      return dispatchTraining03<Type, 32, 4, 1, 8, 1>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);
    }
  } else if (encoding == InputEncoding::GRID) {
    // Dispatch based on grid feature dim and resolution
    if (inputDim == 4 && gridResolution == 32) {
      return dispatchTraining03<Type, 4, 4, 2, 1, 32>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);
    } else if (inputDim == 4 && gridResolution == 64) {
      return dispatchTraining03<Type, 4, 4, 2, 1, 64>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);
    } else if (inputDim == 8 && gridResolution == 32) {
      return dispatchTraining03<Type, 8, 4, 2, 1, 32>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);
    } else if (inputDim == 8 && gridResolution == 64) {
      return dispatchTraining03<Type, 8, 4, 2, 1, 64>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);
    } else if (inputDim == 16 && gridResolution == 32) {
      return dispatchTraining03<Type, 16, 4, 2, 1, 32>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize, lossScale);
    }
  }
  return false;
}

template <ex::Arithmetic Type>
auto dispatchInferenceWithEncoding(const size_t numLayers, const size_t hiddenDim,
                                   const ex::ActivationType hiddenAct,
                                   const InputEncoding encoding,
                                   const size_t positionalFrequencies,
                                   const size_t gridResolution,
                                   const size_t inputDim,
                                   const PackedTrainingBuffers03<Type>& packed,
                                   const std::vector<Type>& uvData,
                                   std::vector<Type>& output,
                                   const size_t numTasks) -> bool
{
  if (encoding == InputEncoding::NONE) {
    return dispatchInference03<Type, 2, 4, 0, 1, 1>(
        numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks);
  } else if (encoding == InputEncoding::POSITIONAL) {
    if (positionalFrequencies == 4 && inputDim == 16) {
      return dispatchInference03<Type, 16, 4, 1, 4, 1>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks);
    } else if (positionalFrequencies == 8 && inputDim == 32) {
      return dispatchInference03<Type, 32, 4, 1, 8, 1>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks);
    }
  } else if (encoding == InputEncoding::GRID) {
    if (inputDim == 4 && gridResolution == 32) {
      return dispatchInference03<Type, 4, 4, 2, 1, 32>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks);
    } else if (inputDim == 4 && gridResolution == 64) {
      return dispatchInference03<Type, 4, 4, 2, 1, 64>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks);
    } else if (inputDim == 8 && gridResolution == 32) {
      return dispatchInference03<Type, 8, 4, 2, 1, 32>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks);
    } else if (inputDim == 8 && gridResolution == 64) {
      return dispatchInference03<Type, 8, 4, 2, 1, 64>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks);
    } else if (inputDim == 16 && gridResolution == 32) {
      return dispatchInference03<Type, 16, 4, 2, 1, 32>(
          numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks);
    }
  }
  return false;
}

// ============================================================================
// C++ fallback training loop + reconstruction (multi-threaded)
// ============================================================================

template <ex::Arithmetic DataT>
auto trainAndReconstructTextureCppFallback(
    std::span<ex::MlpLayer<DataT, DataT, DataT, DataT>> mlpData,
    const std::vector<DataT>& uvData,
    const std::vector<DataT>& texelData,
    const bool hasBias,
    const CliOptions& options) -> ex::PixmapRgb
{
  const size_t numSamples = uvData.size() / 2;
  const size_t numLayers = mlpData.size();
  const size_t hiddenDim = mlpData.front().outputDimension();
  const size_t inputDim = mlpData.front().inputDimension();
  const ex::ActivationType hiddenAct = mlpData.front().configuration().m_activation;
  const float lr = static_cast<float>(options.m_learningRate);
  const ex::OptimizerType optimizerType = ex::getOptimizerTypeFromString(options.m_optimizer);
  const InputEncoding encoding = inputEncodingFromString(options.m_inputEncoding);
  const bool hasGrid = (encoding == InputEncoding::GRID);

  const size_t gridElements = hasGrid
      ? (options.m_gridResolution * options.m_gridResolution * options.m_gridFeatureDim) : 0;

  // Initialize grid features if needed
  std::vector<float> gridFeatures;
  if (hasGrid) {
    ex::Xoshiro128Plus gridRng{options.m_seed ^ 0xA5A5A5A5u};
    gridFeatures.resize(gridElements);
    for (auto& f : gridFeatures) {
      f = (gridRng.draw() - 0.5f) * 0.02f;
    }
  }

  // Pack weights/biases into byte buffers
  PackedTrainingBuffers03<DataT> packed;
  packed.pack(mlpData, hasBias);
  packed.allocateLogitsCache(options.m_batchSize, numLayers);
  if (hasGrid)
    packed.initializeGrid(gridElements, gridFeatures);
  packed.allocateOptimizerState(optimizerType, gridElements);

  const float lossScale = static_cast<float>(options.m_lossScale);
  const float invLossScale = 1.0f / lossScale;
  const float adamBeta1 = static_cast<float>(options.m_adamBeta1);
  const float adamBeta2 = static_cast<float>(options.m_adamBeta2);
  const float adamEpsilon = static_cast<float>(options.m_adamEpsilon);
  const float lionBeta1 = static_cast<float>(options.m_lionBeta1);
  const float lionBeta2 = static_cast<float>(options.m_lionBeta2);
  const float lionWeightDecay = static_cast<float>(options.m_lionWeightDecay);

  // --- Training loop ---
  std::cout << "Backend: C++ fallback (multi-threaded)\n";
  std::cout << "Starting training...\n";
  const std::chrono::high_resolution_clock::time_point trainingStart = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < options.m_epochs; ++epoch) {
    float epochLoss = 0.0f;
    size_t numBatches = 0;

    for (size_t batchStart = 0; batchStart < numSamples; batchStart += options.m_batchSize) {
      const size_t batchEnd = std::min(batchStart + options.m_batchSize, numSamples);
      const size_t currentBatchSize = batchEnd - batchStart;
      const size_t batchIndex = batchStart / options.m_batchSize;

      packed.zeroGradients();

      if (!dispatchTrainingWithEncoding<DataT>(
              numLayers, hiddenDim, hiddenAct, encoding,
              options.m_positionalFrequencies, options.m_gridResolution, inputDim,
              packed, uvData, texelData, options.m_batchSize, batchIndex, currentBatchSize, lossScale)) {
        std::cerr << std::format("[Error] C++ fallback: unsupported config (layers={}, hiddenDim={}, inputDim={}, encoding={})\n",
            numLayers, hiddenDim, inputDim, options.m_inputEncoding);
        std::abort();
      }

      const float batchLoss = packed.lossValue / static_cast<float>(currentBatchSize);
      epochLoss += batchLoss;
      numBatches++;

      packed.applyOptimizer(optimizerType, lr, gridElements,
          adamBeta1, adamBeta2, adamEpsilon,
          lionBeta1, lionBeta2, lionWeightDecay, invLossScale);
    }

    const float avgLoss = epochLoss / static_cast<float>(numBatches);
    std::cout << std::format("Epoch [{}/{}], Loss: {:.6f}\n",
        epoch + 1, options.m_epochs, avgLoss);
  }
  std::cout << "Training completed!\n";
  {
    const std::chrono::high_resolution_clock::time_point trainingEnd = std::chrono::high_resolution_clock::now();
    const double trainingMs = std::chrono::duration<double, std::milli>(trainingEnd - trainingStart).count();
    std::cout << std::format("Training time: {:.3f} ms\n", trainingMs);
  }

  // --- Reconstruct texture using the trained MLP ---
  std::cout << "Reconstructing texture...\n";
  ex::PixmapRgb texture{options.m_textureWidth, options.m_textureHeight};
  const size_t numPixels = texture.width() * texture.height();
  const std::vector reconstructUv = ex::createUvData<DataT>(texture.width(), texture.height());
  std::vector<DataT> output(numPixels * 4);

  const std::chrono::high_resolution_clock::time_point reconstructStart = std::chrono::high_resolution_clock::now();
  if (!dispatchInferenceWithEncoding<DataT>(
          numLayers, hiddenDim, hiddenAct, encoding,
          options.m_positionalFrequencies, options.m_gridResolution, inputDim,
          packed, reconstructUv, output, numPixels)) {
    std::cerr << std::format("[Error] C++ fallback: unsupported inference config (layers={}, hiddenDim={}, inputDim={}, encoding={})\n",
        numLayers, hiddenDim, inputDim, options.m_inputEncoding);
    std::abort();
  }
  const std::chrono::high_resolution_clock::time_point reconstructEnd = std::chrono::high_resolution_clock::now();
  const double reconstructMs = std::chrono::duration<double, std::milli>(reconstructEnd - reconstructStart).count();
  std::cout << std::format("Reconstruction time: {:.3f} ms\n", reconstructMs);

  mapToLdr<DataT>(output, texture);
  return texture;
}

#endif /* MINIDXNN_EXAMPLE_03_CPP_FALLBACK_PATH_HPP */
