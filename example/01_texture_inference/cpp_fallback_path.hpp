/*!
  \file cpp_fallback_path.hpp
  \author Sho Ikeda
  \brief C++ fallback inference path for texture MLP inference example
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This header is included from example.cpp inside the anonymous namespace.
  It depends on types and functions defined earlier in example.cpp:
    - MlpConfig<Type>, mapToLdr<Type>
  and on headers already included by example.cpp:
    - common/cpp_fallback.hpp, kernel/texture_inference_common.hlsl
*/

#ifndef MINIDXNN_EXAMPLE_01_CPP_FALLBACK_PATH_HPP
#define MINIDXNN_EXAMPLE_01_CPP_FALLBACK_PATH_HPP 1

// Templated forward kernel: delegates to texkernel::inferenceStep from shared HLSL.
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          typename ActivationHiddenT, typename ActivationLastT>
auto cppFallbackForwardKernel(const ex::PackedMlpBuffers<Type>& packed,
                              const std::vector<Type>& uvData,
                              std::vector<Type>& output,
                              size_t numTasks) -> void
{
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
        texkernel::inferenceStep<Type, NUM_LAYERS, HIDDEN_DIM,
            mininn::impl::TypeTraits<Type>::COMPONENT_TYPE,
            dx::linalg::MATRIX_LAYOUT_ROW_MAJOR,
            ActivationHiddenT, ActivationLastT,
            128, 16, 64, true>(
            task, uvBuf, outBuf, packed.weightBAB(), packed.biasBAB(),
            packed.matrixSizes, totalTasks);
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Dispatch hidden-layer activation type at runtime.
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM>
auto dispatchActivation(ex::ActivationType hiddenAct,
                        const ex::PackedMlpBuffers<Type>& packed,
                        const std::vector<Type>& uvData,
                        std::vector<Type>& output,
                        size_t numTasks) -> bool
{
  // Last activation is always Sigmoid for this example.
  using Sigmoid = mininn::SigmoidActivation;
  switch (hiddenAct) {
    case ex::ActivationType::RELU:
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, mininn::ReluActivation, Sigmoid>(packed, uvData, output, numTasks);
      return true;
    case ex::ActivationType::IDENTITY:
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, mininn::IdentityActivation, Sigmoid>(packed, uvData, output, numTasks);
      return true;
    case ex::ActivationType::SIGMOID:
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, mininn::SigmoidActivation, Sigmoid>(packed, uvData, output, numTasks);
      return true;
    case ex::ActivationType::LEAKY_RELU:
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, mininn::LeakyReluActivation, Sigmoid>(packed, uvData, output, numTasks);
      return true;
    case ex::ActivationType::TANH:
    default:
      return false;
  }
}

// Dispatch NUM_LAYERS and HIDDEN_DIM at runtime.
// Supports common MLP configurations used in texture inference.
template <ex::Arithmetic Type>
auto dispatchForward(size_t numLayers, size_t hiddenDim,
                     ex::ActivationType hiddenAct,
                     const ex::PackedMlpBuffers<Type>& packed,
                     const std::vector<Type>& uvData,
                     std::vector<Type>& output,
                     size_t numTasks) -> bool
{
  // Macro to reduce boilerplate for each (NUM_LAYERS, HIDDEN_DIM) pair
  #define DISPATCH_CASE(NL, HD) \
    if (numLayers == (NL) && hiddenDim == (HD)) \
      return dispatchActivation<Type, (NL), (HD)>(hiddenAct, packed, uvData, output, numTasks);

  // 2 layers (1 backbone): common small models
  DISPATCH_CASE(2, 8)   DISPATCH_CASE(2, 16)
  DISPATCH_CASE(2, 32)  DISPATCH_CASE(2, 64)
  // 3 layers (2 backbone)
  DISPATCH_CASE(3, 8)   DISPATCH_CASE(3, 16)
  DISPATCH_CASE(3, 32)  DISPATCH_CASE(3, 64)
  // 4 layers (3 backbone)
  DISPATCH_CASE(4, 16)  DISPATCH_CASE(4, 32)
  DISPATCH_CASE(4, 64)
  // 5 layers (4 backbone)
  DISPATCH_CASE(5, 16)  DISPATCH_CASE(5, 32)
  DISPATCH_CASE(5, 64)

  #undef DISPATCH_CASE
  return false;
}

/*!
  \brief Run inference using the C++ fallback path (mlp.hlsl compiled as C++).

  Creates ByteAddressBuffers from packed MLP layer data and calls mininn::forward.
*/
template <ex::Arithmetic Type>
auto runCppFallbackInference(const MlpConfig<Type>& mlpConfig,
                             const std::vector<Type>& uvData,
                             ex::PixmapU8& texture) -> void
{
  const std::span mlpData = std::span{mlpConfig.m_layers};
  const bool hasBias = mlpConfig.m_hasBias;
  const size_t numTasks = texture.width() * texture.height();
  const size_t numLayers = mlpData.size();
  const size_t hiddenDim = mlpData.front().outputDimension();
  const ex::ActivationType hiddenAct = mlpData.front().configuration().m_activation;

  ex::PackedMlpBuffers<Type> packed;
  packed.pack(mlpData, hasBias);

  std::vector<Type> output(numTasks * 2);

  const std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
  if (!dispatchForward<Type>(numLayers, hiddenDim, hiddenAct, packed, uvData, output, numTasks)) {
    std::cerr << std::format("[Error] C++ fallback: unsupported MLP config (layers={}, hiddenDim={})\n",
        numLayers, hiddenDim);
    std::abort();
  }
  const std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
  const double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  std::cout << std::format("Reconstruction time: {:.3f} ms\n", elapsedMs);

  ex::mapToLdr<Type>(output, texture);
}

#endif /* MINIDXNN_EXAMPLE_01_CPP_FALLBACK_PATH_HPP */
