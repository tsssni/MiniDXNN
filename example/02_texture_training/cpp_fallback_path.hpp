/*!
  \file cpp_fallback_path.hpp
  \author Sho Ikeda
  \brief C++ fallback training and inference paths for texture MLP training example
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT

  This header is included from example.cpp inside the anonymous namespace.
  It depends on types and functions defined earlier in example.cpp:
    - CliOptions, MlpConfig<Type>
  and on headers already included by example.cpp:
    - common/cpp_fallback.hpp, common/texture.hpp, common/pixmap.hpp,
      kernel/texture_training_common.hlsl,
      kernel/texture_inference_common.hlsl, kernel/optimizer.hlsl
*/

#ifndef MINIDXNN_EXAMPLE_02_CPP_FALLBACK_PATH_HPP
#define MINIDXNN_EXAMPLE_02_CPP_FALLBACK_PATH_HPP 1

// ============================================================================
// CPU reference training
// ============================================================================

/*!
  \brief Train the MLP on CPU and reconstruct the texture.

  Performs mini-batch stochastic gradient descent training:
    1. For each epoch, iterate over training data in batches
    2. For each batch: forward pass -> MSE loss -> backward pass -> optimizer step
    3. After training completes, evaluate the trained MLP at every pixel

  \param mlpData    MLP layers (modified in-place during training)
  \param uvData     Training UV coordinates (2 * numSamples elements)
  \param texelData  Ground-truth texel values (2 * numSamples elements)
  \param options    Training hyperparameters and output configuration
  \return Reconstructed texture as an 8-bit grayscale pixmap
*/
template <ex::Arithmetic DataT>
auto trainAndReconstructTextureCpu(
    std::span<ex::MlpLayer<DataT, DataT, DataT, DataT>> mlpData,
    const std::vector<DataT>& uvData,
    const std::vector<DataT>& texelData,
    const bool hasBias,
    const CliOptions& options) -> ex::PixmapU8
{
  const size_t numSamples = uvData.size() / 2;
  const size_t numLayers = mlpData.size();
  constexpr size_t outputDim = 2;
  const float lr = static_cast<float>(options.m_learningRate);

  // Create optimizer (SGD, Adam, or Lion)
  const ex::OptimizerType optimizerType = ex::getOptimizerTypeFromString(options.m_optimizer);
  std::unique_ptr optimizer = ex::createOptimizer<DataT, DataT, DataT, DataT>(optimizerType);

  // Pre-allocate logits cache for forward/backward passes
  std::vector<DataT> logitsCache;
  std::array<DataT, outputDim> output;
  const size_t cacheSize = (numLayers - 1) * mlpData.back().inputDimension()
                           + mlpData.back().outputDimension();
  logitsCache.resize(cacheSize);

  // --- Training loop ---
  std::cout << "Backend: CPU (reference)\n";
  std::cout << "Starting training...\n";
  const std::chrono::high_resolution_clock::time_point trainingStart = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < options.m_epochs; ++epoch) {
    float epochLoss = 0.0f;
    size_t numBatches = 0;

    for (size_t batchStart = 0; batchStart < numSamples; batchStart += options.m_batchSize) {
      const size_t batchEnd = std::min(batchStart + options.m_batchSize, numSamples);
      const size_t currentBatchSize = batchEnd - batchStart;

      // Zero gradients before each batch
      using LayerT = std::remove_cvref_t<typename decltype(mlpData)::value_type>;
      std::ranges::for_each(mlpData, [](LayerT& layer) { layer.resetGrads(); });

      float batchLoss = 0.0f;

      for (size_t sampleIdx = batchStart; sampleIdx < batchEnd; ++sampleIdx) {
        const std::span<const DataT> input{uvData.data() + 2 * sampleIdx, 2};
        const std::span<const DataT> target{texelData.data() + 2 * sampleIdx, 2};

        // Forward pass (stores pre-activation logits in cache for backward pass)
        ex::forward<DataT, DataT, DataT, DataT, DataT>(output, input, mlpData, logitsCache);
        batchLoss += ex::mseLoss<DataT>(output, target);

        // MSE gradient scaled by 1/batchSize for batch averaging
        const float batchScale = 1.0f / static_cast<float>(currentBatchSize);
        std::vector lossGradient = ex::mseLossGradient<DataT>(output, target);
        std::ranges::for_each(lossGradient, [batchScale](DataT& v) { v *= batchScale; });

        // Backward pass (accumulates gradients into layer members)
        [[maybe_unused]] const std::vector upstreamGrad =
            ex::backward<DataT, DataT, DataT, DataT, DataT>(
                lossGradient, input, mlpData, logitsCache);
      }

      batchLoss /= static_cast<float>(currentBatchSize);
      epochLoss += batchLoss;
      numBatches++;

      // If bias is disabled, zero out bias gradients to prevent bias updates
      if (!hasBias) {
        std::ranges::for_each(mlpData, [](auto& layer) {
          std::ranges::fill(layer.biasGrads(), static_cast<DataT>(0));
        });
      }

      // Update weights using the selected optimizer
      optimizer->step(mlpData, lr);
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
  ex::PixmapU8 texture{options.m_textureWidth, options.m_textureHeight};
  const std::vector reconstructUv = ex::createUvData<DataT>(texture.width(), texture.height());
  const std::chrono::high_resolution_clock::time_point reconstructStart = std::chrono::high_resolution_clock::now();
  const std::vector inferenceOutput =
      ex::forwardBatch<DataT, DataT, DataT, DataT, DataT>(reconstructUv, mlpData);
  const std::chrono::high_resolution_clock::time_point reconstructEnd = std::chrono::high_resolution_clock::now();
  const double reconstructMs = std::chrono::duration<double, std::milli>(reconstructEnd - reconstructStart).count();
  std::cout << std::format("Reconstruction time: {:.3f} ms\n", reconstructMs);
  ex::mapToLdr<DataT>(inferenceOutput, texture);

  return texture;
}

// ============================================================================
// C++ fallback training (mlp.hlsl compiled as C++)
// ============================================================================

// Pack MLP layer weights/biases and training buffers into flat byte buffers
// matching the layout expected by mlp.hlsl.
// Extends the base PackedMlpBuffers with gradient, logits, and optimizer state.
template <ex::Arithmetic Type>
struct PackedTrainingBuffers
{
  std::vector<std::uint8_t> weightBuf;
  std::vector<std::uint8_t> biasBuf;
  std::vector<std::uint8_t> weightGradBuf;
  std::vector<std::uint8_t> biasGradBuf;
  std::vector<std::uint8_t> logitsCacheBuf;
  float lossValue = 0.0f;
  size_t biasStride = 0;
  [[maybe_unused]] uint32_t m_padd[2];
  uint2 matrixSizes{};

  // Optimizer state buffers (moment values are always float, one per parameter element)
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
  RWByteAddressBuffer weightFirstMomentRWBAB() { return RWByteAddressBuffer{weightFirstMomentBuf}; }
  RWByteAddressBuffer weightSecondMomentRWBAB() { return RWByteAddressBuffer{weightSecondMomentBuf}; }
  RWByteAddressBuffer biasFirstMomentRWBAB() { return RWByteAddressBuffer{biasFirstMomentBuf}; }
  RWByteAddressBuffer biasSecondMomentRWBAB() { return RWByteAddressBuffer{biasSecondMomentBuf}; }

  auto pack(const std::span<const ex::MlpLayer<Type, Type, Type, Type>> mlpData, const bool hasBias) -> void;
  auto allocateLogitsCache(const size_t batchSize, const size_t numLayers) -> void;
  auto allocateOptimizerState(const ex::OptimizerType optType) -> void;
  [[nodiscard]] auto logitsPerSample(const size_t numLayers) const -> size_t;
  auto zeroGradients() -> void;
  auto applyOptimizer(const ex::OptimizerType optType, const float lr) -> void;
};

// ============================================================================
// PackedTrainingBuffers implementation
// ============================================================================

template <ex::Arithmetic Type>
auto PackedTrainingBuffers<Type>::pack(const std::span<const ex::MlpLayer<Type, Type, Type, Type>> mlpData, const bool hasBias) -> void
{
  const size_t numLayers = mlpData.size();
  const size_t hiddenDim = (numLayers > 1) ? mlpData[0].outputDimension()
                                           : mlpData[0].inputDimension();

  // Delegate weight/bias packing to shared PackedMlpBuffers
  ex::PackedMlpBuffers<Type> base;
  base.pack(mlpData, hasBias);
  weightBuf = std::move(base.weightBuf);
  biasBuf = std::move(base.biasBuf);
  matrixSizes = base.matrixSizes;

  biasStride = ex::alignBytes(hiddenDim * sizeof(Type), ex::VECTOR_ALIGNMENT);

  // Allocate gradient buffers (same size as weight/bias buffers)
  weightGradBuf.assign(weightBuf.size(), 0);
  biasGradBuf.assign(biasBuf.size(), 0);
}

template <ex::Arithmetic Type>
auto PackedTrainingBuffers<Type>::allocateLogitsCache(const size_t batchSize, const size_t numLayers) -> void
{
  const size_t perSample = biasStride * numLayers;
  logitsCacheBuf.assign(perSample * batchSize, 0);
}

template <ex::Arithmetic Type>
auto PackedTrainingBuffers<Type>::allocateOptimizerState(const ex::OptimizerType optType) -> void
{
  const size_t weightElements = weightBuf.size() / sizeof(Type);
  const size_t biasElements = biasBuf.size() / sizeof(Type);

  if (optType == ex::OptimizerType::ADAM) {
    weightFirstMomentBuf.assign(weightElements * sizeof(float), 0);
    weightSecondMomentBuf.assign(weightElements * sizeof(float), 0);
    biasFirstMomentBuf.assign(biasElements * sizeof(float), 0);
    biasSecondMomentBuf.assign(biasElements * sizeof(float), 0);
  } else if (optType == ex::OptimizerType::LION) {
    // Lion uses only first moment (momentum)
    weightFirstMomentBuf.assign(weightElements * sizeof(float), 0);
    biasFirstMomentBuf.assign(biasElements * sizeof(float), 0);
  }
  timestep = 0;
}

template <ex::Arithmetic Type>
auto PackedTrainingBuffers<Type>::logitsPerSample(const size_t numLayers) const -> size_t
{
  return biasStride * numLayers;
}

template <ex::Arithmetic Type>
auto PackedTrainingBuffers<Type>::zeroGradients() -> void
{
  std::ranges::fill(weightGradBuf, std::uint8_t{0});
  std::ranges::fill(biasGradBuf, std::uint8_t{0});
  lossValue = 0.0f;
}

template <ex::Arithmetic Type>
auto PackedTrainingBuffers<Type>::applyOptimizer(const ex::OptimizerType optType, const float lr) -> void
{
  const uint wSize = static_cast<uint>(weightBuf.size());
  const uint bSize = static_cast<uint>(biasBuf.size());
  constexpr float invLossScale = 1.0f;

  switch (optType) {
    case ex::OptimizerType::SGD:
      optimizer::sgdUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(), lr, invLossScale, wSize);
      optimizer::sgdUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(), lr, invLossScale, bSize);
      break;
    case ex::OptimizerType::ADAM: {
      ++timestep;
      const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
      const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
      const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));
      optimizer::adamUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(),
          weightFirstMomentRWBAB(), weightSecondMomentRWBAB(),
          lr, beta1, beta2, epsilon, bc1, bc2, invLossScale, wSize);
      optimizer::adamUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(),
          biasFirstMomentRWBAB(), biasSecondMomentRWBAB(),
          lr, beta1, beta2, epsilon, bc1, bc2, invLossScale, bSize);
      break;
    }
    case ex::OptimizerType::LION: {
      const float beta1 = 0.9f, beta2 = 0.99f, weightDecay = 0.3f;
      optimizer::lionUpdateAll<Type>(weightRWBAB(), weightGradRWBAB(),
          weightFirstMomentRWBAB(), lr, beta1, beta2, weightDecay, invLossScale, wSize);
      optimizer::lionUpdateAll<Type>(biasRWBAB(), biasGradRWBAB(),
          biasFirstMomentRWBAB(), lr, beta1, beta2, weightDecay, invLossScale, bSize);
      break;
    }
    default:
      break;
  }
}

// ============================================================================
// Training kernel dispatch
// ============================================================================

// Training kernel: forward + MSE loss + backward for one batch
// Delegates to shared texkernel::trainingStep from texture_training_common.hlsl.
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          typename ActivationHiddenT, typename ActivationLastT>
auto cppFallbackTrainingKernel(PackedTrainingBuffers<Type>& packed,
                               const std::vector<Type>& uvData,
                               const std::vector<Type>& texelData,
                               const size_t batchSize,
                               const size_t batchIndex,
                               const size_t currentBatchSize) -> void
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
        texkernel::trainingStep<Type, NUM_LAYERS, HIDDEN_DIM,
            DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
            128, 16, 64>(
            threadId, uvBuf, targetBuf, packed.weightBAB(), packed.biasBAB(),
            packed.weightGradRWBAB(), packed.biasGradRWBAB(),
            packed.logitsCacheRWBAB(), packed.lossRWBAB(),
            packed.matrixSizes, static_cast<uint>(batchSize),
            static_cast<uint>(batchIndex), static_cast<uint>(currentBatchSize),
            static_cast<uint>(packed.biasStride));
      }
    });
    taskStart = taskEnd;
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Dispatch activation types for training
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM>
auto dispatchTrainingActivation(const ex::ActivationType hiddenAct,
                                PackedTrainingBuffers<Type>& packed,
                                const std::vector<Type>& uvData,
                                const std::vector<Type>& texelData,
                                const size_t batchSize,
                                const size_t batchIndex,
                                const size_t currentBatchSize) -> bool
{
  // Last activation is always Sigmoid for this example
  using Sigmoid = mininn::SigmoidActivation;

  #define DISPATCH_TRAIN_ACT(HiddenT, hiddenE) \
    if (hiddenAct == (hiddenE)) { \
      cppFallbackTrainingKernel<Type, NUM_LAYERS, HIDDEN_DIM, HiddenT, Sigmoid>( \
          packed, uvData, texelData, batchSize, batchIndex, currentBatchSize); \
      return true; \
    }

  DISPATCH_TRAIN_ACT(mininn::IdentityActivation,  ex::ActivationType::IDENTITY)
  DISPATCH_TRAIN_ACT(mininn::SigmoidActivation,   ex::ActivationType::SIGMOID)
  DISPATCH_TRAIN_ACT(mininn::ReluActivation,       ex::ActivationType::RELU)
  DISPATCH_TRAIN_ACT(mininn::LeakyReluActivation,  ex::ActivationType::LEAKY_RELU)

  #undef DISPATCH_TRAIN_ACT
  return false;
}

// Dispatch NUM_LAYERS and HIDDEN_DIM for training
template <ex::Arithmetic Type>
auto dispatchTraining(const size_t numLayers, const size_t hiddenDim,
                      const ex::ActivationType hiddenAct,
                      PackedTrainingBuffers<Type>& packed,
                      const std::vector<Type>& uvData,
                      const std::vector<Type>& texelData,
                      const size_t batchSize,
                      const size_t batchIndex,
                      const size_t currentBatchSize) -> bool
{
  #define DISPATCH_TRAIN(NL, HD) \
    if (numLayers == (NL) && hiddenDim == (HD)) \
      return dispatchTrainingActivation<Type, (NL), (HD)>( \
          hiddenAct, packed, uvData, texelData, batchSize, batchIndex, currentBatchSize);

  // 2 layers (1 backbone)
  DISPATCH_TRAIN(2, 8)   DISPATCH_TRAIN(2, 16)
  DISPATCH_TRAIN(2, 32)  DISPATCH_TRAIN(2, 64)
  // 3 layers (2 backbone)
  DISPATCH_TRAIN(3, 8)   DISPATCH_TRAIN(3, 16)
  DISPATCH_TRAIN(3, 32)  DISPATCH_TRAIN(3, 64)
  // 4 layers (3 backbone) — default configuration
  DISPATCH_TRAIN(4, 8)   DISPATCH_TRAIN(4, 16)
  DISPATCH_TRAIN(4, 32)  DISPATCH_TRAIN(4, 64)
  // 5 layers (4 backbone)
  DISPATCH_TRAIN(5, 8)   DISPATCH_TRAIN(5, 16)
  DISPATCH_TRAIN(5, 32)  DISPATCH_TRAIN(5, 64)

  #undef DISPATCH_TRAIN
  return false;
}

// ============================================================================
// Forward inference kernel dispatch (for reconstruction after training)
// ============================================================================

template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM,
          typename ActivationHiddenT, typename ActivationLastT>
auto cppFallbackForwardKernel(const PackedTrainingBuffers<Type>& packed,
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
        texkernel::inferenceStep<Type, NUM_LAYERS, HIDDEN_DIM,
            DT, dx::linalg::MATRIX_LAYOUT_ROW_MAJOR, ActivationHiddenT, ActivationLastT,
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

// Dispatch activation types for forward inference
template <ex::Arithmetic Type, uint NUM_LAYERS, int HIDDEN_DIM>
auto dispatchForwardActivation(const ex::ActivationType hiddenAct,
                               const PackedTrainingBuffers<Type>& packed,
                               const std::vector<Type>& uvData,
                               std::vector<Type>& output,
                               const size_t numTasks) -> bool
{
  using Sigmoid = mininn::SigmoidActivation;

  #define DISPATCH_FWD_ACT(HiddenT, hiddenE) \
    if (hiddenAct == (hiddenE)) { \
      cppFallbackForwardKernel<Type, NUM_LAYERS, HIDDEN_DIM, HiddenT, Sigmoid>( \
          packed, uvData, output, numTasks); \
      return true; \
    }

  DISPATCH_FWD_ACT(mininn::IdentityActivation,  ex::ActivationType::IDENTITY)
  DISPATCH_FWD_ACT(mininn::SigmoidActivation,   ex::ActivationType::SIGMOID)
  DISPATCH_FWD_ACT(mininn::ReluActivation,       ex::ActivationType::RELU)
  DISPATCH_FWD_ACT(mininn::LeakyReluActivation,  ex::ActivationType::LEAKY_RELU)

  #undef DISPATCH_FWD_ACT
  return false;
}

// Dispatch NUM_LAYERS and HIDDEN_DIM for forward inference
template <ex::Arithmetic Type>
auto dispatchForward(const size_t numLayers, const size_t hiddenDim,
                     const ex::ActivationType hiddenAct,
                     const PackedTrainingBuffers<Type>& packed,
                     const std::vector<Type>& uvData,
                     std::vector<Type>& output,
                     const size_t numTasks) -> bool
{
  #define DISPATCH_FWD(NL, HD) \
    if (numLayers == (NL) && hiddenDim == (HD)) \
      return dispatchForwardActivation<Type, (NL), (HD)>( \
          hiddenAct, packed, uvData, output, numTasks);

  DISPATCH_FWD(2, 8)   DISPATCH_FWD(2, 16)
  DISPATCH_FWD(2, 32)  DISPATCH_FWD(2, 64)
  DISPATCH_FWD(3, 8)   DISPATCH_FWD(3, 16)
  DISPATCH_FWD(3, 32)  DISPATCH_FWD(3, 64)
  DISPATCH_FWD(4, 8)   DISPATCH_FWD(4, 16)
  DISPATCH_FWD(4, 32)  DISPATCH_FWD(4, 64)
  DISPATCH_FWD(5, 8)   DISPATCH_FWD(5, 16)
  DISPATCH_FWD(5, 32)  DISPATCH_FWD(5, 64)

  #undef DISPATCH_FWD
  return false;
}

// ============================================================================
// C++ fallback training loop + reconstruction
// ============================================================================

/*!
  \brief Train the MLP using C++ fallback and reconstruct the texture.

  Uses mlp.hlsl compiled as C++ (via hlsl_compat.hpp) for forward/backward passes.
  The optimizer operates directly on packed byte buffers, eliminating the need
  to unpack gradients or re-pack weights between batches.
*/
template <ex::Arithmetic DataT>
auto trainAndReconstructTextureCppFallback(
    std::span<ex::MlpLayer<DataT, DataT, DataT, DataT>> mlpData,
    const std::vector<DataT>& uvData,
    const std::vector<DataT>& texelData,
    const bool hasBias,
    const CliOptions& options) -> ex::PixmapU8
{
  const size_t numSamples = uvData.size() / 2;
  const size_t numLayers = mlpData.size();
  const size_t hiddenDim = mlpData.front().outputDimension();
  const ex::ActivationType hiddenAct = mlpData.front().configuration().m_activation;
  const float lr = static_cast<float>(options.m_learningRate);
  const ex::OptimizerType optimizerType = ex::getOptimizerTypeFromString(options.m_optimizer);

  // Pack weights/biases into byte buffers
  PackedTrainingBuffers<DataT> packed;
  packed.pack(mlpData, hasBias);
  packed.allocateLogitsCache(options.m_batchSize, numLayers);
  packed.allocateOptimizerState(optimizerType);

  // --- Training loop ---
  std::cout << "Backend: C++ fallback\n";
  std::cout << "Starting training...\n";
  const std::chrono::high_resolution_clock::time_point trainingStart = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < options.m_epochs; ++epoch) {
    float epochLoss = 0.0f;
    size_t numBatches = 0;

    for (size_t batchStart = 0; batchStart < numSamples; batchStart += options.m_batchSize) {
      const size_t batchEnd = std::min(batchStart + options.m_batchSize, numSamples);
      const size_t currentBatchSize = batchEnd - batchStart;
      const size_t batchIndex = batchStart / options.m_batchSize;

      // Zero gradients and loss
      packed.zeroGradients();

      // Run training kernel (forward + backward for all samples in batch)
      if (!dispatchTraining<DataT>(numLayers, hiddenDim, hiddenAct,
              packed, uvData, texelData, options.m_batchSize, batchIndex, currentBatchSize)) {
        std::cerr << std::format("[Error] C++ fallback: unsupported training config (layers={}, hiddenDim={})\n",
            numLayers, hiddenDim);
        std::abort();
      }

      // Accumulate loss
      const float batchLoss = packed.lossValue / static_cast<float>(currentBatchSize);
      epochLoss += batchLoss;
      numBatches++;

      // Apply optimizer directly on packed buffers (no unpack/repack)
      packed.applyOptimizer(optimizerType, lr);
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
  ex::PixmapU8 texture{options.m_textureWidth, options.m_textureHeight};
  const std::vector reconstructUv = ex::createUvData<DataT>(texture.width(), texture.height());
  const size_t numPixels = texture.width() * texture.height();
  std::vector<DataT> output(numPixels * 2);

  const std::chrono::high_resolution_clock::time_point reconstructStart = std::chrono::high_resolution_clock::now();
  if (!dispatchForward<DataT>(numLayers, hiddenDim, hiddenAct,
          packed, reconstructUv, output, numPixels)) {
    std::cerr << std::format("[Error] C++ fallback: unsupported inference config (layers={}, hiddenDim={})\n",
        numLayers, hiddenDim);
    std::abort();
  }
  const std::chrono::high_resolution_clock::time_point reconstructEnd = std::chrono::high_resolution_clock::now();
  const double reconstructMs = std::chrono::duration<double, std::milli>(reconstructEnd - reconstructStart).count();
  std::cout << std::format("Reconstruction time: {:.3f} ms\n", reconstructMs);

  ex::mapToLdr<DataT>(output, texture);
  return texture;
}

#endif /* MINIDXNN_EXAMPLE_02_CPP_FALLBACK_PATH_HPP */
