#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "AvgPooling2d.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void AvgPooling2dForwardKernel(const real *input, real *output,
                                          const int batch, const int inputHeight, const int inputWidth,
                                          const int outputHeight, const int outputWidth, const int channel,
                                          const int filterHeight, const int filterWidth,
                                          const int padTop, const int padLeft,
                                          const int strideY, const int strideX, const int N) {

    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int b = i / (outputHeight * outputWidth * channel);
        int outputY = (i % (outputHeight * outputWidth * channel)) / (outputWidth * channel);
        int outputX = (i % (outputWidth * channel)) / channel;
        int offset  = i % channel;

        real sum = 0;

        for (int y = 0; y < filterHeight; ++y) {
            for (int x = 0; x < filterWidth; ++x) {
                int inputY = outputY * strideY + padTop + y;
                int inputX = outputX * strideX + padLeft + x;

                if (0 <= inputY && inputY < inputHeight && 0 <= inputX && inputX < inputWidth) {
                    sum += input[((b * inputHeight + inputY) * inputWidth + inputX) * channel + offset];
                }
            }
        }

        output[i] = sum / real(filterHeight * filterWidth);
    }
}

template <typename real>
__global__ void AvgPooling2dBackwardKernel(real *dx, const real *dy,
                                           const int batch, const int inputHeight, const int inputWidth,
                                           const int outputHeight, const int outputWidth, const int channel,
                                           const int filterHeight, const int filterWidth,
                                           const int padTop, const int padLeft,
                                           const int strideY, const int strideX, const int N) {

    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    real ratio = real(1) / real(filterHeight * filterWidth);

    for (int i = start; i < N; i += stride) {
        int b = i / (inputHeight * inputWidth * channel);
        int inputY = (i % (inputHeight * inputWidth * channel)) / (inputWidth * channel);
        int inputX = (i % (inputWidth * channel)) / channel;
        int offset = i % channel;

        for (int y = 0; y < filterHeight; ++y) {
            for (int x = 0; x < filterWidth; ++x) {
                int outputY = inputY - padTop  - y;
                int outputX = inputX - padLeft - x;

                if (0 == outputY % strideY && 0 == outputX % strideX) {
                    outputY /= strideY;
                    outputX /= strideX;

                    if (0 <= outputY && outputY < outputHeight && 0 <= outputX && outputX < outputWidth) {
                        dx[i] += ratio * dy[((b * outputHeight + outputY) * outputWidth + outputX) * channel + offset];
                    }
                }
            }
        }
    }
}

template <typename T>
void AvgPooling2d<T>::forwardGPUImpl(const T *x, T *y,
                                    const int batch,
                                    const int inputHeight, const int inputWidth,
                                    const int outputHeight, const int outputWidth,
                                    const int channel,
                                    const int filterHeight, const int filterWidth,
                                    const int padTop, const int padLeft,
                                    const int strideY, const int strideX) {

    int N = batch * outputHeight * outputWidth * channel;

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AvgPooling2dForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    AvgPooling2dForwardKernel<T><<<grideSize, blockSize>>>(x, y,
                                                        batch, inputHeight, inputWidth,
                                                        outputHeight, outputWidth, channel,
                                                        filterHeight, filterWidth,
                                                        padTop, padLeft,
                                                        strideY, strideX, N);
}

#ifdef HAVE_HALF

template <>
void AvgPooling2d<half>::forwardGPUImpl(const half *x, half *y,
                                    const int batch,
                                    const int inputHeight, const int inputWidth,
                                    const int outputHeight, const int outputWidth,
                                    const int channel,
                                    const int filterHeight, const int filterWidth,
                                    const int padTop, const int padLeft,
                                    const int strideY, const int strideX) {
    int N = batch * outputHeight * outputWidth * channel;

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    AvgPooling2dForwardKernel<half> << <grideSize, blockSize >> > (x, y,
        batch, inputHeight, inputWidth,
        outputHeight, outputWidth, channel,
        filterHeight, filterWidth,
        padTop, padLeft,
        strideY, strideX, N);
}
#endif // HAVE_HALF

template <typename T>
void AvgPooling2d<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto batch       = static_cast<int>(inputs[0]->shape.dim(0));
    auto inputHeight = static_cast<int>(inputs[0]->shape.dim(1));
    auto inputWidth  = static_cast<int>(inputs[0]->shape.dim(2));
    auto channel     = static_cast<int>(inputs[0]->shape.dim(3));

    auto outputHeight = static_cast<int>(output->shape.dim(1));
    auto outputWidth  = static_cast<int>(output->shape.dim(2));

    int padY = std::max<int>(0, (outputHeight - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputHeight);
    int padX = std::max<int>(0, (outputWidth  - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth)  - inputWidth);

    int padTop  = -(padY / 2);
    int padLeft = -(padX / 2);

    forwardGPUImpl(inputs[0]->data(), output->data(),
        batch, inputHeight, inputWidth, outputHeight, outputWidth, channel, (int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);
}

template <typename T>
void AvgPooling2d<T>::backwardGPUImpl(T *dx, const T *dy,
                                     const int batch, const int inputHeight, const int inputWidth,
                                     const int outputHeight, const int outputWidth, const int channel,
                                     const int filterHeight, const int filterWidth,
                                     const int padTop, const int padLeft,
                                     const int strideY, const int strideX) {

    int N = batch * inputHeight * inputWidth * channel;

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AvgPooling2dBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    AvgPooling2dBackwardKernel<T> << <grideSize, blockSize >> > (dx, dy,
            batch, inputHeight, inputWidth,
            outputHeight, outputWidth, channel,
            filterHeight, filterWidth,
            padTop, padLeft,
            strideY, strideX, N);
}

#ifdef HAVE_HALF
template <>
void AvgPooling2d<half>::backwardGPUImpl(half *dx, const half *dy,
                                    const int batch, const int inputHeight, const int inputWidth,
                                    const int outputHeight, const int outputWidth, const int channel,
                                    const int filterHeight, const int filterWidth,
                                    const int padTop, const int padLeft,
                                    const int strideY, const int strideX) {

    int N = batch * inputHeight * inputWidth * channel;

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    AvgPooling2dBackwardKernel<half> << <grideSize, blockSize >> > (dx, dy,
        batch, inputHeight, inputWidth,
        outputHeight, outputWidth, channel,
        filterHeight, filterWidth,
        padTop, padLeft,
        strideY, strideX, N);
}
#endif

template <typename T>
void AvgPooling2d<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {

    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto batch       = static_cast<int>(inputs[0]->shape.dim(0));
    auto inputHeight = static_cast<int>(inputs[0]->shape.dim(1));
    auto inputWidth  = static_cast<int>(inputs[0]->shape.dim(2));
    auto channel     = static_cast<int>(inputs[0]->shape.dim(3));

    auto outputHeight = static_cast<int>(output->shape.dim(1));
    auto outputWidth  = static_cast<int>(output->shape.dim(2));

    int padY = std::max<int>(0, (outputHeight - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputHeight);
    int padX = std::max<int>(0, (outputWidth  - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth) - inputWidth);

    int padTop  = -(padY / 2);
    int padLeft = -(padX / 2);

    backwardGPUImpl(iGradient->data(), outputGradient->data(),
			batch, inputHeight, inputWidth, outputHeight, outputWidth, channel,
			(int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);
}

DEEP8_DECLARATION_GPU_FUNC(AvgPooling2d);

#endif

}