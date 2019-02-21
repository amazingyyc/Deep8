#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/AvgPooling2d.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void AvgPooling2dKernel(const T *x, T *y,
                                   const int batch,
                                   const int inputHeight,
                                   const int inputWidth,
                                   const int outputHeight,
                                   const int outputWidth,
                                   const int channel,
                                   const int filterHeight,
                                   const int filterWidth,
                                   const int strideY,
                                   const int strideX,
                                   const int padTop,
                                   const int padLeft,
                                   const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int b = i / (outputHeight * outputWidth * channel);
        int outputY = (i % (outputHeight * outputWidth * channel)) / (outputWidth * channel);
        int outputX = (i % (outputWidth * channel)) / channel;
        int offset  = i % channel;

        T sum = 0;

        for (int h = 0; h < filterHeight; ++h) {
            for (int w = 0; w < filterWidth; ++w) {
                int inputY = outputY * strideY + padTop  + h;
                int inputX = outputX * strideX + padLeft + w;

                if (0 <= inputY && inputY < inputHeight && 0 <= inputX && inputX < inputWidth) {
                    sum += x[((b * inputHeight + inputY) * inputWidth + inputX) * channel + offset];
                }
            }
        }

        y[i] = sum / T(filterHeight * filterWidth);
    }
}

void AvgPooling2dGPU(const Tensor &x, Tensor &y,
                     bool covered,
                     int filterHeight,
                     int filterWidth,
                     int strideY,
                     int strideX) {
    auto batch       = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth  = (int)x.dim(1);
    auto channel     = (int)x.dim(2);

    auto outputHeight = (int)y.dim(0);
    auto outputWidth  = (int)y.dim(1);

    auto padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    auto padX = std::max<int>(0, (outputWidth  - 1) * strideX + filterWidth  - inputWidth);

    auto padTop  = -(padY / 2);
    auto padLeft = -(padX / 2);

    int N = batch * outputHeight * outputWidth * channel;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        AvgPooling2dKernel<float><<<grideSize, blockSize >>>(
            x.data<float>(), 
            y.data<float>(),
            batch,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            channel,
            filterHeight,
            filterWidth,
            strideY,
            strideX,
            padTop,
            padLeft,
            N);
        break;
    case DType::Float64:
        AvgPooling2dKernel<double> << <grideSize, blockSize >> > (
            x.data<double>(),
            y.data<double>(),
            batch,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            channel,
            filterHeight,
            filterWidth,
            strideY,
            strideX,
            padTop,
            padLeft,
            N);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        AvgPooling2dKernel<half> << <grideSize, blockSize >> > (
            x.data<half>(),
            y.data<half>(),
            batch,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            channel,
            filterHeight,
            filterWidth,
            strideY,
            strideX,
            padTop,
            padLeft,
            N);
        break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
__global__ void AvgPooling2dGradKernel(T *dx, T *dy,
                                       const int batch, 
                                       const int inputHeight, 
                                       const int inputWidth,
                                       const int outputHeight, 
                                       const int outputWidth, 
                                       const int channel,
                                       const int filterHeight, 
                                       const int filterWidth,
                                       const int strideY,
                                       const int strideX,
                                       const int padTop, 
                                       const int padLeft,
                                       const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    T ratio = T(1) / T(filterHeight * filterWidth);

    for (int i = start; i < N; i += stride) {
        int b = i / (inputHeight * inputWidth * channel);
        int inputY = (i % (inputHeight * inputWidth * channel)) / (inputWidth * channel);
        int inputX = (i % (inputWidth * channel)) / channel;
        int offset = i % channel;

        for (int y = 0; y < filterHeight; ++y) {
            for (int x = 0; x < filterWidth; ++x) {
                int outputY = inputY - padTop - y;
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

void AvgPooling2dGradGPU(const Tensor &x, Tensor &dx, 
                         const Tensor &y, const Tensor &dy, 
                         bool covered, 
                         int filterHeight, 
                         int filterWidth, 
                         int strideY, 
                         int strideX) {
    auto batch       = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth  = (int)x.dim(1);
    auto channel     = (int)x.dim(2);

    auto outputHeight = (int)y.dim(0);
    auto outputWidth  = (int)y.dim(1);

    int padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    int padX = std::max<int>(0, (outputWidth - 1)  * strideX + filterWidth  - inputWidth);

    int padTop  = -(padY / 2);
    int padLeft = -(padX / 2);

    int N = batch * inputHeight * inputWidth * channel;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        AvgPooling2dGradKernel<float> << <grideSize, blockSize >> > (
            dx.data<float>(),
            dy.data<float>(),
            batch,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            channel,
            filterHeight,
            filterWidth,
            strideY,
            strideX,
            padTop,
            padLeft,
            N);
        break;
    case DType::Float64:
        AvgPooling2dGradKernel<double> << <grideSize, blockSize >> > (
            dx.data<double>(),
            dy.data<double>(),
            batch,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            channel,
            filterHeight,
            filterWidth,
            strideY,
            strideX,
            padTop,
            padLeft,
            N);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        AvgPooling2dGradKernel<half> << <grideSize, blockSize >> > (
            dx.data<half>(),
            dy.data<half>(),
            batch,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            channel,
            filterHeight,
            filterWidth,
            strideY,
            strideX,
            padTop,
            padLeft,
            N);
        break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


}
}