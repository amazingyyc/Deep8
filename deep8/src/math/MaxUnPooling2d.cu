#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/MaxUnPooling2d.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void MaxUnPooling2dKernel(const T *x,
                                    int *index,
                                    T *y,
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
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int b       =  i / (outputHeight * outputWidth * channel);
        int outputY = (i % (outputHeight * outputWidth * channel)) / (outputWidth * channel);
        int outputX = (i % (outputWidth * channel)) / channel;
        int c       =  i % channel;

        for (int h = 0; h < filterHeight; ++h) {
            for (int w = 0; w < filterWidth; ++w) {
                int inputY = outputY - padTop  - h;
                int inputX = outputX - padLeft - w;

                if (0 == inputY % strideY && 0 == inputX % strideX) {
                    inputY /= strideY;
                    inputX /= strideX;

                    if (0 <= inputY && inputY < inputHeight && 0 <= inputX && inputX <= inputWidth) {
                        auto xi = ((b * inputHeight + inputY) * inputWidth + inputX) * channel + c;

                        if (i == index[xi]) {
                            y[i] = x[xi];
                        }
                    }
                }
            }
        }
    }
}

void MaxUnPooling2dGPU(const Tensor &x,
                       const Tensor &index,
                       Tensor &y,
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

    int padY = std::max<int>(0, (inputHeight - 1) * strideY + filterHeight - outputHeight);
    int padX = std::max<int>(0, (inputWidth  - 1) * strideX + filterWidth  - outputWidth);

    int padTop  = -(padY / 2);
    int padLeft = -(padX / 2);

    int N = batch * outputHeight * outputWidth * channel;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        MaxUnPooling2dKernel<float> << <grideSize, blockSize >> > (
            x.data<float>(),
            index.data<int>(),
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
        MaxUnPooling2dKernel<double> << <grideSize, blockSize >> > (
            x.data<double>(),
            index.data<int>(),
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
        MaxUnPooling2dKernel<half> << <grideSize, blockSize >> > (
            x.data<half>(),
            index.data<int>(),
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
__global__ void MaxUnPooling2dGradKernel(T *dx,
                                        const int *index,
                                        const T *dy,
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
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int maxj = batch * outputHeight * outputWidth * channel;

    for (int i = start; i < N; i += stride) {
        int j = index[i];

        if (0 <= j && j < maxj) {
            dx[i] += dy[j];
        }
    }
}

void MaxUnPooling2dGradGPU(const Tensor &x,
                           Tensor &dx,
                           const Tensor &index,
                           const Tensor &y,
                           const Tensor &dy,
                           bool covered,
                           int filterHeight,
                           int filterWidth,
                           int strideY,
                           int strideX) {
    auto batch = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth = (int)x.dim(1);
    auto channel = (int)x.dim(2);

    auto outputHeight = (int)y.dim(0);
    auto outputWidth = (int)y.dim(1);

    int padY = std::max<int>(0, (inputHeight - 1) * strideY + filterHeight - outputHeight);
    int padX = std::max<int>(0, (inputWidth  - 1) * strideX + filterWidth - outputWidth);

    int padTop = -(padY / 2);
    int padLeft = -(padX / 2);

    int N = batch * inputHeight * inputWidth * channel;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        MaxUnPooling2dGradKernel<float> << <grideSize, blockSize >> > (
            dx.data<float>(),
            index.data<int>(),
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
        MaxUnPooling2dGradKernel<double> << <grideSize, blockSize >> > (
            dx.data<double>(),
            index.data<int>(),
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
        MaxUnPooling2dGradKernel<half> << <grideSize, blockSize >> > (
            dx.data<half>(),
            index.data<int>(),
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