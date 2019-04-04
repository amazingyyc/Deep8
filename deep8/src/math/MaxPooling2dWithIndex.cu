#include <crt/device_functions.h>
#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/MaxPooling2dWithIndex.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void MaxPooling2dWithIndexKernel(const T *x,
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
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int b       = i / (outputHeight * outputWidth * channel);
        int outputY = (i % (outputHeight * outputWidth * channel)) / (outputWidth * channel);
        int outputX = (i % (outputWidth * channel)) / channel;
        int c       = i % channel;

        auto starth = max(0, padTop + outputY * strideY);
        auto endh   = min(inputHeight, padTop + outputY * strideY + filterHeight);

        auto startw = max(0, padLeft + outputX * strideX);
        auto endw   = min(inputWidth, padLeft + outputX * strideX + filterWidth);

        int maxh = starth;
        int maxw = startw;

        auto maxvalue = x[((b * inputHeight + starth) * inputWidth + startw) * channel + c];

        for (int h = starth; h < endh; ++h) {
            for (int w = startw; w < endw; ++w) {
                auto value = x[((b * inputHeight + h) * inputWidth + w) * channel + c];

                if (value > maxvalue) {
                    maxvalue = value;

                    maxh = h;
                    maxw = w;
                }
            }
        }

        index[i] = ((b * inputHeight + maxh) * inputWidth + maxw) * channel + c;
        y[i]     = maxvalue;
    }
}

void MaxPooling2dWithIndexGPU(const Tensor &x,
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

    auto padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    auto padX = std::max<int>(0, (outputWidth  - 1) * strideX + filterWidth  - inputWidth);

    auto padTop  = -(padY / 2);
    auto padLeft = -(padX / 2);

    int N = batch * outputHeight * outputWidth * channel;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
        case DType::Float32:
        MaxPooling2dWithIndexKernel<float><<<grideSize, blockSize >>>(
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
        MaxPooling2dWithIndexKernel<double><<<grideSize, blockSize >>>(
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
        MaxPooling2dWithIndexKernel<half><<<grideSize, blockSize >>>(
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
__global__ void MaxPooling2dWithIndexGradKernel(T *dx,
                                               const int *index,
                                               const T *dy,
                                               const int xsize,
                                               const int ysize) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int yi = start; yi < ysize; yi += stride) {
        int xi = index[yi];

        if (0 <= xi && xi < xsize) {
            atomicAdd((T*)(dx + xi), dy[yi]);
        }
    }
}

void MaxPooling2dWithIndexGradGPU(const Tensor &x,
                         Tensor &dx,
                         const Tensor &index,
                         const Tensor &y, 
                         const Tensor &dy,
                         bool covered, 
                         int filterHeight, 
                         int filterWidth, 
                         int strideY, 
                         int strideX) {
    int xsize = (int) x.size();
    int ysize = (int) y.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (ysize + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        MaxPooling2dWithIndexGradKernel<float> << <grideSize, blockSize >> > (
            dx.data<float>(),
            index.data<int>(),
            dy.data<float>(),
            xsize,
            ysize);
        break;
    case DType::Float64:
        MaxPooling2dWithIndexGradKernel<double> << <grideSize, blockSize >> > (
            dx.data<double>(),
            index.data<int>(),
            dy.data<double>(),
            xsize,
            ysize);
        break;

#if defined(HAVE_HALF)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

    case DType::Float16:
        MaxPooling2dWithIndexGradKernel<half> << <grideSize, blockSize >> > (
            dx.data<half>(),
            index.data<int>(),
            dy.data<half>(),
            xsize,
            ysize);
        break;

#endif
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }

}






}
}