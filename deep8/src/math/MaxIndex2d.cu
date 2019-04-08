#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUUnaryElementWise.h"
#include "math/MaxIndex2D.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void MaxIndex2dKernel(const T *x,
                                int *index,
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
    }
}

void MaxIndex2dGPU(const Tensor &x,
                    Tensor &index,
                    bool covered, 
                    int filterHeight, 
                    int filterWidth, 
                    int strideY, 
                    int strideX) {
    auto batch       = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth  = (int)x.dim(1);
    auto channel     = (int)x.dim(2);

    auto outputHeight = (int)index.dim(0);
    auto outputWidth  = (int)index.dim(1);

    auto padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    auto padX = std::max<int>(0, (outputWidth  - 1) * strideX + filterWidth  - inputWidth);

    auto padTop  = -(padY / 2);
    auto padLeft = -(padX / 2);

    int N = batch * outputHeight * outputWidth * channel;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        MaxIndex2dKernel<float><<<grideSize, blockSize >>>(
            x.data<float>(),
            index.data<int>(),
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
        MaxIndex2dKernel<double><<<grideSize, blockSize >>>(
            x.data<double>(),
            index.data<int>(),
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
        MaxIndex2dKernel<half><<<grideSize, blockSize >>>(
            x.data<half>(),
            index.data<int>(),
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