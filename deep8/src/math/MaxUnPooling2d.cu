#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/Constant.h"
#include "math/MaxUnPooling2d.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void MaxUnPooling2dKernel(const T *x,
                                    int *index,
                                    T *y,
                                    const int xsize,
                                    const int ysize) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int xi = start; xi < xsize; ++xi) {
        int yi = index[xi];

        if (0 <= yi && yi < ysize) {
            y[yi] = x[xi];
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
    /**set y to 0 first*/
    Constant(y, 0);

    int xsize = (int) x.size();
    int ysize = (int) y.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (xsize + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        MaxUnPooling2dKernel<float> << <grideSize, blockSize >> > (
            x.data<float>(),
            index.data<int>(),
            y.data<float>(),
            xsize,
            ysize);
        break;

    case DType::Float64:
        MaxUnPooling2dKernel<double> << <grideSize, blockSize >> > (
            x.data<double>(),
            index.data<int>(),
            y.data<double>(),
            xsize,
            ysize);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        MaxUnPooling2dKernel<half> << <grideSize, blockSize >> > (
            x.data<half>(),
            index.data<int>(),
            y.data<half>(),
            xsize,
            ysize);
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
                                        const int xsize,
                                        const int ysize) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int xi = start; xi < xsize; xi += stride) {
        int yi = index[xi];

        if (0 <= yi && yi < ysize) {
            dx[xi] += dy[yi];
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
    int xsize = (int)x.size();
    int ysize = (int)y.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (xsize + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        MaxUnPooling2dGradKernel<float> << <grideSize, blockSize >> > (
            dx.data<float>(),
            index.data<int>(),
            dy.data<float>(),
            xsize,
            ysize);
        break;
    case DType::Float64:
        MaxUnPooling2dGradKernel<double> << <grideSize, blockSize >> > (
            dx.data<double>(),
            index.data<int>(),
            dy.data<double>(),
            xsize,
            ysize);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        MaxUnPooling2dGradKernel<half> << <grideSize, blockSize >> > (
            dx.data<half>(),
            index.data<int>(),
            dy.data<half>(),
            xsize,
            ysize);
        break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}







}
}