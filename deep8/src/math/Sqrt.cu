#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUUnaryElementWise.h"
#include "math/Sqrt.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct SqrtKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x) {
        return cudaSqrt(x);
    }
};

void SqrtGPU(const Tensor &x, Tensor &y) {
    auto n = (int)x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
        case DType::Float32:
        UnaryElementWiseKernel<float, SqrtKernelOp<float>> <<<grideSize, blockSize>>>(
            x.data<float>(), 
            y.data<float>(), 
            SqrtKernelOp<float>(), 
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseKernel<double, SqrtKernelOp<double>> <<<grideSize, blockSize>>>(
            x.data<double>(), 
            y.data<double>(), 
            SqrtKernelOp<double>(), 
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseKernel<half, SqrtKernelOp<half>> <<<grideSize, blockSize>>>(
            x.data<half>(),
            y.data<half>(),
            SqrtKernelOp<half>(),
            n
        );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct SqrtGradKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
        return T(0.5) * dy / cudaSqrt(x);
    }
};

void SqrtGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto n = (int)dx.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
        case DType::Float32:
        UnaryElementWiseGradKernel<float, SqrtGradKernelOp<float>> <<<grideSize, blockSize>>> (
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            SqrtGradKernelOp<float>(),
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseGradKernel<double, SqrtGradKernelOp<double>> <<<grideSize, blockSize>>> (
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            SqrtGradKernelOp<double>(),
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseGradKernel<half, SqrtGradKernelOp<half>> <<<grideSize, blockSize>>> (
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            SqrtGradKernelOp<half>(),
            n
        );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}



}
}