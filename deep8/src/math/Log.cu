#include "math/Log.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct LogKernelOp {

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x) {
        return CuMath::cuLog(x);
    }
};

void LogGPU(const Tensor &x, Tensor &y) {
    auto n = (int)x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseKernel<float, LogKernelOp<float>> <<<grideSize, blockSize>>>(
            x.data<float>(), 
            y.data<float>(), 
            LogKernelOp<float>(), 
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseKernel<double, LogKernelOp<double>> <<<grideSize, blockSize>>>(
            x.data<double>(),
            y.data<double>(),
            LogKernelOp<double>(),
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseKernel<half, LogKernelOp<half>> <<<grideSize, blockSize>>>(
            x.data<half>(),
            y.data<half>(),
            LogKernelOp<half>(),
            n
        );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
struct LogGradKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
        return dy / x;
    }
};

void LogGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto n = (int)dx.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseGradKernel<float, LogGradKernelOp<float>> <<<grideSize, blockSize>>> (
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            LogGradKernelOp<float>(),
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseGradKernel<double, LogGradKernelOp<double>> <<<grideSize, blockSize>>> (
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            LogGradKernelOp<double>(),
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseGradKernel<half, LogGradKernelOp<half>> <<<grideSize, blockSize>>> (
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            LogGradKernelOp<half>(),
            n
        );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}



}
}