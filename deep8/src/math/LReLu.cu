#include "math:LReLu.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct LReLuKernelOp {
    T a;

    LReLuKernelOp(T aa): a(aa) {
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x) {
        return x >= T(0) ? x : a * x;
    }
};

void LReLuGPU(const Tensor &x, const float a, Tensor &y) {
    auto n = (int)x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseKernel<float, LReLuKernelOp<float>> <<<grideSize, blockSize>>>(
            x.data<float>(), 
            y.data<float>(), 
            LReLuKernelOp<float>(a),
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseKernel<double, LReLuKernelOp<double>> <<<grideSize, blockSize>>>(
            x.data<double>(), 
            y.data<double>(), 
            LReLuKernelOp<double>(double(a)), 
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseKernel<half, LReLuKernelOp<half>> <<<grideSize, blockSize>>>(
            x.data<half>(),
            y.data<half>(),
            LReLuKernelOp<half>(__float2half(a)),
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
struct LReLuGradKernelOp {
    T a;

    LReLuGradKernelOp(T aa): a(aa) {
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
        return x >= T(0) ? dy : a * dy;
    }
};

void LReLuGradGPU(const Tensor &x, Tensor &dx, const float a, const Tensor &y, const Tensor &dy) {
    auto n = (int)dx.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseGradKernel<float, LReLuGradKernelOp<float>> <<<grideSize, blockSize>>> (
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            LReLuGradKernelOp<float>(a),
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseGradKernel<double, LReLuGradKernelOp<double>> <<<grideSize, blockSize>>> (
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            LReLuGradKernelOp<double>(double(a)),
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseGradKernel<half, LReLuGradKernelOp<half>> <<<grideSize, blockSize>>> (
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            LReLuGradKernelOp<half>(__float2half(a)),
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