#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/Linear.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct LinearKernelOp {
    T a;
    T b;

    LinearKernelOp(T aa, T bb): a(aa), b(bb) {
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x) {
        return a * x + b;
    }
};

void LinearGPU(const Tensor &x, const float a, const float b, Tensor &y) {
    auto n = (int)x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseKernel<float, LinearKernelOp<float>> <<<grideSize, blockSize>>>(
            x.data<float>(), 
            y.data<float>(), 
            LinearKernelOp<float>(a, b), 
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseKernel<double, LinearKernelOp<double>> <<<grideSize, blockSize>>>(
            x.data<double>(), 
            y.data<double>(), 
            LinearKernelOp<double>(double(a), double(b)), 
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseKernel<half, LinearKernelOp<half>> <<<grideSize, blockSize>>>(
            x.data<half>(),
            y.data<half>(),
            LinearKernelOp<half>(__float2half(a), __float2half(b)),
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
struct LinearGradKernelOp {
    T a;

    LinearGradKernelOp(T aa): a(aa) {
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
        return a * dy;
    }
};


void LinearGradGPU(const Tensor &x, Tensor &dx, const float a, const float b, const Tensor &y, const Tensor &dy) {
    auto n = (int)dx.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseGradKernel<float, LinearGradKernelOp<float>> <<<grideSize, blockSize>>> (
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            LinearGradKernelOp<float>(a),
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseGradKernel<double, LinearGradKernelOp<double>> <<<grideSize, blockSize>>> (
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            LinearGradKernelOp<double>(double(a)),
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseGradKernel<half, LinearGradKernelOp<half>> <<<grideSize, blockSize>>> (
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            LinearGradKernelOp<half>(__float2half(a)),
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