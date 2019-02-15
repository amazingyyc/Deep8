#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/ReLu.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct ReLuKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x) {
        return x >= T(0) ? x : T(0);
    }
};

void ReLuGPU(const Tensor &x, Tensor &y) {
    auto n = (int)x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseKernel<float, ReLuKernelOp<float>> <<<grideSize, blockSize>>>(
            x.data<float>(), 
            y.data<float>(), 
            ReLuKernelOp<float>(), 
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseKernel<double, ReLuKernelOp<double>> <<<grideSize, blockSize>>>(
            x.data<double>(), 
            y.data<double>(), 
            ReLuKernelOp<double>(), 
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseKernel<half, ReLuKernelOp<half>> <<<grideSize, blockSize>>>(
            x.data<half>(), 
            y.data<half>(), 
            ReLuKernelOp<half>(), 
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
struct ReLuGradKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
        return x >= T(0) ? dy : T(0);
    }
};

void ReLuGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto n = (int)dx.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseGradKernel<float, ReLuGradKernelOp<float>> <<<grideSize, blockSize>>> (
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            ReLuGradKernelOp<float>(),
            n
        );
        break;
    case DType::Float64:
        UnaryElementWiseGradKernel<double, ReLuGradKernelOp<double>> <<<grideSize, blockSize>>> (
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            ReLuGradKernelOp<double>(),
            n
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseGradKernel<half, ReLuGradKernelOp<half>> <<<grideSize, blockSize>>> (
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            ReLuGradKernelOp<half>(),
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