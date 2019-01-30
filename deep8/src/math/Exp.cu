#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "utils/GPUMathUtils.h"
#include "math/Exp.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct ExpKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x) {
        return CuMath::cuExp(x);
    }
};

void ExpGPU(const Tensor &x, Tensor &y) {
    auto n = (int)x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
            UnaryElementWiseKernel<float, ExpKernelOp<float>> <<<grideSize, blockSize>>>(x.data<float>(), y.data<float>(), ExpKernelOp<float>(), n);
            break;
        case DType::Float64:
            UnaryElementWiseKernel<double, ExpKernelOp<double>> <<<grideSize, blockSize>>>(x.data<double>(), y.data<double>(), ExpKernelOp<double>(), n);
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            UnaryElementWiseKernel<half, ExpKernelOp<half>> <<<grideSize, blockSize>>>(x.data<half>(), y.data<half>(), ExpKernelOp<half>(), n);
            break;
#endif
    
        default:
            DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
            break;
        }
}

template <typename T>
struct ExpGradKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
        return y * dy;
    }
};

void ExpGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto n = (int)dx.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.type.id) {
        case DType::Float32:
        UnaryElementWiseGradKernel<float, ExpGradKernelOp<float>> <<<grideSize, blockSize>>>(
                                                                                            x.data<float>(), 
                                                                                            dx.data<float>(), 
                                                                                            y.data<float>(), 
                                                                                            dy.data<float>(), 
                                                                                            ExpGradKernelOp<float>(), 
                                                                                            n);
        break;
    case DType::Float64:
        UnaryElementWiseGradKernel<double, ExpGradKernelOp<double>> <<<grideSize, blockSize>>>(
                                                                                            x.data<double>(), 
                                                                                            dx.data<double>(), 
                                                                                            y.data<double>(), 
                                                                                            dy.data<double>(), 
                                                                                            ExpGradKernelOp<double>(), 
                                                                                            n);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseGradKernel<half, ExpGradKernelOp<half>> <<<grideSize, blockSize>>>(
                                                                                            x.data<half>(), 
                                                                                            dx.data<half>(), 
                                                                                            y.data<half>(), 
                                                                                            dy.data<half>(), 
                                                                                            ExpGradKernelOp<half>(), 
                                                                                            n);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}



}
}