#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUUnaryElementWise.h"
#include "math/GPUMath.h"
#include "math/Abs.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct AbsKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x) {
        return cudaAbs(x);
    }
};

template <typename T>
void AbsGPUImpl(const T *x, T *y, int n) {
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    UnaryElementWiseKernel<T, AbsKernelOp<T>> << <grideSize, blockSize >> > (x, y, AbsKernelOp<T>(), n);
}

void AbsGPU(const Tensor &x, Tensor &y) {
    auto n = (int) x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        AbsGPUImpl<float>(x.data<float>(), y.data<float>(), n);
        break;
    case DType::Float64:
        AbsGPUImpl<double>(x.data<double>(), y.data<double>(), n);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        AbsGPUImpl<half>(x.data<half>(), y.data<half>(), n);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct AbsGradKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
        if (x > T(0)) {
            return dy;
        } else if (T(0) == x) {
            return T(0);
        } else {
            return -dy;
        }
    }
};

template <typename T>
void AbsGradGPUImpl(const T *x, T *dx, const T *y, const T *dy, int n) {
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    UnaryElementWiseGradKernel<T, AbsGradKernelOp<T>> << <grideSize, blockSize >> > (x, dx, y, dy, AbsGradKernelOp<T>(), n);
}

void AbsGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto n = (int)x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        AbsGradGPUImpl<float>(x.data<float>(), dx.data<float>(), y.data<float>(), dy.data<float>(), n);
        break;
    case DType::Float64:
        AbsGradGPUImpl<double>(x.data<double>(), dx.data<double>(), y.data<double>(), dy.data<double>(), n);
        break;
        
#ifdef HAVE_HALF
    case DType::Float16:
        AbsGradGPUImpl<half>(x.data<half>(), dx.data<half>(), y.data<half>(), dy.data<half>(), n);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}