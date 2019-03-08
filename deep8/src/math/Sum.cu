#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/Sum.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct SumKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
        return T(0);
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
        return ret + cur;
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
        return ret1 + ret2;
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
        return ret;
    }
};

void SumGPU(const Tensor& x, Tensor& y) {
    int size = (int)x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceKernel<float, SumKernelOp<float>>(
            x.data<float>(),
            y.data<float>(),
            size,
            SumKernelOp<float>());
        break;
    case DType::Float64:
        CallReduceKernel<double, SumKernelOp<double>>(
            x.data<double>(),
            y.data<double>(),
            size,
            SumKernelOp<double>());
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, SumKernelOp<half>>(
            x.data<half>(),
            y.data<half>(),
            size,
            SumKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct SumGradKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T& x, const T& y, const T& dy) {
        return dy;
    }
};

void SumGradGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy) {
    auto xsize = (int)x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceGradKernel<float, SumGradKernelOp<float>>(
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            xsize,
            SumGradKernelOp<float>());
        break;
    case DType::Float64:
        CallReduceGradKernel<double, SumGradKernelOp<double>>(
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            xsize,
            SumGradKernelOp<double>());
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceGradKernel<half, SumGradKernelOp<half>>(
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            xsize,
            SumGradKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}
