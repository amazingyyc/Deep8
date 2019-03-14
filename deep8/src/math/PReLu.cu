#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/GPUBinaryElementWise.h"
#include "math/PReLu.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct PReLuKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T& x, const T& y) {
        return x >= T(0) ? x : x * y;
    }
};

void PReLuGPU(const Tensor& x, const Tensor& y, Tensor& z) {
    switch (x.elementType.id) {
    case DType::Float32:
        CallBinaryElementWiseKernel<float, PReLuKernelOp<float>>(
            x.data<float>(), x.shape,
            y.data<float>(), y.shape,
            z.data<float>(), z.shape,
            PReLuKernelOp<float>());
        break;
    case DType::Float64:
        CallBinaryElementWiseKernel<double, PReLuKernelOp<double>>(
            x.data<double>(), x.shape,
            y.data<double>(), y.shape,
            z.data<double>(), z.shape,
            PReLuKernelOp<double>());
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseKernel<half, PReLuKernelOp<half>>(
            x.data<half>(), x.shape,
            y.data<half>(), y.shape,
            z.data<half>(), z.shape,
            PReLuKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct PReLuGradXKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T& x, const T& y, const T& z, const T& dz) {
        return x >= T(0) ? dz : y * dz;
    }
};

void PReLuGradXGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz) {
    switch (dx.elementType.id) {
    case DType::Float32:
        CallBinaryElementWiseGradXKernel<float, PReLuGradXKernelOp<float>>(
            x.data<float>(), dx.data<float>(), x.shape,
            y.data<float>(), y.shape,
            z.data<float>(), dz.data<float>(), z.shape,
            PReLuGradXKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallBinaryElementWiseGradXKernel<double, PReLuGradXKernelOp<double>>(
            x.data<double>(), dx.data<double>(), x.shape,
            y.data<double>(), y.shape,
            z.data<double>(), dz.data<double>(), z.shape,
            PReLuGradXKernelOp<double>()
            );
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseGradXKernel<half, PReLuGradXKernelOp<half>>(
            x.data<half>(), dx.data<half>(), x.shape,
            y.data<half>(), y.shape,
            z.data<half>(), dz.data<half>(), z.shape,
            PReLuGradXKernelOp<half>()
            );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct PReLuGradYKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T& x, const T& y, const T& z, const T& dz) {
        return x >= T(0) ? T(0) : dz * x;
    }
};

void PReLuGradYGPU(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz) {
    switch (x.elementType.id) {
    case DType::Float32:
        CallBinaryElementWiseGradYKernel<float, PReLuGradYKernelOp<float>>(
            x.data<float>(), x.shape,
            y.data<float>(), dy.data<float>(), y.shape,
            z.data<float>(), dz.data<float>(), z.shape,
            PReLuGradYKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallBinaryElementWiseGradYKernel<double, PReLuGradYKernelOp<double>>(
            x.data<double>(), x.shape,
            y.data<double>(), dy.data<double>(), y.shape,
            z.data<double>(), dz.data<double>(), z.shape,
            PReLuGradYKernelOp<double>()
            );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseGradYKernel<half, PReLuGradYKernelOp<half>>(
            x.data<half>(), x.shape,
            y.data<half>(), dy.data<half>(), y.shape,
            z.data<half>(), dz.data<half>(), z.shape,
            PReLuGradYKernelOp<half>()
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