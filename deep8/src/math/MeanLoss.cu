#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/MeanLoss.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct MeanLossKernelOp {
    T ratio;

    MeanLossKernelOp(T r) : ratio(r) {
    }

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
        return ret * ratio;
    }
};

void MeanLossGPU(const Tensor& x, Tensor& y) {
    int size = (int)x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceKernel<float, MeanLossKernelOp<float>>(
            x.data<float>(),
            y.data<float>(),
            size,
            MeanLossKernelOp<float>(1.0 / float(size)));
        break;
    case DType::Float64:
        CallReduceKernel<double, MeanLossKernelOp<double>>(
            x.data<double>(),
            y.data<double>(),
            size,
            MeanLossKernelOp<double>(1.0 / double(size)));
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, MeanLossKernelOp<half>>(
            x.data<half>(),
            y.data<half>(),
            size,
            MeanLossKernelOp<half>(__float2half(1.0 / float(size))));
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct MeanLossGradKernelOp {
    T ratio;

    MeanLossGradKernelOp(T r) : ratio(r) {
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T& x, const T& y, const T& dy) {
        return dy * ratio;
    }
};

void MeanLossGradGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy) {
    auto xsize = (int)x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceGradKernel<float, MeanLossGradKernelOp<float>>(
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            xsize,
            MeanLossGradKernelOp<float>(1.0 / float(xsize)));
        break;
    case DType::Float64:
        CallReduceGradKernel<double, MeanLossGradKernelOp<double>>(
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            xsize,
            MeanLossGradKernelOp<double>(1.0 / double(xsize)));
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceGradKernel<half, MeanLossGradKernelOp<half>>(
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            xsize,
            MeanLossGradKernelOp<half>(__float2half(1.0 / float(xsize))));
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


}
}