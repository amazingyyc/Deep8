#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "GPUBinaryElementWise.h"
#include "math/Divide.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct DivideKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y) {
        return x / y;
    }
};

void DivideGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    switch (x.elementType.id) {
        case DType::Float32:
            CallBinaryElementWiseKernel<float, DivideKernelOp<float>>(
                                                            x.data<float>(), x.shape, 
                                                            y.data<float>(), y.shape, 
                                                            z.data<float>(), z.shape, 
                                                            DivideKernelOp<float>());
            break;
        case DType::Float64:
            CallBinaryElementWiseKernel<double, DivideKernelOp<double>>(
                                                            x.data<double>(), x.shape, 
                                                            y.data<double>(), y.shape, 
                                                            z.data<double>(), z.shape, 
                                                            DivideKernelOp<double>());
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            CallBinaryElementWiseKernel<half, DivideKernelOp<half>>(
                                                            x.data<half>(), x.shape, 
                                                            y.data<half>(), y.shape, 
                                                            z.data<half>(), z.shape, 
                                                            DivideKernelOp<half>());
            break;
#endif
    
        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}

template <typename T>
struct DivideGradXKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return dz / y;
    }
};

void DivideGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    switch (dx.elementType.id) {
        case DType::Float32:
        CallBinaryElementWiseGradXKernel<float, DivideGradXKernelOp<float>>(
            x.data<float>(), dx.data<float>(), x.shape,
            y.data<float>(),                   y.shape,
            z.data<float>(), dz.data<float>(), z.shape,
            DivideGradXKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallBinaryElementWiseGradXKernel<double, DivideGradXKernelOp<double>>(
            x.data<double>(), dx.data<double>(), x.shape,
            y.data<double>(),                    y.shape,
            z.data<double>(), dz.data<double>(), z.shape,
            DivideGradXKernelOp<double>()
            );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseGradXKernel<half, DivideGradXKernelOp<half>>(
            x.data<half>(), dx.data<half>(), x.shape,
            y.data<half>(),                  y.shape,
            z.data<half>(), dz.data<half>(), z.shape,
            DivideGradXKernelOp<half>()
            );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct DivideGradYKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return -x * dz / (y * y);
    }
};

void DivideGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    switch (x.elementType.id) {
        case DType::Float32:
            CallBinaryElementWiseGradYKernel<float, DivideGradYKernelOp<float>>(
                x.data<float>(), x.shape,
                y.data<float>(), dy.data<float>(), y.shape,
                z.data<float>(), dz.data<float>(), z.shape,
                DivideGradYKernelOp<float>()
                );
            break;
        case DType::Float64:
            CallBinaryElementWiseGradYKernel<double, DivideGradYKernelOp<double>>(
                x.data<double>(), x.shape,
                y.data<double>(), dy.data<double>(), y.shape,
                z.data<double>(), dz.data<double>(), z.shape,
                DivideGradYKernelOp<double>()
                );
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            CallBinaryElementWiseGradYKernel<half, DivideGradYKernelOp<half>>(
                x.data<half>(), x.shape,
                y.data<half>(), dy.data<half>(), y.shape,
                z.data<half>(), dz.data<half>(), z.shape,
                DivideGradYKernelOp<half>()
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