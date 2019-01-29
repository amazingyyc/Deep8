#include "math/Minus.h"

namespace Deep8 {
namespace Math {

/**
 * Minus operator ofr GPU
 */
template <typename T>
struct MinusKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y) {
        return x - y;
    }
};

void MinusGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    switch (x.type.id) {
        case DType::Float32:
            CallBinaryElementWiseKernel<float, MinusKernelOp<float>>(
                                                            x.data<float>(), x.shape, 
                                                            y.data<float>(), y.shape, 
                                                            z.data<float>(), z.shape, 
                                                            MinusKernelOp<float>());
            break;
        case DType::Float64:
            CallBinaryElementWiseKernel<double, MinusKernelOp<double>>(
                                                            x.data<double>(), x.shape, 
                                                            y.data<double>(), y.shape, 
                                                            z.data<double>(), z.shape, 
                                                            MinusKernelOp<double>());
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            CallBinaryElementWiseKernel<half, MinusKernelOp<half>>(
                                                            x.data<half>(), x.shape, 
                                                            y.data<half>(), y.shape, 
                                                            z.data<half>(), z.shape, 
                                                            MinusKernelOp<half>());
            break;
#endif
    
        default:
            DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
            break;
        }
}

/**
 * Gradient X of Minus for GPU
 */
template <typename T>
struct MinusGradXKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return dz;
    }
};

void MinusGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    switch (dx.type.id) {
        case DType::Float32:
        CallBinaryElementWiseGradXKernel<float, MinusGradXKernelOp<float>>(
            x.data<float>(), dx.data<float>(), x.shape,
            y.data<float>(),                 , y.shape,
            z.data<float>(), dz.data<float>(), z.shape,
            MinusGradXKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallBinaryElementWiseGradXKernel<double, MinusGradXKernelOp<double>>(
            x.data<double>(), dx.data<double>(), x.shape,
            y.data<double>(),                  , y.shape,
            z.data<double>(), dz.data<double>(), z.shape,
            MinusGradXKernelOp<double>()
            );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseGradXKernel<half, MinusGradXKernelOp<half>>(
            x.data<half>(), dx.data<half>(), x.shape,
            y.data<half>(),                , y.shape,
            z.data<half>(), dz.data<half>(), z.shape,
            MinusGradXKernelOp<half>()
            );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


/**
 * Gradient Y of Minus for GPU
 */
template <typename T>
struct MinusGradYKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return -dz;
    }
};

void MinusGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    switch (dx.type.id) {
        case DType::Float32:
            CallBinaryElementWiseGradYKernel<float, MinusGradYKernelOp<float>>(
                x.data<float>, x.shape,
                y.data<float>, dy.data<float>(), y.shape,
                z.data<float>, dz.data<float>(), z.shape,
                MinusGradYKernelOp<float>()
                );
            break;
        case DType::Float64:
            CallBinaryElementWiseGradYKernel<double, MinusGradYKernelOp<double>>(
                x.data<double>, x.shape,
                y.data<double>, dy.data<double>(), y.shape,
                z.data<double>, dz.data<double>(), z.shape,
                MinusGradYKernelOp<double>()
                );
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            CallBinaryElementWiseGradYKernel<half, MinusGradYKernelOp<half>>(
                x.data<half>, x.shape,
                y.data<half>, dy.data<half>(), y.shape,
                z.data<half>, dz.data<half>(), z.shape,
                MinusGradYKernelOp<half>()
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