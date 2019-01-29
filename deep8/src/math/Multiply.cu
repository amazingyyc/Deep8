#include "math/Multiply.cu"

namespace Deep8 {
namespace Math {

struct MultiplyKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y) {
        return x * y;
    }
};

void MultiplyGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    switch (x.type.id) {
        case DType::Float32:
            CallBinaryElementWiseKernel<float, MultiplyKernelOp<float>>(
                                                            x.data<float>(), x.shape, 
                                                            y.data<float>(), y.shape, 
                                                            z.data<float>(), z.shape, 
                                                            MultiplyKernelOp<float>());
            break;
        case DType::Float64:
            CallBinaryElementWiseKernel<double, MultiplyKernelOp<double>>(
                                                            x.data<double>(), x.shape, 
                                                            y.data<double>(), y.shape, 
                                                            z.data<double>(), z.shape, 
                                                            MultiplyKernelOp<double>());
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            CallBinaryElementWiseKernel<half, MultiplyKernelOp<half>>(
                                                            x.data<half>(), x.shape, 
                                                            y.data<half>(), y.shape, 
                                                            z.data<half>(), z.shape, 
                                                            MultiplyKernelOp<half>());
            break;
#endif
    
        default:
            DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
            break;
    }
}

/**
 * Gradient X of  for GPU
 */
template <typename T>
struct MultiplyGradXKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return y * dz;
    }
};

void MultiplyGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    switch (dx.type.id) {
        case DType::Float32:
        CallBinaryElementWiseGradXKernel<float, MultiplyGradXKernelOp<float>>(
            x.data<float>(), dx.data<float>(), x.shape,
            y.data<float>(),                 , y.shape,
            z.data<float>(), dz.data<float>(), z.shape,
            MultiplyGradXKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallBinaryElementWiseGradXKernel<double, MultiplyGradXKernelOp<double>>(
            x.data<double>(), dx.data<double>(), x.shape,
            y.data<double>(),                  , y.shape,
            z.data<double>(), dz.data<double>(), z.shape,
            MultiplyGradXKernelOp<double>()
            );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseGradXKernel<half, MultiplyGradXKernelOp<half>>(
            x.data<half>(), dx.data<half>(), x.shape,
            y.data<half>(),                , y.shape,
            z.data<half>(), dz.data<half>(), z.shape,
            MultiplyGradXKernelOp<half>()
            );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

/**
 * Gradient Y of for GPU
 */
template <typename T>
struct MultiplyGradYKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return x * dz;
    }
};

void MultiplyGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    switch (dx.type.id) {
        case DType::Float32:
            CallBinaryElementWiseGradYKernel<float, MultiplyGradYKernelOp<float>>(
                x.data<float>, x.shape,
                y.data<float>, dy.data<float>(), y.shape,
                z.data<float>, dz.data<float>(), z.shape,
                MultiplyGradYKernelOp<float>()
                );
            break;
        case DType::Float64:
            CallBinaryElementWiseGradYKernel<double, MultiplyGradYKernelOp<double>>(
                x.data<double>, x.shape,
                y.data<double>, dy.data<double>(), y.shape,
                z.data<double>, dz.data<double>(), z.shape,
                MultiplyGradYKernelOp<double>()
                );
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            CallBinaryElementWiseGradYKernel<half, MultiplyGradYKernelOp<half>>(
                x.data<half>, x.shape,
                y.data<half>, dy.data<half>(), y.shape,
                z.data<half>, dz.data<half>(), z.shape,
                MultiplyGradYKernelOp<half>()
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