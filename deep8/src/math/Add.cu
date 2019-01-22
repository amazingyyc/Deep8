#include "math/Add.h"

namespace Deep8 {
namespace Math {

/**
 * Add operator ofr GPU
 */
template <typename T>
struct AddKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y) {
        return x + y;
    }
};

void AddGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    switch (x.type.id) {
    case DType::Float32:
        CallBinaryElementWiseKernel<float, AddKernelOp<float>>(
                                                        x.data<float>(), x.shape, 
                                                        y.data<float>(), y.shape, 
                                                        z.data<float>(), z.shape, 
                                                        AddKernelOp<float>());
        break;
    case DType::Float64:
        CallBinaryElementWiseKernel<double, AddKernelOp<double>>(
                                                        x.data<double>(), x.shape, 
                                                        y.data<double>(), y.shape, 
                                                        z.data<double>(), z.shape, 
                                                        AddKernelOp<double>());
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseKernel<half, AddKernelOp<half>>(
                                                        x.data<half>(), x.shape, 
                                                        y.data<half>(), y.shape, 
                                                        z.data<half>(), z.shape, 
                                                        AddKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


/**
 * Gradient of Add for GPU
 */
template <typename T>
struct AddGradKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const real &x, const real &y, const real &z, const real &dz) {
        return dz;
    }
};

void AddGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    switch (dx.type.id) {
    case DType::Float32:
        CallBinaryElementWiseGradXKernel<float, AddGradKernelOp<float>>(
            x.data<float>(), dx.data<float>(), x.shape,
            y.data<float>(),                 , y.shape,
            z.data<float>(), dz.data<float>(), z.shape,
            AddGradKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallBinaryElementWiseGradXKernel<double, AddGradKernelOp<double>>(
            x.data<double>(), dx.data<double>(), x.shape,
            y.data<double>(),                  , y.shape,
            z.data<double>(), dz.data<double>(), z.shape,
            AddGradKernelOp<double>()
            );
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseGradXKernel<half, AddGradKernelOp<half>>(
            x.data<half>(), dx.data<half>(), x.shape,
            y.data<half>(),                , y.shape,
            z.data<half>(), dz.data<half>(), z.shape,
            AddGradKernelOp<half>()
            );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

void AddGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    switch (dx.type.id) {
    case DType::Float32:
        CallBinaryElementWiseGradYKernel<float, AddGradKernelOp<float>>(
            x.data<float>, x.shape,
            y.data<float>, dy.data<float>(), y.shape,
            z.data<float>, dz.data<float>(), z.shape,
            AddGradKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallBinaryElementWiseGradYKernel<double, AddGradKernelOp<double>>(
            x.data<double>, x.shape,
            y.data<double>, dy.data<double>(), y.shape,
            z.data<double>, dz.data<double>(), z.shape,
            AddGradKernelOp<double>()
            );
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        CallBinaryElementWiseGradYKernel<half, AddGradKernelOp<half>>(
            x.data<half>, x.shape,
            y.data<half>, dy.data<half>(), y.shape,
            z.data<half>, dz.data<half>(), z.shape,
            AddGradKernelOp<half>()
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