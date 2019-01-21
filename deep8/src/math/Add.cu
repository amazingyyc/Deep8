#include "math/Add.h"

namespace Deep8 {
namespace Math {

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

}
}