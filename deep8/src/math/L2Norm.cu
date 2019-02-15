#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/L2Norm.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L2NormKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
        return T(0);
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
        return ret + cur * cur;
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
        return ret1 + ret2;
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
        return cudaSqrt(ret);
    }
};
    
void L2NormGPU(const Tensor &x, Tensor &y) {
    switch (x.type.id) {
    case DType::Float32:
        CallReduceKernel<float, L2NormKernelOp<float>>(x.data<float>(), y.data<float>(), (int)x.shape.size(), L2NormKernelOp<float>());
        break;
    case DType::Float64:
        CallReduceKernel<double, L2NormKernelOp<double>>(x.data<double>(), y.data<double>(), (int)x.shape.size(), L2NormKernelOp<double>());
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, L2NormKernelOp<half>>(x.data<half>(), y.data<half>(), (int)x.shape.size(), L2NormKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
struct L2NormGradKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
		return x * dy / y;
	}
};

void L2NormGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    switch (x.type.id) {
        case DType::Float32:
            CallReduceGradKernel<float, L2NormGradKernelOp<float>>(x.data<float>(), dx.data<float>(), dy.data<float>(), (int)dx.shape.size(), L2NormGradKernelOp<float>());
            break;
        case DType::Float64:
            CallReduceGradKernel<double, L2NormGradKernelOp<double>>(x.data<double>(), dx.data<double>(), dy.data<double>(), (int)dx.shape.size(), L2NormGradKernelOp<double>());
            break;
    
    #ifdef HAVE_HALF
        case DType::Float16:
            CallReduceGradKernel<half, L2NormGradKernelOp<half>>(x.data<half>(), dx.data<half>(), dy.data<half>(), (int)dx.shape.size(), L2NormGradKernelOp<half>());
            break;
    #endif
    
        default:
            DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
            break;
        }
    }

}
}