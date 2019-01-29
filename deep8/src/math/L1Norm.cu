#incldue "math/GPUReduce.h"
#include "math/L1Norm.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L1NormKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
		return ret + CuMath::cuAbs(cur);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
		return ret;
	}
};

void L1NormGPU(const Tensor &x, Tensor &y) {
    switch (x.type.id) {
    case DType::Float32:
        CallReduceKernel<float, L1NormKernelOp<float>>(x.data<float>(), y.data<float>(), (int)x.shape.size(), L1NormKernelOp<float>());
        break;
    case DType::Float64:
        CallReduceKernel<double, L1NormKernelOp<double>>(x.data<double>(), y.data<double>(), (int)x.shape.size(), L1NormKernelOp<double>());
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, L1NormKernelOp<half>>(x.data<half>(), y.data<half>(), (int)x.shape.size(), L1NormKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
struct L1NormGradKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
		return x >= T (0) ? dy : -dy;
	}
};

void L1NormGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
switch (x.type.id) {
    case DType::Float32:
        CallReduceGradKernel<float, L1NormGradKernelOp<float>>(x.data<float>(), dx.data<float>(), dy.data<float>(), (int)dx.shape.size(), L1NormGradKernelOp<float>());
        break;
    case DType::Float64:
        CallReduceGradKernel<double, L1NormGradKernelOp<double>>(x.data<double>(), dx.data<double>(), dy.data<double>(), (int)dx.shape.size(), L1NormGradKernelOp<double>());
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceGradKernel<half, L1NormGradKernelOp<half>>(x.data<half>(), dx.data<half>(), dy.data<half>(), (int)dx.shape.size(), L1NormGradKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}




}
}