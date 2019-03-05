#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/L1Norm.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L1NormKernelOp {
    T ratio;

    L1NormKernelOp(T r): ratio(r) {
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
		return ret + cudaAbs(cur);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
		return ret * ratio;
	}
};

void L1NormGPU(const Tensor &x, Tensor &y) {
    int size = (int) x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceKernel<float, L1NormKernelOp<float>>(
            x.data<float>(), 
            y.data<float>(), 
            (int)x.shape.size(), 
            L1NormKernelOp<float>(1.0/float(size)));
        break;
    case DType::Float64:
        CallReduceKernel<double, L1NormKernelOp<double>>(
            x.data<double>(), 
            y.data<double>(), 
            (int)x.shape.size(), 
            L1NormKernelOp<double>(1.0/double(size)));
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, L1NormKernelOp<half>>(
            x.data<half>(), 
            y.data<half>(), 
            (int)x.shape.size(), 
            L1NormKernelOp<half>(__float2half(1.0/float(size))));
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L1NormGradKernelOp {
    T ratio;

    L1NormGradKernelOp(T r): ratio(r) {
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
		return x >= T (0) ? ratio * dy : -ratio * dy;
	}
};

void L1NormGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto size = (int) x.shape.size();

    switch (x.elementType.id) {
        case DType::Float32:
            CallReduceGradKernel<float, L1NormGradKernelOp<float> >(
                x.data<float>(), 
                dx.data<float>(), 
                y.data<float>(), 
                dy.data<float>(), 
                (int)dx.shape.size(), 
                L1NormGradKernelOp<float>(1.0 / float(size)));
            break;
        case DType::Float64:
            CallReduceGradKernel<double, L1NormGradKernelOp<double> >(
                x.data<double>(), 
                dx.data<double>(), 
                y.data<double>(),
                 dy.data<double>(), 
                 (int)dx.shape.size(), 
                 L1NormGradKernelOp<double>(1.0 / double(size)));
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            CallReduceGradKernel<half, L1NormGradKernelOp<half>>(
                x.data<half>(),
                dx.data<half>(),
                y.data<half>(), 
                dy.data<half>(), 
                (int)dx.shape.size(),
                L1NormGradKernelOp<half>(__float2half(1.0 / float(size))));
            break;
#endif

        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}




}
}