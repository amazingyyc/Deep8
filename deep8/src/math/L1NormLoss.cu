#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/L1NormLoss.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L1NormLossKernelOp {
    T ratio;

    L1NormLossKernelOp(T r): ratio(r) {
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

void L1NormLossGPU(const Tensor &x, Tensor &y) {
    int size = (int) x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceKernel<float, L1NormLossKernelOp<float>>(
            x.data<float>(), 
            y.data<float>(), 
            (int)x.shape.size(), 
            L1NormLossKernelOp<float>(1.0/float(size)));
        break;
    case DType::Float64:
        CallReduceKernel<double, L1NormLossKernelOp<double>>(
            x.data<double>(), 
            y.data<double>(), 
            (int)x.shape.size(), 
            L1NormLossKernelOp<double>(1.0/double(size)));
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, L1NormLossKernelOp<half>>(
            x.data<half>(), 
            y.data<half>(), 
            (int)x.shape.size(), 
            L1NormLossKernelOp<half>(__float2half(1.0/float(size))));
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L1NormLossGradKernelOp {
    T ratio;

    L1NormLossGradKernelOp(T r): ratio(r) {
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
		return x >= T (0) ? ratio * dy : -ratio * dy;
	}
};

void L1NormLossGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto size = (int) x.shape.size();

    switch (x.elementType.id) {
        case DType::Float32:
            CallReduceGradKernel<float, L1NormLossGradKernelOp<float> >(
                x.data<float>(), 
                dx.data<float>(), 
                y.data<float>(), 
                dy.data<float>(), 
                (int)dx.shape.size(), 
                L1NormLossGradKernelOp<float>(1.0 / float(size)));
            break;
        case DType::Float64:
            CallReduceGradKernel<double, L1NormLossGradKernelOp<double> >(
                x.data<double>(), 
                dx.data<double>(), 
                y.data<double>(),
                 dy.data<double>(), 
                 (int)dx.shape.size(), 
                 L1NormLossGradKernelOp<double>(1.0 / double(size)));
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            CallReduceGradKernel<half, L1NormLossGradKernelOp<half>>(
                x.data<half>(),
                dx.data<half>(),
                y.data<half>(), 
                dy.data<half>(), 
                (int)dx.shape.size(),
                L1NormLossGradKernelOp<half>(__float2half(1.0 / float(size))));
            break;
#endif

        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}




}
}