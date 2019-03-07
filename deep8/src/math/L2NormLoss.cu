#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/L2NormLoss.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L2NormLossKernelOp {
    T ratio;

    L2NormLossKernelOp(T r): ratio(r) {
    }

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
        return cudaSqrt(ret) * ratio;
    }
};
    
void L2NormLossGPU(const Tensor &x, Tensor &y) {
    auto xsize = (int) x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceKernel<float, L2NormLossKernelOp<float>>(
            x.data<float>(), 
            y.data<float>(), 
            xsize, 
            L2NormLossKernelOp<float>(1.0 / float(xsize)));
        break;
    case DType::Float64:
        CallReduceKernel<double, L2NormLossKernelOp<double>>(
            x.data<double>(), 
            y.data<double>(), 
            xsize,
            L2NormLossKernelOp<double>(1.0 / double(xsize)));
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, L2NormLossKernelOp<half>>(
            x.data<half>(), 
            y.data<half>(), 
            xsize, 
            L2NormLossKernelOp<half>(__float2half(1.0 / float(xsize))));
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L2NormLossGradKernelOp {
    T ratio;

    L2NormLossGradKernelOp(T r): ratio(r) {
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
		return x * dy * ratio / y;
	}
};

void L2NormLossGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto xsize = (int) x.shape.size();

    switch (x.elementType.id) {
        case DType::Float32:
            CallReduceGradKernel<float, L2NormLossGradKernelOp<float>>(
                x.data<float>(), 
                dx.data<float>(), 
                y.data<float>(), 
                dy.data<float>(), 
                (int)dx.shape.size(), 
                L2NormLossGradKernelOp<float>(1.0 / float(xsize)));
            break;
        case DType::Float64:
            CallReduceGradKernel<double, L2NormLossGradKernelOp<double>>(
                x.data<double>(), 
                dx.data<double>(), 
                y.data<double>(), 
                dy.data<double>(), 
                (int)dx.shape.size(), 
                L2NormLossGradKernelOp<double>(1.0 / double(xsize)));
            break;
    
    #ifdef HAVE_HALF
        case DType::Float16:
            CallReduceGradKernel<half, L2NormLossGradKernelOp<half>>(
                x.data<half>(), 
                dx.data<half>(), 
                y.data<half>(), 
                dy.data<half>(), 
                (int)dx.shape.size(), 
                L2NormLossGradKernelOp<half>(__float2half(1.0 / float(xsize))));
            break;
    #endif
    
        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}

}
}