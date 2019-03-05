#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/L2Norm.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L2NormKernelOp {
    T ratio;

    L2NormKernelOp(T r): ratio(r) {
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
    
void L2NormGPU(const Tensor &x, Tensor &y) {
    auto xsize = (int) x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceKernel<float, L2NormKernelOp<float>>(
            x.data<float>(), 
            y.data<float>(), 
            xsize, 
            L2NormKernelOp<float>(1.0 / float(xsize)));
        break;
    case DType::Float64:
        CallReduceKernel<double, L2NormKernelOp<double>>(
            x.data<double>(), 
            y.data<double>(), 
            xsize,
            L2NormKernelOp<double>(1.0 / double(xsize)));
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, L2NormKernelOp<half>>(
            x.data<half>(), 
            y.data<half>(), 
            xsize, 
            L2NormKernelOp<half>(__float2half(1.0 / float(xsize))));
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L2NormGradKernelOp {
    T ratio;

    L2NormGradKernelOp(T r): ratio(r) {
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
		return x * dy * ratio / y;
	}
};

void L2NormGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto xsize = (int) x.shape.size();

    switch (x.elementType.id) {
        case DType::Float32:
            CallReduceGradKernel<float, L2NormGradKernelOp<float>>(
                x.data<float>(), 
                dx.data<float>(), 
                y.data<float>(), 
                dy.data<float>(), 
                (int)dx.shape.size(), 
                L2NormGradKernelOp<float>(1.0 / float(xsize)));
            break;
        case DType::Float64:
            CallReduceGradKernel<double, L2NormGradKernelOp<double>>(
                x.data<double>(), 
                dx.data<double>(), 
                y.data<double>(), 
                dy.data<double>(), 
                (int)dx.shape.size(), 
                L2NormGradKernelOp<double>(1.0 / double(xsize)));
            break;
    
    #ifdef HAVE_HALF
        case DType::Float16:
            CallReduceGradKernel<half, L2NormGradKernelOp<half>>(
                x.data<half>(), 
                dx.data<half>(), 
                y.data<half>(), 
                dy.data<half>(), 
                (int)dx.shape.size(), 
                L2NormGradKernelOp<half>(__float2half(1.0 / double(xsize))));
            break;
    #endif
    
        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}

}
}