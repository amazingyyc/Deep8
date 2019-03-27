#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/GPUBinaryElementWise.h"
#include "math/GPUBinaryReduce.h"
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
    int batch = (int) x.shape.batch;
    int size  = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailReduceKernel<float, L2NormKernelOp<float>>(
            x.data<float>(), 
            y.data<float>(),
            batch,
            size,
            L2NormKernelOp<float>());
        break;
    case DType::Float64:
        CallTailReduceKernel<double, L2NormKernelOp<double>>(
            x.data<double>(), 
            y.data<double>(),
            batch,
            size,
            L2NormKernelOp<double>());
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailReduceKernel<half, L2NormKernelOp<half>>(
            x.data<half>(), 
            y.data<half>(),
            batch,
            size,
            L2NormKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
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
    int batch = (int) x.shape.batch;
    int size  = (int) x.shape.batchSize();

    int N = batch * size;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
        case DType::Float32:
            TailReduceGradKernel<float, L2NormGradKernelOp<float>> <<<blockSize, grideSize >>> (
                x.data<float>(), 
                dx.data<float>(), 
                y.data<float>(), 
                dy.data<float>(),
                batch,
                size,
                L2NormGradKernelOp<float>(),
                N);
            break;
        case DType::Float64:
            TailReduceGradKernel<double, L2NormGradKernelOp<double>> << <blockSize, grideSize >> > (
                x.data<double>(), 
                dx.data<double>(), 
                y.data<double>(), 
                dy.data<double>(),
                batch,
                size,
                L2NormGradKernelOp<double>(),
                N);
            break;
    
    #ifdef HAVE_HALF
        case DType::Float16:
            TailReduceGradKernel<half, L2NormGradKernelOp<half>> << <blockSize, grideSize >> > (
                x.data<half>(), 
                dx.data<half>(), 
                y.data<half>(), 
                dy.data<half>(),
                batch,
                size,
                L2NormGradKernelOp<half>(),
                N);
            break;
    #endif
    
        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}


}
}