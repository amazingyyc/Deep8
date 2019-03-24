#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/GPUBinaryElementWise.h"
#include "math/L1Norm.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L1NormKernelOp {
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
		return ret;
	}
};

void L1NormGPU(const Tensor &x, Tensor &y) {
    int batch = (int) x.shape.batch;
    int size  = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailReduceKernel<float, L1NormKernelOp<float>>(
            x.data<float>(), 
            y.data<float>(),
            batch,
            size,
            L1NormKernelOp<float>());
        break;
    case DType::Float64:
        CallTailReduceKernel<double, L1NormKernelOp<double>>(
            x.data<double>(), 
            y.data<double>(),
            batch,
            size,
            L1NormKernelOp<double>());
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailReduceKernel<half, L1NormKernelOp<half>>(
            x.data<half>(), 
            y.data<half>(),
            batch,
            size,
            L1NormKernelOp<half>());
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L1NormGradKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &dy) {
        if (x > T(0)) {
            return dy;
        } else if (T(0) == x) {
            return T(0);
        } else {
            return -dy;
        }
	}
};

void L1NormGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    int batch = (int) x.shape.batch;
    int size  = (int) x.shape.batchSize();

    int N = batch * size;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
        case DType::Float32:
            TailReduceGradKernel<float, L1NormGradKernelOp<float>> << <blockSize, grideSize >> > (
                x.data<float>(), 
                dx.data<float>(), 
                y.data<float>(), 
                dy.data<float>(),
                batch,
                size,
                L1NormGradKernelOp<float>(),
                N);
            break;
        case DType::Float64:
            TailReduceGradKernel<double, L1NormGradKernelOp<double>> << <blockSize, grideSize >> > (
                x.data<double>(), 
                dx.data<double>(), 
                y.data<double>(), 
                dy.data<double>(),
                batch,
                size,
                L1NormGradKernelOp<double>(),
                N);
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            TailReduceGradKernel<half, L1NormGradKernelOp<half>> << <blockSize, grideSize >> > (
                x.data<half>(), 
                dx.data<half>(), 
                y.data<half>(), 
                dy.data<half>(),
                batch,
                size,
                L1NormGradKernelOp<half>(),
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