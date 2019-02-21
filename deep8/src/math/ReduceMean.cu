#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/ReduceMean.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct ReduceMeanKernelOp {
	T ratios;

    ReduceMeanKernelOp(T r) : ratios(r) {
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return 0;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
		return ret + cur;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
		return ret / ratios;
	}
};

void ReduceMeanGPU(const Tensor &x, Tensor &y, int axis) {
    auto shape = x.shape;
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) shape.dim(i);
        }

        for (int i = axis + 1; i < shape.nDims; ++i) {
            dim2 *= (int) shape.dim(i);
        }
    }

    float fratio = float(dim1);
    double dratio = double(dim1);

#ifdef HAVE_HALF
    half hratio = float(dim1);
#endif

    switch (x.elementType.id) {
    case DType::Float32:
        if (1 == dim2) {
            CallTailReduceKernel<float, ReduceMeanKernelOp<float>>(x.data<float>(), y.data<float>(), dim0, dim1, ReduceMeanKernelOp<float>(fratio));
        } else if (1 == dim0) {
            CallHeadReduceKernel<float, ReduceMeanKernelOp<float>>(x.data<float>(), y.data<float>(), dim1, dim2, ReduceMeanKernelOp<float>(fratio));
        } else {
            int N = dim0 * dim2;
            int blockSize = DEEP8_GPU_BLOCK_SIZE;
            int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

            MiddleReduceKernel<float, ReduceMeanKernelOp<float>> <<<grideSize, blockSize>>>(x.data<float>(), y.data<float>(), dim0, dim1, dim2, ReduceMeanKernelOp<float>(fratio), N);
        }

        break;
    case DType::Float64:        
		if (1 == dim2) {
            CallTailReduceKernel<double, ReduceMeanKernelOp<double>>(x.data<double>(), y.data<double>(), dim0, dim1, ReduceMeanKernelOp<double>(dratio));
        } else if (1 == dim0) {
            CallHeadReduceKernel<double, ReduceMeanKernelOp<double>>(x.data<double>(), y.data<double>(), dim1, dim2, ReduceMeanKernelOp<double>(dratio));
        } else {
            int N = dim0 * dim2;
            int blockSize = DEEP8_GPU_BLOCK_SIZE;
            int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

            MiddleReduceKernel<double, ReduceMeanKernelOp<double>> <<<grideSize, blockSize>>>(x.data<double>(), y.data<double>(), dim0, dim1, dim2, ReduceMeanKernelOp<double>(dratio), N);
        }

        break;

#ifdef HAVE_HALF
    case DType::Float16:        
		if (1 == dim2) {
            CallTailReduceKernel<half, ReduceMeanKernelOp<half>>(x.data<half>(), y.data<half>(), dim0, dim1, ReduceMeanKernelOp<half>(hratio));
        } else if (1 == dim0) {
            CallHeadReduceKernel<half, ReduceMeanKernelOp<half>>(x.data<half>(), y.data<half>(), dim1, dim2, ReduceMeanKernelOp<half>(hratio));
        } else {
            int N = dim0 * dim2;
            int blockSize = DEEP8_GPU_BLOCK_SIZE;
            int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

            MiddleReduceKernel<half, ReduceMeanKernelOp<half>> <<<grideSize, blockSize>>>(x.data<half>(), y.data<half>(), dim0, dim1, dim2, ReduceMeanKernelOp<half>(hratio), N);
        }

        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct ReduceMeanGradKernelOp {
    T ratio;

    ReduceMeanGradKernelOp(T r) : ratio(r) {}

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(T x, T y, T dy) {
        return dy / ratio;
    }
};

void ReduceMeanGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis) {
	auto shape = x.shape;
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) shape.dim(i);
        }

        for (int i = axis + 1; i < shape.nDims; ++i) {
            dim2 *= (int) shape.dim(i);
        }
    }

    int N = dim0 * dim2;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
    
    float fratio = float(dim1);
    double dratio = double(dim1);

#ifdef HAVE_HALF
    half hratio = float(dim1);
#endif

    switch (x.elementType.id) {
    case DType::Float32:
        MiddleReduceGradKernel<float, ReduceMeanGradKernelOp<float>> <<<grideSize, blockSize>>>(
            x.data<float>(), 
            dx.data<float>(), 
            y.data<float>(), 
            dy.data<float>(), 
            dim0, 
            dim1, 
            dim2, 
            ReduceMeanGradKernelOp<float>(fratio),
            N);
        break;
    case DType::Float64:
        MiddleReduceGradKernel<double, ReduceMeanGradKernelOp<double>> <<<grideSize, blockSize>>>(
            x.data<double>(), 
            dx.data<double>(), 
            y.data<double>(), 
            dy.data<double>(), 
            dim0, 
            dim1, 
            dim2, 
            ReduceMeanGradKernelOp<double>(dratio),
            N);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        MiddleReduceGradKernel<half, ReduceMeanGradKernelOp<half>> <<<grideSize, blockSize>>>(
            x.data<half>(), 
            dx.data<half>(), 
            y.data<half>(), 
            dy.data<half>(), 
            dim0, 
            dim1, 
            dim2, 
            ReduceMeanGradKernelOp<half>(hratio),
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