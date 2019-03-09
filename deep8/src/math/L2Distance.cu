#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/GPUBinaryElementWise.h"
#include "math/GPUBinaryReduce.h"
#include "math/L2Distance.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L2DistanceKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur1, T cur2) {
        T diff = cur1 - cur2;
		return ret + diff * diff;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
		return cudaSqrt(ret);
	}
};

void L2DistanceGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceKernel<float, L2DistanceKernelOp<float>>(
            x.data<float>(),
            y.data<float>(),
            z.data<float>(),
            row,
            col,
            L2DistanceKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceKernel<double, L2DistanceKernelOp<double>>(
            x.data<double>(),
            y.data<double>(),
            z.data<double>(),
            row,
            col,
            L2DistanceKernelOp<double>()
        );

        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceKernel<half, L2DistanceKernelOp<half>>(
            x.data<half>(),
            y.data<half>(),
            z.data<half>(),
            row,
            col,
            L2DistanceKernelOp<half>()
        );

        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L2DistanceGradXKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return dz * (x - y) / z;
	}
};

void L2DistanceGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceGradXKernel<float, L2DistanceGradXKernelOp<float>>(
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            z.data<float>(),
            dz.data<float>(),
            row,
            col,
            L2DistanceGradXKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceGradXKernel<double, L2DistanceGradXKernelOp<double>>(
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            z.data<double>(),
            dz.data<double>(),
            row,
            col,
            L2DistanceGradXKernelOp<double>()
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceGradXKernel<half, L2DistanceGradXKernelOp<half>>(
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            z.data<half>(),
            dz.data<half>(),
            row,
            col,
            L2DistanceGradXKernelOp<half>()
        );
        break;
#endif

        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}

template <typename T>
struct L2DistanceGradYKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return (y - x) * dz / z;
	}
};

void L2DistanceGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceGradYKernel<float, L2DistanceGradYKernelOp<float>>(
            x.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            z.data<float>(),
            dz.data<float>(),
            row,
            col,
            L2DistanceGradYKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceGradYKernel<double, L2DistanceGradYKernelOp<double>>(
            x.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            z.data<double>(),
            dz.data<double>(),
            row,
            col,
            L2DistanceGradYKernelOp<double>()
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceGradYKernel<half, L2DistanceGradYKernelOp<half>>(
            x.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            z.data<half>(),
            dz.data<half>(),
            row,
            col,
            L2DistanceGradYKernelOp<half>()
        );
        break;
#endif

        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}




}
}