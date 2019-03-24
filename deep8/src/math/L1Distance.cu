#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/GPUBinaryElementWise.h"
#include "math/GPUBinaryReduce.h"
#include "math/L1Distance.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct L1DistanceKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur1, T cur2) {
		return ret + cudaAbs(cur1 - cur2);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
		return ret;
	}
};

void L1DistanceGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceKernel<float, L1DistanceKernelOp<float>>(
            x.data<float>(),
            y.data<float>(),
            z.data<float>(),
            row,
            col,
            L1DistanceKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceKernel<double, L1DistanceKernelOp<double>>(
            x.data<double>(),
            y.data<double>(),
            z.data<double>(),
            row,
            col,
            L1DistanceKernelOp<double>()
        );

        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceKernel<half, L1DistanceKernelOp<half>>(
            x.data<half>(),
            y.data<half>(),
            z.data<half>(),
            row,
            col,
            L1DistanceKernelOp<half>()
        );

        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L1DistanceGradXKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        if (x > y) {
            return dz;
        } else if (x == y) {
            return T(0);
        } else {
            return -dz;
        }
	}
};

void L1DistanceGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceGradXKernel<float, L1DistanceGradXKernelOp<float>>(
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            z.data<float>(),
            dz.data<float>(),
            row,
            col,
            L1DistanceGradXKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceGradXKernel<double, L1DistanceGradXKernelOp<double>>(
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            z.data<double>(),
            dz.data<double>(),
            row,
            col,
            L1DistanceGradXKernelOp<double>()
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceGradXKernel<half, L1DistanceGradXKernelOp<half>>(
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            z.data<half>(),
            dz.data<half>(),
            row,
            col,
            L1DistanceGradXKernelOp<half>()
        );
        break;
#endif

        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}

template <typename T>
struct L1DistanceGradYKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        if (y > x) {
            return dz;
        } else if (y == x) {
            return T(0);
        } else {
            return -dz;
        }
	}
};

void L1DistanceGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceGradYKernel<float, L1DistanceGradYKernelOp<float>>(
            x.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            z.data<float>(),
            dz.data<float>(),
            row,
            col,
            L1DistanceGradYKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceGradYKernel<double, L1DistanceGradYKernelOp<double>>(
            x.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            z.data<double>(),
            dz.data<double>(),
            row,
            col,
            L1DistanceGradYKernelOp<double>()
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceGradYKernel<half, L1DistanceGradYKernelOp<half>>(
            x.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            z.data<half>(),
            dz.data<half>(),
            row,
            col,
            L1DistanceGradYKernelOp<half>()
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