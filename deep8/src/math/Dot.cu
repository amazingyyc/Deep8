#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/GPUBinaryElementWise.h"
#include "math/GPUBinaryReduce.h"
#include "math/Dot.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct DotKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur1, T cur2) {
		return ret + cur1 * cur2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
		return ret;
	}
};

void DotGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceKernel<float, DotKernelOp<float>>(
            x.data<float>(),
            y.data<float>(),
            z.data<float>(),
            row,
            col,
            DotKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceKernel<double, DotKernelOp<double>>(
            x.data<double>(),
            y.data<double>(),
            z.data<double>(),
            row,
            col,
            DotKernelOp<double>()
        );

        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceKernel<half, DotKernelOp<half>>(
            x.data<half>(),
            y.data<half>(),
            z.data<half>(),
            row,
            col,
            DotKernelOp<half>()
        );

        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct DotGradXKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return y * dz;
	}
};

void DotGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceGradXKernel<float, DotGradXKernelOp<float>>(
            x.data<float>(),
            dx.data<float>(),
            y.data<float>(),
            z.data<float>(),
            dz.data<float>(),
            row,
            col,
            DotGradXKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceGradXKernel<double, DotGradXKernelOp<double>>(
            x.data<double>(),
            dx.data<double>(),
            y.data<double>(),
            z.data<double>(),
            dz.data<double>(),
            row,
            col,
            DotGradXKernelOp<double>()
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceGradXKernel<half, DotGradXKernelOp<half>>(
            x.data<half>(),
            dx.data<half>(),
            y.data<half>(),
            z.data<half>(),
            dz.data<half>(),
            row,
            col,
            DotGradXKernelOp<half>()
        );
        break;
#endif

        default:
            DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
            break;
    }
}

template <typename T>
struct DotGradYKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y, const T &z, const T &dz) {
        return x * dz;
	}
};

void DotGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    int row = (int) x.shape.batch;
    int col = (int) x.shape.batchSize();

    switch (x.elementType.id) {
    case DType::Float32:
        CallTailBinaryReduceGradYKernel<float, DotGradYKernelOp<float>>(
            x.data<float>(),
            y.data<float>(),
            dy.data<float>(),
            z.data<float>(),
            dz.data<float>(),
            row,
            col,
            DotGradYKernelOp<float>()
        );

        break;
    case DType::Float64:
        CallTailBinaryReduceGradYKernel<double, DotGradYKernelOp<double>>(
            x.data<double>(),
            y.data<double>(),
            dy.data<double>(),
            z.data<double>(),
            dz.data<double>(),
            row,
            col,
            DotGradYKernelOp<double>()
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallTailBinaryReduceGradYKernel<half, DotGradYKernelOp<half>>(
            x.data<half>(),
            y.data<half>(),
            dy.data<half>(),
            z.data<half>(),
            dz.data<half>(),
            row,
            col,
            DotGradYKernelOp<half>()
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