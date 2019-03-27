#include "utils/ShapeUtils.h"
#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/ReduceSum.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct ReduceSumKernelOp {
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
		return ret;
	}
};

void ReduceSumGPU(const Tensor& x, Tensor& y, const std::vector<int>& axis, bool keepDims) {
    auto xshape = convertShapeToVector(x.shape);

    int rank = xshape.size();

    std::vector<bool> reduceAxis(rank);

    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = true;
        }
    } else {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = false;
        }

        for (int i = 0; i < axis.size(); ++i) {
            DEEP8_ARGUMENT_CHECK(-rank <= axis[i] && axis[i] < rank, "the reduce dims is error");

            reduceAxis[(axis[i] + rank) % rank] = true;
        }
    }

    auto yshape = xshape;

    for (int i = 0; i < rank; ++i) {
        if (reduceAxis[i]) {
            yshape[i] = 1;
        } else {
            yshape[i] = xshape[i];
        }
    }

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceKernel<float, ReduceSumKernelOp<float>>(
            x.data<float>(),
            xshape,
            y.data<float>(),
            yshape,
            ReduceSumKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallReduceKernel<double, ReduceSumKernelOp<double>>(
            x.data<double>(),
            xshape,
            y.data<double>(),
            yshape,
            ReduceSumKernelOp<double>()
            );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceKernel<half, ReduceSumKernelOp<half>>(
            x.data<half>(),
            xshape,
            y.data<half>(),
            yshape,
            ReduceSumKernelOp<half>()
            );
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct ReduceSumGradKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(T x, T y, T dy) {
		return dy;
	}
};

void ReduceSumGradGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy, const std::vector<int>& axis, bool keepDims) {
    auto xshape = convertShapeToVector(x.shape);

    int rank = xshape.size();

    std::vector<bool> reduceAxis(rank);

    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = true;
        }
    } else {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = false;
        }

        for (int i = 0; i < axis.size(); ++i) {
            DEEP8_ARGUMENT_CHECK(-rank <= axis[i] && axis[i] < rank, "the reduce dims is error");

            reduceAxis[(axis[i] + rank) % rank] = true;
        }
    }

    auto yshape = xshape;

    for (int i = 0; i < rank; ++i) {
        if (reduceAxis[i]) {
            yshape[i] = 1;
        } else {
            yshape[i] = xshape[i];
        }
    }

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceGradKernel<float, ReduceSumGradKernelOp<float>>(
            x.data<float>(),
            dx.data<float>(),
            xshape,
            y.data<float>(),
            dy.data<float>(),
            yshape,
            ReduceSumGradKernelOp<float>()
            );
        break;
    case DType::Float64:
        CallReduceGradKernel<double, ReduceSumGradKernelOp<double>>(
            x.data<double>(),
            dx.data<double>(),
            xshape,
            y.data<double>(),
            dy.data<double>(),
            yshape,
            ReduceSumGradKernelOp<double>()
            );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceGradKernel<half, ReduceSumGradKernelOp<half>>(
            x.data<half>(),
            dx.data<half>(),
            xshape,
            y.data<half>(),
            dy.data<half>(),
            yshape,
            ReduceSumGradKernelOp<half>()
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