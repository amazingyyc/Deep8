#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/ReduceMean.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct ReduceMeanKernelOp {
	T ratio;

    ReduceMeanKernelOp(T r) : ratio(r) {
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
		return ret / ratio;
	}
};

void ReduceMeanGPU(const Tensor &x, Tensor &y, const std::vector<int> &axis, bool keepDims) {
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

    int ratio = 1;

    for (int i = 0; i < rank; ++i) {
        if (reduceAxis[i]) {
            yshape[i] = 1;

            ratio *= xshape[i];
        } else {
            yshape[i] = xshape[i];
        }
    }

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceKernel<float, ReduceMeanKernelOp<float>>(
            x.data<float>(), 
            xshape,
            y.data<float>(),
            yshape,
            ReduceMeanKernelOp<float>(float(ratio))
            );
        break;
    case DType::Float64:        
        CallReduceKernel<double, ReduceMeanKernelOp<double>>(
            x.data<double>(), 
            xshape,
            y.data<double>(),
            yshape,
            ReduceMeanKernelOp<double>(double(ratio))
            );
        break;

#ifdef HAVE_HALF
    case DType::Float16:        
        CallReduceKernel<half, ReduceMeanKernelOp<half>>(
            x.data<half>(), 
            xshape,
            y.data<half>(),
            yshape,
            ReduceMeanKernelOp<half>(__float2half(float(ratio)))
            );
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

void ReduceMeanGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const std::vector<int> &axis, bool keepDims) {
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

    int ratio = 1;

    for (int i = 0; i < rank; ++i) {
        if (reduceAxis[i]) {
            yshape[i] = 1;

            ratio *= xshape[i];
        } else {
            yshape[i] = xshape[i];
        }
    }

    switch (x.elementType.id) {
    case DType::Float32:
        CallReduceGradKernel<float, ReduceMeanGradKernelOp<float>>(
            x.data<float>(), 
            dx.data<float>(),
            xshape,
            y.data<float>(), 
            dy.data<float>(), 
            yshape,
            ReduceMeanGradKernelOp<float>(float(ratio))
        );
        break;
    case DType::Float64:
        CallReduceGradKernel<double, ReduceMeanGradKernelOp<double>>(
            x.data<double>(), 
            dx.data<double>(),
            xshape,
            y.data<double>(), 
            dy.data<double>(), 
            yshape,
            ReduceMeanGradKernelOp<double>(double(ratio))
        );
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CallReduceGradKernel<half, ReduceMeanGradKernelOp<half>>(
            x.data<half>(), 
            dx.data<half>(),
            xshape,
            y.data<half>(), 
            dy.data<half>(), 
            yshape,
            ReduceMeanGradKernelOp<half>(__float2half(float(ratio)))
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