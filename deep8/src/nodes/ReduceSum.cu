#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUReduce.h"
#include "ReduceSum.h"

namespace Deep8 {

template <typename T>
struct ReduceSumForwardOp {
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

template <typename T>
struct ReduceSumBackwardOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(T x, T y, T dy) {
		return dy;
	}
};

template <typename T>
void ReduceSum<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto shape = inputs[0]->shape;
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

    if (1 == dim2) {
        /**tail reduce*/
        callTailReduceForward<T, ReduceSumForwardOp<T>>(inputs[0]->data(), output->data(), dim0, dim1);
    } else if (1 == dim0) {
        /**head reduce*/
        callHeadReduceForward<T, ReduceSumForwardOp<T>>(inputs[0]->data(), output->data(), dim1, dim2);
    } else {
		/**middle reduce*/
        int N = dim0 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        MiddleReduceForward<T, ReduceSumForwardOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(inputs[0]->data(), output->data(), dim0, dim1, dim2, ReduceSumForwardOp<T>(), N);
    }
}

template <typename T>
void ReduceSum<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of ReduceSum backwardCPU is error");

    auto shape = iGradient->shape;
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
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
    
    MiddleReduceBackward<T, ReduceSumBackwardOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(
        inputs[0]->data(), iGradient->data(), output->data(), outputGradient->data(), dim0, dim1, dim2, ReduceSumBackwardOp<T>(), N);
}

DEEP8_DECLARATION_GPU_FUNC(ReduceSum);

}