#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.h"
#include "GPUReduce.h"
#include "LogSoftmax.h"

namespace Deep8 {

/**
 * find the max value and put it in y
 */
template <typename T>
struct LogSoftmaxFindMaxOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
        return CuMath::cuMinValue<T>();
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
        return ret >= cur ? ret : cur;
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
        return ret1 >= ret2 ? ret1 : ret2;
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
        return ret;
    }
};

template <typename real>
struct LogSoftmaxExpMinusOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real forward(const real &x, const real &y) {
        return CuMath::cuExp(x - y);
    }
};

template <typename T>
struct LogSoftmaxSumLogOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
        return T(0);
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
        return ret + cur;
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
        return ret1 + ret2;
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
        return CuMath::cuLog(ret);
    }
};

template <typename real>
__global__ void LogSoftmaxForwardKernel(const real *x, const real *maxptr, const real sumlogptr, real *y, const int dim0, const int dim1, const int dim2, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int d0 = i / (dim1 * dim2);
        int d2 = i % dim2;

        int j = d0 * dim2 + d2;

        y[i] = x[i] - maxptr[j] - sumlogptr[j];
    }
}

template <typename T>
struct LogSoftmaxSumOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
        return T(0);
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

template <typename real>
__global__ void LogSoftmaxBackwardKernel(real *dx, const real *y, const real *dy, const real *sumptr, const int dim0, const int dim1, const int dim2, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int d0 = i / (dim1 * dim2);
        int d2 = i % dim2;

        int j = d0 * dim2 + d2;

        dx[i] += dy[i] - CuMath::cuExp(y[i]) * sumptr[j];
    }
}

template <typename T>
void LogSoftmax<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = (GPUDevice*)output->device();

    auto x = inputs[0]->data();
    auto y = output->data();

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

    auto maxptr    = (T*)device->malloc(sizeof(T) * dim0 * dim2);
    auto sumlogptr = (T*)device->malloc(sizeof(T) * dim0 * dim2);

    if (1 == dim2) {
        callTailReduceForward<T, LogSoftmaxFindMaxOp<T>>(x, maxptr, dim0, dim1, LogSoftmaxFindMaxOp<T>());
    } else if (1 == dim0) {
        callHeadReduceForward<T, LogSoftmaxFindMaxOp<T>>(x, maxptr, dim1, dim2, LogSoftmaxFindMaxOp<T>());
    } else {
        int N = dim0 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
        MiddleReduceForward<T, LogSoftmaxFindMaxOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, maxptr, dim0, dim1, dim2, LogSoftmaxFindMaxOp<T>(), N);
    }

    /**y = exp(x - max)*/
    if (1 == dim2) {
        NVShape<2> xNVShape;
        NVShape<2> maxNVShape;
        NVShape<2> yNVShape;

        xNVShape.dims[0] = dim0;
        xNVShape.dims[1] = dim1;
        xNVShape.strides[0] = dim1;
        xNVShape.strides[1] = 1;

        maxNVShape.dims[0] = dim0;
        maxNVShape.dims[1] = 1;
        maxNVShape.strides[0] = 1;
        maxNVShape.strides[1] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.strides[0] = dim1;
        yNVShape.strides[1] = 1;

        int N = dim0 * dim1;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, LogSoftmaxExpMinusOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, maxptr, maxNVShape, y, yNVShape, LogSoftmaxExpMinusOp<T>(), N);
    } else if (1 == dim0) {
        NVShape<2> xNVShape;
        NVShape<2> maxNVShape;
        NVShape<2> yNVShape;

        xNVShape.dims[0] = dim1;
        xNVShape.dims[1] = dim2;
        xNVShape.strides[0] = dim2;
        xNVShape.strides[1] = 1;

        maxNVShape.dims[0] = 1;
        maxNVShape.dims[1] = dim2;
        maxNVShape.strides[0] = dim2;
        maxNVShape.strides[1] = 1;

        yNVShape.dims[0] = dim1;
        yNVShape.dims[1] = dim2;
        yNVShape.strides[0] = dim2;
        yNVShape.strides[1] = 1;

        int N = dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, LogSoftmaxExpMinusOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, maxptr, maxNVShape, y, yNVShape, LogSoftmaxExpMinusOp<T>(), N);
    } else {
        NVShape<3> xNVShape;
        NVShape<3> maxNVShape;
        NVShape<3> yNVShape;

        xNVShape.dims[0] = dim0;
        xNVShape.dims[1] = dim1;
        xNVShape.dims[2] = dim2;
        xNVShape.strides[0] = dim1 * dim2;
        xNVShape.strides[1] = dim2;
        xNVShape.strides[2] = 1;

        maxNVShape.dims[0] = dim0;
        maxNVShape.dims[1] = 1;
        maxNVShape.dims[2] = dim2;
        maxNVShape.strides[0] = dim2;
        maxNVShape.strides[1] = dim2;
        maxNVShape.strides[2] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.dims[2] = dim2;
        yNVShape.strides[0] = dim1 * dim2;
        yNVShape.strides[1] = dim2;
        yNVShape.strides[2] = 1;

        int N = dim0 * dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, LogSoftmaxExpMinusOp<T>, 3> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, maxptr, maxNVShape, y, yNVShape, LogSoftmaxExpMinusOp<T>(), N);
    }

    /**calculate sum*/
    if (1 == dim2) {
        /**tail reduce*/
        callTailReduceForward<T, LogSoftmaxSumLogOp<T>>(y, sumlogptr, dim0, dim1, LogSoftmaxSumLogOp<T>());
    } else if (1 == dim0) {
        /**head reduce*/
        callHeadReduceForward<T, LogSoftmaxSumLogOp<T>>(y, sumlogptr, dim1, dim2, LogSoftmaxSumLogOp<T>());
    } else {
        /**middle reduce*/
        int N = dim0 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
        MiddleReduceForward<T, LogSoftmaxSumLogOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, sumlogptr, dim0, dim1, dim2, LogSoftmaxSumLogOp<T>(), N);
    }

    int N = dim0 * dim1 * dim2;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    LogSoftmaxForwardKernel<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(x, maxptr, sumlogptr, y, dim0, dim1, dim2, N);

    device->free(maxptr);
    device->free(sumlogptr);
}

template <typename T>
void LogSoftmax<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of LogSoftmax backwardCPU is error");

    auto device = (GPUDevice*)iGradient->device();

    auto x  = inputs[0]->data();
    auto dx = iGradient->data();
    auto y  = output->data();
    auto dy = outputGradient->data();

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

    /**store the temp data*/
    auto sumptr = (T*)device->malloc(sizeof(T) * dim0 * dim2);

    /**cal sum*/
    if (1 == dim2) {
        callTailReduceForward<T, LogSoftmaxSumOp<T>>(dy, sumptr, dim0, dim1, LogSoftmaxSumOp<T>());
    } else if (1 == dim0) {
        callHeadReduceForward<T, LogSoftmaxSumOp<T>>(dy, sumptr, dim1, dim2, LogSoftmaxSumOp<T>());
    } else {
        int N = dim0 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
        MiddleReduceForward<T, LogSoftmaxSumOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(dy, sumptr, dim0, dim1, dim2, LogSoftmaxSumOp<T>(), N);
    }

    int N = dim0 * dim1 * dim2;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    LogSoftmaxBackwardKernel<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(dx, y, dy, sumptr, dim0, dim1, dim2, N);

    device->free(sumptr);
}

}










