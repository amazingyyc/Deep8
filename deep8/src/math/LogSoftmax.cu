#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/LogSoftmax.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct LogSoftmaxMaxKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
        return cudaMinValue<T>();
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

template <typename T>
struct LogSoftmaxExpMinusKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y) {
        return cudaExp(x - y);
    }
};

template <typename T>
struct LogSoftmaxSumLogKernelOp {
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
        return cudaLog(ret);
    }
};

template <typename T>
__global__ void LogSoftmaxKernel(const T *x, const T *maxptr, const T *sumlogptr, T *y, const int dim0, const int dim1, const int dim2, const int N) {
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
void LogSoftmaxGPUImpl(GPUDevice *device, const T *x, const Shape &xshape, T *y, const Shape &yshape, T *maxptr, T *sumlogptr) {
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) xshape.batch;
        dim1 = (int) xshape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) xshape.batch;
        dim1 = (int) xshape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) xshape.dim(i);
        }

        for (int i = axis + 1; i < xshape.nDims; ++i) {
            dim2 *= (int) xshape.dim(i);
        }
    }

    if (1 == dim2) {
        CallTailReduceKernel<T, LogSoftmaxMaxKernelOp<T>>(x, maxptr, dim0, dim1, LogSoftmaxMaxKernelOp<T>());
    } else if (1 == dim0) {
        CallHeadReduceKernel<T, LogSoftmaxMaxKernelOp<T>>(x, maxptr, dim1, dim2, LogSoftmaxMaxKernelOp<T>());
    } else {
        int N = dim0 * dim2;
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        MiddleReduceKernel<T, LogSoftmaxMaxKernelOp<T>> <<<grideSize, blockSize>>>(x, maxptr, dim0, dim1, dim2, LogSoftmaxMaxKernelOp<T>(), N);
    }

    {
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
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, LogSoftmaxExpMinusKernelOp<T>, 3> <<<grideSize, blockSize>>>(x, xNVShape, maxptr, maxNVShape, y, yNVShape, LogSoftmaxExpMinusKernelOp<T>(), N);
    }

    /**calculate sum*/
    if (1 == dim2) {
        /**tail reduce*/
        CallTailReduceKernel<T, LogSoftmaxSumLogKernelOp<T>>(y, sumlogptr, dim0, dim1, LogSoftmaxSumLogKernelOp<T>());
    } else if (1 == dim0) {
        /**head reduce*/
        CallHeadReduceKernel<T, LogSoftmaxSumLogKernelOp<T>>(y, sumlogptr, dim1, dim2, LogSoftmaxSumLogKernelOp<T>());
    } else {
        /**middle reduce*/
        int N = dim0 * dim2;
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        MiddleReduceKernel<T, LogSoftmaxSumLogKernelOp<T>> <<<grideSize, blockSize>>>(y, sumlogptr, dim0, dim1, dim2, LogSoftmaxSumLogKernelOp<T>(), N);
    }

    {
        int N = dim0 * dim1 * dim2;
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        LogSoftmaxKernel<T><<<grideSize, blockSize >>>(x, maxptr, sumlogptr, y, dim0, dim1, dim2, N);
    }
}

void LogSoftmaxGPU(const Tensor &x, Tensor &y, int axis, void *maxptr, void *sumptr) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        LogSoftmaxGPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, axis, (float*)maxptr, (float*)sumptr);
        break;
    case DType::Float64:
        LogSoftmaxGPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, axis, (double*)maxptr, (double*)sumptr);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        LogSoftmaxGPUImpl<half>(device, x.data<half>(), x.shape, y.data<half>(), y.shape, axis, (half*)maxptr, (half*)sumptr);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


template <typename T>
struct LogSoftmaxSumKernelOp {
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

template <typename T>
__global__ void LogSoftmaxGradKernel(T *dx, const T *y, const T *dy, const T *sumptr, const int dim0, const int dim1, const int dim2, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int d0 = i / (dim1 * dim2);
        int d2 = i % dim2;

        int j = d0 * dim2 + d2;

        dx[i] += dy[i] - cudaExp(y[i]) * sumptr[j];
    }
}

template <typename T>
void LogSoftmaxGradGPUImpl(GPUDevice *device, const T *x, T *dx, const Shape &xshape, const T *y, const T *dy, const Shape &yshape, int axis, T *sumptr) {
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) xshape.batch;
        dim1 = (int) xshape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) xshape.batch;
        dim1 = (int) xshape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) xshape.dim(i);
        }

        for (int i = axis + 1; i < xshape.nDims; ++i) {
            dim2 *= (int) xshape.dim(i);
        }
    }

    /**cal sum*/
    if (1 == dim2) {
        CallTailReduceKernel<T, LogSoftmaxSumKernelOp<T>>(dy, sumptr, dim0, dim1, LogSoftmaxSumKernelOp<T>());
    } else if (1 == dim0) {
        CallHeadReduceKernel<T, LogSoftmaxSumKernelOp<T>>(dy, sumptr, dim1, dim2, LogSoftmaxSumKernelOp<T>());
    } else {
        int N = dim0 * dim2;
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        MiddleReduceKernel<T, LogSoftmaxSumKernelOp<T>> <<<grideSize, blockSize>>>(dy, sumptr, dim0, dim1, dim2, LogSoftmaxSumKernelOp<T>(), N);
    }

    int N = dim0 * dim1 * dim2;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    LogSoftmaxGradKernel<T><<<grideSize, blockSize >>>(dx, y, dy, sumptr, dim0, dim1, dim2, N);
}

void LogSoftmaxGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis, void *sumptr) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        LogSoftmaxGradGPUImpl<float>(device,
                                 x.data<float>(),
                                 dx.data<float>(),
                                 x.shape,
                                 y.data<float>(),
                                 dy.data<float>(),
                                 y.shape,
                                 axis,
                                 (float*)sumptr);
        break;
    case DType::Float64:
        LogSoftmaxGradGPUImpl<double>(device,
                                 x.data<double>(),
                                 dx.data<double>(),
                                 x.shape,
                                 y.data<double>(),
                                 dy.data<double>(),
                                 y.shape,
                                 axis,
                                 (double*)sumptr);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        LogSoftmaxGradGPUImpl<half>(device,
                                 x.data<half>(),
                                 dx.data<half>(),
                                 x.shape,
                                 y.data<half>(),
                                 dy.data<half>(),
                                 y.shape,
                                 axis,
                                 (half*)sumptr);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


}
}