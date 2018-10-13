#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "Device.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void DivideForwardKernel(const real *x, const int *xshape, const int *xdims,
                                    const real *y, const int *yshape, const int *ydims,
                                    real *z, const int *zshape, const int *zdims, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int zIndex[MAX_TENSOR_DIMS];

    for (int i = start; i < N; i += stride) {
        for (int k = 0, index = i; k < MAX_TENSOR_DIMS; ++k) {
            zIndex[k] = index / zdims[k];
            index %= zdims[k];
        }

        int xIndex = 0;
        int yIndex = 0;

        for (int k = 0; k < MAX_TENSOR_DIMS; ++k) {
            if (xshape[k] == zshape[k]) {
                xIndex += zIndex[k] * xdims[k];
            }

            if (yshape[k] == zshape[k]) {
                yIndex += zIndex[k] * ydims[k];
            }
        }

        z[i] = x[xIndex] / y[yIndex];
    }
}

template <typename real>
__global__ void DivideBackwardKernelX(real *xGrad, const int *xshape, const int *xdims,
                                      const real *y,     const int *yshape, const int *ydims,
                                      const real *zGrad, const int *zshape, const int *zdims, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int xIndex[MAX_TENSOR_DIMS];
    int yIndex[MAX_TENSOR_DIMS];
    int zIndex[MAX_TENSOR_DIMS];

    for (int xI = start; xI < N; xI += stride) {
        for (int k = 0, index = xI; k < MAX_TENSOR_DIMS; ++k) {
            xIndex[k] = index / xdims[k];
            zIndex[k] = xIndex[k];

            index %= xdims[k];
        }

        int j = MAX_TENSOR_DIMS - 1;

        while (j >= 0) {
            if (j == MAX_TENSOR_DIMS - 1) {
                for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
                    if (yshape[l] == zshape[l]) {
                        yIndex[l] = zIndex[l];
                    } else {
                        yIndex[l] = 0;
                    }
                }

                int yI = 0;
                int zI = 0;

                for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
                    yI += yIndex[l] * ydims[l];
                    zI += zIndex[l] * zdims[l];
                }

                xGrad[xI] += zGrad[zI] / y[yI];
            }

            if (xshape[j] == zshape[j]) {
                j--;
            } else {
                zIndex[j]++;

                if (zIndex[j] >= zshape[j]) {
                    j--;
                } else {
                    for (int l = j + 1; l < MAX_TENSOR_DIMS; ++l) {
                        zIndex[l] = xIndex[l];
                    }

                    j = MAX_TENSOR_DIMS - 1;
                }
            }
        }
    }
}

template <typename real>
__global__ void DivideBackwardKernelY(const real *x, const int *xshape, const int *xdims,
                                      const real *y, real *yGrad, const int *yshape, const int *ydims,
                                      const real *zGrad, const int *zshape, const int *zdims, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int xIndex[MAX_TENSOR_DIMS];
    int yIndex[MAX_TENSOR_DIMS];
    int zIndex[MAX_TENSOR_DIMS];

    for (int yI = start; yI < N; yI += stride) {
        for (int k = 0, index = yI; k < MAX_TENSOR_DIMS; ++k) {
            yIndex[k] = index / ydims[k];
            zIndex[k] = yIndex[k];

            index %= ydims[k];
        }

        int j = MAX_TENSOR_DIMS - 1;

        while (j >= 0) {
            if (j == MAX_TENSOR_DIMS - 1) {
                for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
                    if (xshape[l] == zshape[l]) {
                        xIndex[l] = zIndex[l];
                    } else {
                        xIndex[l] = 0;
                    }
                }

                int xI = 0;
                int zI = 0;

                for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
                    xI += xIndex[l] * xdims[l];
                    zI += zIndex[l] * zdims[l];
                }

                yGrad[yI] -= x[xI] * zGrad[zI] / (y[yI] * y[yI]);
            }

            if (yshape[j] == zshape[j]) {
                j--;
            } else {
                zIndex[j]++;

                if (zIndex[j] >= zshape[j]) {
                    j--;
                } else {
                    for (int l = j + 1; l < MAX_TENSOR_DIMS; ++l) {
                        zIndex[l] = yIndex[l];
                    }

                    j = MAX_TENSOR_DIMS - 1;
                }
            }
        }
    }
}

template <typename T>
void Divide<T>::forwardGPUImpl(const T *x, const int *xshape, const int *xdims,
                               const T *y, const int *yshape, const int *ydims,
                                     T *z, const int *zshape, const int *zdims, const int N) {
    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DivideForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    DivideForwardKernel<T> << <grideSize, blockSize >> > (x, xshape, xdims, y, yshape, ydims, z, zshape, zdims, N);
}

#ifdef HAVE_HALF
template <>
void Divide<half>::forwardGPUImpl(const half *x, const int *xshape, const int *xdims,
                                  const half *y, const int *yshape, const int *ydims,
                                        half *z, const int *zshape, const int *zdims, const int N) {
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    DivideForwardKernel<half> << <grideSize, blockSize >> > (x, xshape, xdims, y, yshape, ydims, z, zshape, zdims, N);
}
#endif

template <typename T>
void Divide<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
auto device = static_cast<GPUDevice*>(output->device());

    auto x = inputs[0];
    auto y = inputs[1];
    auto z = output;

    int xshape[MAX_TENSOR_DIMS];
    int yshape[MAX_TENSOR_DIMS];
    int zshape[MAX_TENSOR_DIMS];

    int xdims[MAX_TENSOR_DIMS];
    int ydims[MAX_TENSOR_DIMS];
    int zdims[MAX_TENSOR_DIMS];

    enlongateShapeToMaxDim(x->shape, xshape);
    enlongateShapeToMaxDim(y->shape, yshape);
    enlongateShapeToMaxDim(z->shape, zshape);

    xdims[MAX_TENSOR_DIMS - 1] = 1;
    ydims[MAX_TENSOR_DIMS - 1] = 1;
    zdims[MAX_TENSOR_DIMS - 1] = 1;

    for (int i = MAX_TENSOR_DIMS - 2; i >= 0; --i) {
        xdims[i] = xdims[i + 1] * xshape[i + 1];
        ydims[i] = ydims[i + 1] * yshape[i + 1];
        zdims[i] = zdims[i + 1] * zshape[i + 1];
    }

    auto cudaPtr = (int*)device->malloc(sizeof(int) * MAX_TENSOR_DIMS * 6);

    int *xshapePtr = cudaPtr;
    int *yshapePtr = xshapePtr + MAX_TENSOR_DIMS;
    int *zshapePtr = yshapePtr + MAX_TENSOR_DIMS;
    int *xdimsPtr  = zshapePtr + MAX_TENSOR_DIMS;
    int *ydimsPtr  = xdimsPtr  + MAX_TENSOR_DIMS;
    int *zdimsPtr  = ydimsPtr  + MAX_TENSOR_DIMS;

    device->copyFromCPUToGPU(xshape, xshapePtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(yshape, yshapePtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(zshape, zshapePtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(xdims, xdimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(ydims, ydimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(zdims, zdimsPtr, sizeof(int) * MAX_TENSOR_DIMS);

    forwardGPUImpl(x->data(), xshapePtr, xdimsPtr, y->data(), yshapePtr, ydimsPtr, z->data(), zshapePtr, zdimsPtr, static_cast<int>(z->shape.size()));

    device->free(cudaPtr);
}

template <typename T>
void Divide<T>::backwardGPUImplX(T *xGrad,  const int *xshape, const int *xdims,
                           const T *y,      const int *yshape, const int *ydims,
                           const T *zGrad,  const int *zshape, const int *zdims, const int N) {
    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DivideBackwardKernelX<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    DivideBackwardKernelX<T> << <grideSize, blockSize >> > (xGrad, xshape, xdims, y, yshape, ydims, zGrad, zshape, zdims, N);
}

#ifdef HAVE_HALF
template <>
void Divide<half>::backwardGPUImplX(half *xGrad, const int *xshape, const int *xdims,
                      const half *y, const int *yshape, const int *ydims,
                      const half *zGrad, const int *zshape, const int *zdims, const int N) {
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    DivideBackwardKernelX<half> << <grideSize, blockSize >> > (xGrad, xshape, xdims, y, yshape, ydims, zGrad, zshape, zdims, N);
}
#endif

template <typename T>
void Divide<T>::backwardGPUImplY(const T *x, const int *xshape, const int *xdims,
                      const T *y, T *yGrad, const int *yshape, const int *ydims,
                      const T *zGrad, const int *zshape, const int *zdims, const int N) {
    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DivideBackwardKernelY<real>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    DivideBackwardKernelY<real> << <grideSize, blockSize >> > (x, xshape, xdims, y, yGrad, yshape, ydims, zGrad, zshape, zdims, N);
}

#ifdef HAVE_HALF
template <>
void Divide<T>::backwardGPUImplY(const half *x, const int *xshape, const int *xdims,
                            const half *y, half *yGrad, const int *yshape, const int *ydims,
                            const half *zGrad, const int *zshape, const int *zdims, const int N) {
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    DivideBackwardKernelY<half> << <grideSize, blockSize >> > (x, xshape, xdims, y, yGrad, yshape, ydims, zGrad, zshape, zdims, N);
}
#endif

template <typename T>
void Divide<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                         const Tensor<T> *output,
                         const Tensor<T> *outputGradient,
                         size_t index,
                         Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    int xShape[MAX_TENSOR_DIMS];
    int yShape[MAX_TENSOR_DIMS];
    int zShape[MAX_TENSOR_DIMS];

    int xDims[MAX_TENSOR_DIMS];
    int yDims[MAX_TENSOR_DIMS];
    int zDims[MAX_TENSOR_DIMS];

    enlongateShapeToMaxDim(inputs[0]->shape, xShape);
    enlongateShapeToMaxDim(inputs[1]->shape, yShape);
    enlongateShapeToMaxDim(outputGradient->shape, zShape);

    xDims[MAX_TENSOR_DIMS - 1] = 1;
    yDims[MAX_TENSOR_DIMS - 1] = 1;
    zDims[MAX_TENSOR_DIMS - 1] = 1;

    for (int i = MAX_TENSOR_DIMS - 2; i >= 0; --i) {
        xDims[i] = xDims[i + 1] * xShape[i + 1];
        yDims[i] = yDims[i + 1] * yShape[i + 1];
        zDims[i] = zDims[i + 1] * zShape[i + 1];
    }

    auto cudaPtr = (int*)device->malloc(sizeof(int) * MAX_TENSOR_DIMS * 6);

    int *xShapePtr = cudaPtr;
    int *yShapePtr = xShapePtr + MAX_TENSOR_DIMS;
    int *zShapePtr = yShapePtr + MAX_TENSOR_DIMS;

    int *xDimsPtr = zShapePtr + MAX_TENSOR_DIMS;
    int *yDimsPtr = xDimsPtr  + MAX_TENSOR_DIMS;
    int *zDimsPtr = yDimsPtr  + MAX_TENSOR_DIMS;

    device->copyFromCPUToGPU(xShape, xShapePtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(yShape, yShapePtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(zShape, zShapePtr, sizeof(int) * MAX_TENSOR_DIMS);

    device->copyFromCPUToGPU(xDims, xDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(yDims, yDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
    device->copyFromCPUToGPU(zDims, zDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);

    if (0 == index) {
        backwardGPUImplX(iGradient->data(), xShapePtr, xDimsPtr,
            inputs[1]->data(), yShapePtr, yDimsPtr,
            outputGradient->data(), zShapePtr, zDimsPtr, iGradient->size());
    } else {
        backwardGPUImplY(inputs[0]->data(), xShapePtr, xDimsPtr,
                         inputs[1]->data(), iGradient->data(), yShapePtr, yDimsPtr,
                         outputGradient->data(), zShapePtr, zDimsPtr, iGradient->size());
    }

    device->free(cudaPtr);
}

DEEP8_DECLARATION_GPU_FUNC(Divide);

#endif

}