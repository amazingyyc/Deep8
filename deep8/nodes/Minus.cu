#include "GPUBasic.h"
#include "GPUDevice.h"
#include "Minus.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void MinusForwardKernel(const real *x, const int *xshape, const int *xdims,
	const real *y, const int *yshape, const int *ydims,
	real *z, const int *zshape, const int *zdims, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
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

		z[i] = x[xIndex] - y[yIndex];
	}
}

template <typename real>
__global__ void MinusBackwardKernel0(real *inGrad, const int *inShape, const int *inDims,
	const real *outGrad, const int *outShape, const int *outDims, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int inIndex[MAX_TENSOR_DIMS];
	int outIndex[MAX_TENSOR_DIMS];

	for (int inI = start; inI < N; inI += stride) {
		for (int k = 0, index = inI; k < MAX_TENSOR_DIMS; ++k) {
			inIndex[k] = index / inDims[k];
			outIndex[k] = inIndex[k];

			index %= inDims[k];
		}

		int j = MAX_TENSOR_DIMS - 1;

		while (j >= 0) {
			if (j == MAX_TENSOR_DIMS - 1) {
				int zI = 0;

				for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
					zI += outIndex[l] * outDims[l];
				}

				inGrad[inI] += outGrad[zI];
			}

			if (inShape[j] == outShape[j]) {
				j--;
			} else {
				outIndex[j]++;

				if (outIndex[j] >= outShape[j]) {
					j--;
				} else {
					for (int l = j + 1; l < MAX_TENSOR_DIMS; ++l) {
						outIndex[l] = inIndex[l];
					}

					j = MAX_TENSOR_DIMS - 1;
				}
			}
		}
	}
}

template <typename real>
__global__ void MinusBackwardKernel1(real *inGrad, const int *inShape, const int *inDims,
	const real *outGrad, const int *outShape, const int *outDims, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int inIndex[MAX_TENSOR_DIMS];
	int outIndex[MAX_TENSOR_DIMS];

	for (int inI = start; inI < N; inI += stride) {
		for (int k = 0, index = inI; k < MAX_TENSOR_DIMS; ++k) {
			inIndex[k] = index / inDims[k];
			outIndex[k] = inIndex[k];

			index %= inDims[k];
		}

		int j = MAX_TENSOR_DIMS - 1;

		while (j >= 0) {
			if (j == MAX_TENSOR_DIMS - 1) {
				int zI = 0;

				for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
					zI += outIndex[l] * outDims[l];
				}

				inGrad[inI] -= outGrad[zI];
			}

			if (inShape[j] == outShape[j]) {
				j--;
			} else {
				outIndex[j]++;

				if (outIndex[j] >= outShape[j]) {
					j--;
				} else {
					for (int l = j + 1; l < MAX_TENSOR_DIMS; ++l) {
						outIndex[l] = inIndex[l];
					}

					j = MAX_TENSOR_DIMS - 1;
				}
			}
		}
	}
}

template <typename T>
void Minus<T>::forwardGPUImpl(const T *x, const int *xshape, const int *xdims,
							  const T *y, const int *yshape, const int *ydims,
	                                T *z, const int *zshape, const int *zdims, const int N) {
	int minGrideSize;
	int blockSize;
	int grideSize;

	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MinusForwardKernel<T>, 0, N));

	grideSize = (N + blockSize - 1) / blockSize;

	MinusForwardKernel<T> << <grideSize, blockSize >> > (x, xshape, xdims, y, yshape, ydims, z, zshape, zdims, N);
}

#ifdef HAVE_HALF
template <>
void Minus<half>::forwardGPUImpl(const half *x, const int *xshape, const int *xdims,
								 const half *y, const int *yshape, const int *ydims,
									   half *z, const int *zshape, const int *zdims, const int N) {
	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	MinusForwardKernel<half> << <grideSize, blockSize >> > (x, xshape, xdims, y, yshape, ydims, z, zshape, zdims, N);
}
#endif

template <typename T>
void Minus<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
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
	int *xdimsPtr = zshapePtr + MAX_TENSOR_DIMS;
	int *ydimsPtr = xdimsPtr + MAX_TENSOR_DIMS;
	int *zdimsPtr = ydimsPtr + MAX_TENSOR_DIMS;

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
void Minus<T>::backwardGPUImpl0(T *inGrad, const int *inShape, const int *inDims, const T *outGrad, const int *outShape, const int *outDims, const int N) {
	int minGrideSize;
	int blockSize;
	int grideSize;

	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MinusBackwardKernel0<T>, 0, N));

	grideSize = (N + blockSize - 1) / blockSize;

	MinusBackwardKernel0<T> << <grideSize, blockSize >> > (inGrad, inShape, inDims, outGrad, outShape, outDims, N);
}

#ifdef HAVE_HALF
template <>
void Minus<half>::backwardGPUImpl0(half *inGrad, const int *inShape, const int *inDims, const half *outGrad, const int *outShape, const int *outDims, const int N) {
	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	MinusBackwardKernel0<half> << <grideSize, blockSize >> > (inGrad, inShape, inDims, outGrad, outShape, outDims, N);
}
#endif

template <typename T>
void Minus<T>::backwardGPUImpl1(T *inGrad, const int *inShape, const int *inDims, const T *outGrad, const int *outShape, const int *outDims, const int N) {
	int minGrideSize;
	int blockSize;
	int grideSize;

	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MinusBackwardKernel1<T>, 0, N));

	grideSize = (N + blockSize - 1) / blockSize;

	MinusBackwardKernel1<T> << <grideSize, blockSize >> > (inGrad, inShape, inDims, outGrad, outShape, outDims, N);
}

#ifdef HAVE_HALF
template <>
void Minus<half>::backwardGPUImpl1(half *inGrad, const int *inShape, const int *inDims, const half *outGrad, const int *outShape, const int *outDims, const int N) {
	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	MinusBackwardKernel1<half> << <grideSize, blockSize >> > (inGrad, inShape, inDims, outGrad, outShape, outDims, N);
}
#endif

template <typename T>
void Minus<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

	auto device = static_cast<GPUDevice*>(iGradient->device());

	int inShape[MAX_TENSOR_DIMS];
	int outShape[MAX_TENSOR_DIMS];

	int inDims[MAX_TENSOR_DIMS];
	int outDims[MAX_TENSOR_DIMS];

	enlongateShapeToMaxDim(iGradient->shape, inShape);
	enlongateShapeToMaxDim(outputGradient->shape, outShape);

	inDims[MAX_TENSOR_DIMS - 1] = 1;
	outDims[MAX_TENSOR_DIMS - 1] = 1;

	for (int i = MAX_TENSOR_DIMS - 2; i >= 0; --i) {
		inDims[i] = inDims[i + 1] * inShape[i + 1];
		outDims[i] = outDims[i + 1] * outShape[i + 1];
	}

	auto cudaPtr = (int*)device->malloc(sizeof(int) * MAX_TENSOR_DIMS * 4);

	int *inShapePtr  = cudaPtr;
	int *outShapePtr = inShapePtr + MAX_TENSOR_DIMS;
	int *inDimsPtr   = outShapePtr + MAX_TENSOR_DIMS;
	int *outDimsPtr  = inDimsPtr + MAX_TENSOR_DIMS;

	device->copyFromCPUToGPU(inShape, inShapePtr, sizeof(int) * MAX_TENSOR_DIMS);
	device->copyFromCPUToGPU(outShape, outShapePtr, sizeof(int) * MAX_TENSOR_DIMS);
	device->copyFromCPUToGPU(inDims, inDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
	device->copyFromCPUToGPU(outDims, outDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);

	if (0 == index) {
		backwardGPUImpl0(iGradient->data(), inShapePtr, inDimsPtr, outputGradient->data(), outShapePtr, outDimsPtr, static_cast<int>(iGradient->size()));
	} else if (1 == index) {
		backwardGPUImpl1(iGradient->data(), inShapePtr, inDimsPtr, outputGradient->data(), outShapePtr, outDimsPtr, static_cast<int>(iGradient->size()));
	}

	device->free(cudaPtr);
}

DEEP8_DECLARATION_GPU_FUNC(Minus)

#endif
}