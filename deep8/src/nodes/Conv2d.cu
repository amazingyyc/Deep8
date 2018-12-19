#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "Conv2d.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void Conv2dIm2ColKernel(const real *im, real *col,
                                   const int batch, const int inputHeight, const int inputWidth, const int inputChannel,
                                   const int filterHeight, const int filterWidth, const int padTop, const int padLeft,
                                   const int strideY, const int strideX, const int dilationY, const int dilationX,
                                   const int outputHeight, const int outputWidth, const int N) {

    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int colY = i / (filterHeight * filterWidth);
        int colX = i % (filterHeight * filterWidth);

        int b = colY / (outputHeight * outputWidth);

        int outputIndex = colY % (outputHeight * outputWidth);

        int outputY = outputIndex / outputWidth;
        int outputX = outputIndex % outputWidth;

        int filterY = colX / filterWidth;
        int filterX = colX % filterWidth;

        int inputY = padTop  + outputY * strideY + filterY * dilationY;
        int inputX = padLeft + outputX * strideX + filterX * dilationX;

        real *colPtr = col + colY * filterHeight * filterWidth * inputChannel + colX * inputChannel;

        if (0 > inputY || inputY >= inputHeight || 0 > inputX || inputX >= inputWidth) {
            for (int k = 0; k < inputChannel; ++k) {
                colPtr[k] = 0;
            }
        } else {
            const real *imPtr = im + b * inputHeight * inputWidth * inputChannel + inputY * inputWidth * inputChannel + inputX * inputChannel;

            for (int k = 0; k < inputChannel; ++k) {
                colPtr[k] = imPtr[k];
            }
        }
    }
}

template <typename real>
__global__ void Conv2dCol2ImKernel(const real *col, real *im,
                                   const int batch, const int inputHeight, const int inputWidth, const int inputChannel,
                                   const int filterHeight, const int filterWidth, const int padTop, const int padLeft,
                                   const int strideY, const int strideX, const int dilationY, const int dilationX,
                                   const int outputHeight, const int outputWidth, const int N) {

    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int colWidth = filterHeight * filterWidth * inputChannel;

    for (int i = start; i < N; i += stride) {
        int b = i / (inputHeight * inputWidth);

        int inputIndex = i % (inputHeight * inputWidth);

        int inputY = inputIndex / inputWidth;
        int inputX = inputIndex % inputWidth;

        real *imPtr = im + b * inputHeight * inputWidth * inputChannel + inputY * inputWidth * inputChannel + inputX * inputChannel;

        for (int filterY = 0; filterY < filterHeight; ++filterY) {
            for (int filterX = 0; filterX < filterWidth; ++filterX) {
                int outputY = inputY - padTop - filterY * dilationY;
                int outputX = inputX - padLeft - filterX * dilationX;

                if (0 == (outputY % strideY) && 0 == (outputX % strideX)) {
                    outputY /= strideY;
                    outputX /= strideX;

                    if (0 <= outputY && outputY < outputHeight && 0 <= outputX && outputX < outputWidth) {
                        const real *colPtr = col + (b * outputHeight * outputWidth + outputY * outputWidth + outputX) * colWidth
                                             + (filterY * filterWidth + filterX) * inputChannel;

                        for (int k = 0; k < inputChannel; ++k) {
                            imPtr[k] += colPtr[k];
                        }
                    }
                }
            }
        }
    }
}


template <>
void Conv2d<float>::forwardGPUImpl(Device* d, const float *x, const float *filter, float *y,
                                int batch, int inputHeight, int inputWidth, int inputChannel,
                                int outputHeight, int outputWidth, int outputChannel,
                                int filterHeight, int filterWidth, int strideY, int strideX,
                                int padTop, int padLeft, int dilationY, int dilationX) {
    auto device = (GPUDevice*)d;
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    auto xCol = (float*)device->malloc(sizeof(float) * size * inputChannel);

    Conv2dIm2ColKernel<float> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, xCol,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth, padTop, padLeft,
            strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

    int m = batch * outputHeight * outputWidth;
    int k = filterHeight * filterWidth * inputChannel;
    int n = outputChannel;

    float alpha = 1;
    float beta  = 0;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, filter, k, xCol, k, &beta, y, n));

    device->free(xCol);
}

template <>
void Conv2d<double>::forwardGPUImpl(Device* d, const double *x, const double *filter, double *y,
                    int batch, int inputHeight, int inputWidth, int inputChannel,
                    int outputHeight, int outputWidth, int outputChannel,
                    int filterHeight, int filterWidth, int strideY, int strideX,
                    int padTop, int padLeft, int dilationY, int dilationX) {
    auto device = (GPUDevice*)d;
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    auto xCol = (double*)device->malloc(sizeof(double) * size * inputChannel);

    Conv2dIm2ColKernel<double> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, xCol,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth, padTop, padLeft,
            strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

    int m = batch * outputHeight * outputWidth;
    int k = filterHeight * filterWidth * inputChannel;
    int n = outputChannel;

    double alpha = 1;
    double beta = 0;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, filter, k, xCol, k, &beta, y, n));

    device->free(xCol);
}

#ifdef HAVE_HALF
template <>
void Conv2d<half>::forwardGPUImpl(Device* d, const half *x, const half *filter, half *y,
                            int batch, int inputHeight, int inputWidth, int inputChannel,
                            int outputHeight, int outputWidth, int outputChannel,
                            int filterHeight, int filterWidth, int strideY, int strideX,
                            int padTop, int padLeft, int dilationY, int dilationX) {
        auto device = (GPUDevice*)d;
		int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

		int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

		auto xCol = (half*)device->malloc(sizeof(half) * size * inputChannel);

		Conv2dIm2ColKernel<half> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, xCol,
			batch, inputHeight, inputWidth, inputChannel,
			filterHeight, filterWidth, padTop, padLeft,
			strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

		int m = batch * outputHeight * outputWidth;
		int k = filterHeight * filterWidth * inputChannel;
		int n = outputChannel;

		half alpha(1.0);
		half beta(0.0);

		CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, filter, k, xCol, k, &beta, y, n));

		device->free(xCol);
	}
#endif

template <typename T>
void Conv2d<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = output->device();

    auto x = inputs[0];
    auto filter = inputs[1];

    auto y = output;

    auto batch        = (int)x->shape.batch;
    auto inputHeight  = (int)x->shape.dim(0);
    auto inputWidth   = (int)x->shape.dim(1);
    auto inputChannel = (int)x->shape.dim(2);

    auto outputHeight  = (int)y->shape.dim(0);
    auto outputWidth   = (int)y->shape.dim(1);
    auto outputChannel = (int)y->shape.dim(2);

    auto filterHeight = (int)filter->shape.dim(1);
    auto filterWidth  = (int)filter->shape.dim(2);

    auto realFilterHeight = filterHeight + (filterHeight - 1) * ((int)dilationY - 1);
    auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * ((int)dilationX - 1);

    auto padTop  = -(std::max<int>(0, (outputHeight - 1) * (int)(strideY) + realFilterHeight - inputHeight) / 2);
    auto padLeft = -(std::max<int>(0, (outputWidth  - 1) * (int)(strideX) + realFilterWidth  - inputWidth)  / 2);

    forwardGPUImpl(device, x->data(), filter->data(), y->data(),
        batch, inputHeight, inputWidth, inputChannel, outputHeight, outputWidth, outputChannel,
        filterHeight, filterWidth, (int)strideY, (int)strideX, padTop, padLeft, (int)dilationY, (int)dilationX);
}

/**for input*/
template <>
void Conv2d<float>::backwardGPUInputImpl(Device* d, float *dx, const float *w, const float *dy,
                          int batch, int inputHeight, int inputWidth, int inputChannel,
                          int outputHeight, int outputWidth, int outputChannel,
                          int filterHeight, int filterWidth, int strideY, int strideX,
                          int padTop, int padLeft, int dilationY, int dilationX) {
    auto device = (GPUDevice*)d;
    int size = batch * inputHeight * inputWidth;

    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    auto xCol = (float*)device->malloc(sizeof(float) * batch * outputHeight * outputWidth * filterHeight * filterWidth * inputChannel);

    int m = filterHeight * filterWidth * inputChannel;
    int k = outputChannel;
    int n = batch * outputHeight * outputWidth;

    float alpha = 1;
    float beta  = 0;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, w, m, dy, k, &beta, xCol, m));

    Conv2dCol2ImKernel<float> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (xCol, dx,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth, padTop, padLeft,
            strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

    device->free(xCol);
}

template <>
void Conv2d<double>::backwardGPUInputImpl(Device* d, double *dx, const double *w, const double *dy,
                          int batch, int inputHeight, int inputWidth, int inputChannel,
                          int outputHeight, int outputWidth, int outputChannel,
                          int filterHeight, int filterWidth, int strideY, int strideX,
                          int padTop, int padLeft, int dilationY, int dilationX) {
    auto device = (GPUDevice*)d;
    int size = batch * inputHeight * inputWidth;

    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    auto xCol = (double*)device->malloc(sizeof(double) * batch * outputHeight * outputWidth * filterHeight * filterWidth * inputChannel);

    int m = filterHeight * filterWidth * inputChannel;
    int k = outputChannel;
    int n = batch * outputHeight * outputWidth;

    double alpha = 1;
    double beta = 0;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, w, m, dy, k, &beta, xCol, m));

    Conv2dCol2ImKernel<double> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (xCol, dx,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth, padTop, padLeft,
            strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

    device->free(xCol);
}

#ifdef HAVE_HALF
template <>
void Conv2d<half>::backwardGPUInputImpl(Device* d, half *dx, const half *w, const half *dy,
                                int batch, int inputHeight, int inputWidth, int inputChannel,
                                int outputHeight, int outputWidth, int outputChannel,
                                int filterHeight, int filterWidth, int strideY, int strideX,
                                int padTop, int padLeft, int dilationY, int dilationX) {
        auto device = (GPUDevice*)d;
		int size = batch * inputHeight * inputWidth;

		int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

		auto xCol = (half*)device->malloc(sizeof(half) * batch * outputHeight * outputWidth * filterHeight * filterWidth * inputChannel);

		int m = filterHeight * filterWidth * inputChannel;
		int k = outputChannel;
		int n = batch * outputHeight * outputWidth;

		half alpha(1.0);
		half beta(0.0);

		CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, w, m, dy, k, &beta, xCol, m));

		Conv2dCol2ImKernel<half> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (xCol, dx,
			batch, inputHeight, inputWidth, inputChannel,
			filterHeight, filterWidth, padTop, padLeft,
			strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

		device->free(xCol);
	}
#endif

/**for filter*/
template <>
void Conv2d<float>::backwardGPUFilterImpl(Device* d, const float *x, float *dw, const float *dy,
                           int batch, int inputHeight, int inputWidth, int inputChannel,
                           int outputHeight, int outputWidth, int outputChannel,
                           int filterHeight, int filterWidth, int strideY, int strideX,
                           int padTop, int padLeft, int dilationY, int dilationX) {
    auto device = (GPUDevice*)d;
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    auto xCol = (float*)device->malloc(sizeof(float) * size * inputChannel);

    Conv2dIm2ColKernel<float> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, xCol,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth, padTop, padLeft,
            strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

    int m = filterHeight * filterWidth * inputChannel;
    int k = batch * outputHeight * outputWidth;
    int n = outputChannel;

    float alpha = 1;
    float beta = 1;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, xCol, m, dy, n, &beta, dw, m));

    device->free(xCol);
}

template <>
void Conv2d<double>::backwardGPUFilterImpl(Device* d, const double *x, double *dw, const double *dy,
                           int batch, int inputHeight, int inputWidth, int inputChannel,
                           int outputHeight, int outputWidth, int outputChannel,
                           int filterHeight, int filterWidth, int strideY, int strideX,
                           int padTop, int padLeft, int dilationY, int dilationX) {
    auto device = (GPUDevice*)d;
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    auto xCol = (double*)device->malloc(sizeof(double) * size * inputChannel);

    Conv2dIm2ColKernel<double> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, xCol,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth, padTop, padLeft,
            strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

    int m = filterHeight * filterWidth * inputChannel;
    int k = batch * outputHeight * outputWidth;
    int n = outputChannel;

    double alpha = 1;
    double beta = 1;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, xCol, m, dy, n, &beta, dw, m));

    device->free(xCol);
}

#ifdef HAVE_HALF
template <>
void Conv2d<half>::backwardGPUFilterImpl(Device* d, const half *x, half *dw, const half *dy,
                        int batch, int inputHeight, int inputWidth, int inputChannel,
                        int outputHeight, int outputWidth, int outputChannel,
                        int filterHeight, int filterWidth, int strideY, int strideX,
                        int padTop, int padLeft, int dilationY, int dilationX) {
    auto device = (GPUDevice*)d;
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    auto xCol = (half*)device->malloc(sizeof(half) * size * inputChannel);

    Conv2dIm2ColKernel<half> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, xCol,
        batch, inputHeight, inputWidth, inputChannel,
        filterHeight, filterWidth, padTop, padLeft,
        strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

    int m = filterHeight * filterWidth * inputChannel;
    int k = batch * outputHeight * outputWidth;
    int n = outputChannel;

    half alpha(1.0);
    half beta(1.0);

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, xCol, m, dy, n, &beta, dw, m));

    device->free(xCol);
}
#endif

template <typename T>
void Conv2d<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient)  {
    DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    auto xShape = inputs[0]->shape;
    auto wShape = inputs[1]->shape;
    auto yShape = output->shape;

    auto batch        = (int)xShape.batch;
    auto inputHeight  = (int)xShape.dim(0);
    auto inputWidth   = (int)xShape.dim(1);
    auto inputChannel = (int)xShape.dim(2);

    auto outputHeight  = (int)yShape.dim(0);
    auto outputWidth   = (int)yShape.dim(1);
    auto outputChannel = (int)yShape.dim(2);

    auto filterHeight = (int)wShape.dim(1);
    auto filterWidth  = (int)wShape.dim(2);

    auto realFilterHeight = filterHeight + (filterHeight - 1) * ((int)dilationY - 1);
    auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * ((int)dilationX - 1);

    auto padTop  = -(std::max<int>(0, (outputHeight - 1) * (int)(strideY) +realFilterHeight - inputHeight) / 2);
    auto padLeft = -(std::max<int>(0, (outputWidth  - 1) * (int)(strideX) +realFilterWidth  - inputWidth)  / 2);

    if (0 == index) {
            backwardGPUInputImpl(device, iGradient->data(), inputs[1]->data(), outputGradient->data(),
            batch, inputHeight, inputWidth, inputChannel, outputHeight, outputWidth, outputChannel,
            filterHeight, filterWidth, strideY, strideX, padTop, padLeft, dilationY, dilationX);
    } else if (1 == index) {
        backwardGPUFilterImpl(device, inputs[0]->data(), iGradient->data(),
                              outputGradient->data(),
                              batch, inputHeight, inputWidth, inputChannel, outputHeight,
                              outputWidth, outputChannel,
                              filterHeight, filterWidth, strideY, strideX, padTop, padLeft,
                              dilationY, dilationX);
    }
}

DEEP8_DECLARATION_GPU_FUNC(Conv2d);

template void Conv2d<float>::forwardGPUImpl(Device* device, const float *x, const float *filter, float *y,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

template void Conv2d<double>::forwardGPUImpl(Device* device, const double *x, const double *filter, double *y,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

#ifdef HAVE_HALF
template void Conv2d<half>::forwardGPUImpl(Device* device, const half *x, const half *filter, half *y,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);
#endif

template void Conv2d<float>::backwardGPUInputImpl(Device* device, float *dx, const float *w, const float *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

template void Conv2d<double>::backwardGPUInputImpl(Device* device, double *dx, const double *w, const double *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

#ifdef HAVE_HALF
template void Conv2d<half>::backwardGPUInputImpl(Device* device, half *dx, const half *w, const half *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);
#endif

/**for filter*/
template void Conv2d<float>::backwardGPUFilterImpl(Device* device, const float *x, float *dw, const float *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

template void Conv2d<double>::backwardGPUFilterImpl(Device* device, const double *x, double *dw, const double *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

#ifdef HAVE_HALF
template void Conv2d<half>::backwardGPUFilterImpl(Device* device, const half *x, half *dw, const half *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);
#endif

#endif

}