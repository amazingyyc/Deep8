#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "DeConv2d.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void DeConv2dForwardKernel(const real *outputMat, real *output,
    const int batch, const int inputHeight, const int inputWidth, const int inputChannel,
	const int filterHeight, const int filterWidth,
    const int outputHeight, const int outputWidth, const int outputChannel,
    const int forwardStrideY, const int forwardStrideX,
    const int padTop, const int padLeft,
	 const int N) {

    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int b = i / (outputHeight * outputWidth * outputChannel);

        int outputY = (i % (outputHeight * outputWidth * outputChannel)) / (outputWidth * outputChannel);
        int outputX = (i % (outputWidth * outputChannel)) / outputChannel;
        int outputOffset = i % outputChannel;

        real out = 0;

        for (int y = 0; y < filterHeight; ++y) {
            for (int x = 0; x < filterWidth; ++x) {
                int inputY = outputY + padTop + y;
                int inputX = outputX + padLeft + x;

                if (0 == inputY % forwardStrideY && 0 == inputX % forwardStrideX) {
                    inputY /= forwardStrideY;
                    inputX /= forwardStrideX;

                    if (0 <= inputY && inputY < inputHeight && 0 <= inputX && inputX < inputWidth) {
                        out += outputMat[(b * inputHeight * inputWidth + inputY * inputWidth + inputX) * (outputChannel * filterHeight * filterWidth)
                        + outputOffset * filterHeight * filterWidth + y * filterWidth + x];
                    }
                }
            }
        }

        output[i] = out;
    }
}

template <typename real>
__global__ void DeConv2dBackwardKernel(real *dyMat, const real *dy,
    const int batch, const int inputHeight, const int inputWidth, const int inputChannel,
	const int filterHeight, const int filterWidth,
    const int outputHeight, const int outputWidth, const int outputChannel,
    const int forwardStrideY, const int forwardStrideX,
    const int padTop, const int padLeft, const int N) {

    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int row = i / (outputChannel * filterHeight * filterWidth);
        int col = i % (outputChannel * filterHeight * filterWidth);

        int b = row / (inputHeight * inputWidth);
        int inputY = (row % (inputHeight * inputWidth)) / inputWidth;
        int inputX = row % inputWidth;

        int outputOffset = col / (filterHeight * filterWidth);
        int filterY = (col % (filterHeight * filterWidth)) / filterWidth;
        int filterX = col % filterWidth;

        int outputY = inputY * forwardStrideY - padTop - filterY;
        int outputX = inputX * forwardStrideX - padLeft - filterX;

        if (0 <= outputY && outputY < outputHeight && 0 <= outputX && outputX < outputWidth) {
            dyMat[i] = dy[((b * outputHeight + outputY) * outputWidth + outputX) * outputChannel + outputOffset];
        } else {
            dyMat[i] = 0;
        }
    }
}

template <>
void DeConv2d<float>::forwardGPUImpl(Device *d, const float *x, const float *filter, float *y,
                    int batch, int inputHeight, int inputWidth, int inputChannel,
                    int outputHeight, int outputWidth, int outputChannel,
                    int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
                    int padTop, int padLeft) {
    auto device = (GPUDevice*)d;

    int m = batch * inputHeight * inputWidth;
    int k = inputChannel;
    int n = outputChannel * filterHeight * filterWidth;

    float alpha = 1;
    float beta  = 0;

    auto yMat = (float*)device->malloc(sizeof(float) * m * n);

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, filter, k, x, k, &beta, yMat, n));

    int size = batch * outputHeight * outputWidth * outputChannel;

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DeConv2dForwardKernel<float>, 0, size));

    grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dForwardKernel<float><<<grideSize, blockSize>>>(yMat, y,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth,
            outputHeight, outputWidth, outputChannel,
            forwardStrideY, forwardStrideX,
            padTop, padLeft, size);

    device->free(yMat);
}

template <>
void DeConv2d<double>::forwardGPUImpl(Device *d, const double *x, const double *filter, double *y,
                    int batch, int inputHeight, int inputWidth, int inputChannel,
                    int outputHeight, int outputWidth, int outputChannel,
                    int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
                    int padTop, int padLeft) {
    auto device = (GPUDevice*)d;

    int m = batch * inputHeight * inputWidth;
    int k = inputChannel;
    int n = outputChannel * filterHeight * filterWidth;

    double alpha = 1;
    double beta = 0;

    auto yMat = (double*)device->malloc(sizeof(double) * m * n);

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, filter, k, x, k, &beta, yMat, n));

    int size = batch * outputHeight * outputWidth * outputChannel;

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DeConv2dForwardKernel<double>, 0, size));

    grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dForwardKernel<double> << <grideSize, blockSize >> > (yMat, y,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth,
            outputHeight, outputWidth, outputChannel,
            forwardStrideY, forwardStrideX,
            padTop, padLeft, size);

    device->free(yMat);
}

#ifdef HAVE_HALF

template <>
void DeConv2d<half>::forwardGPUImpl(Device *d, const half *x, const half *filter, half *y,
                            int batch, int inputHeight, int inputWidth, int inputChannel,
                            int outputHeight, int outputWidth, int outputChannel,
                            int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
                            int padTop, int padLeft) {
    auto device = (GPUDevice*)d;
    int m = batch * inputHeight * inputWidth;
    int k = inputChannel;
    int n = outputChannel * filterHeight * filterWidth;

    half alpha = 1;
    half beta = 0;

    auto yMat = (half*)device->malloc(sizeof(half) * m * n);

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, filter, k, x, k, &beta, yMat, n));

    int size = batch * outputHeight * outputWidth * outputChannel;

    int blockSize = 1024;
    int grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dForwardKernel<half> << <grideSize, blockSize >> > (yMat, y,
        batch, inputHeight, inputWidth, inputChannel,
        filterHeight, filterWidth,
        outputHeight, outputWidth, outputChannel,
        forwardStrideY, forwardStrideX,
        padTop, padLeft, size);

    device->free(yMat);
}
#endif

template <typename T>
void DeConv2d<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = static_cast<GPUDevice*>(output->device());

    auto x = inputs[0];
    auto filter = inputs[1];

    auto y = output;

    auto batch        = (int)x->shape.dim(0);
    auto inputHeight  = (int)x->shape.dim(1);
    auto inputWidth   = (int)x->shape.dim(2);
    auto inputChannel = (int)x->shape.dim(3);

    auto outputHeight  = (int)y->shape.dim(1);
    auto outputWidth   = (int)y->shape.dim(2);
    auto outputChannel = (int)y->shape.dim(3);

    auto filterHeight = (int)filter->shape.dim(1);
    auto filterWidth  = (int)filter->shape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * (int)(forwardStrideY) - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * (int)(forwardStrideX) - 2) / 2);

    forwardGPUImpl(device, x->data(), filter->data(), y->data(),
                    batch, inputHeight, inputWidth, inputChannel,
                    outputHeight, outputWidth, outputChannel,
                    filterHeight, filterWidth, (int)forwardStrideY, (int)forwardStrideX,
                    padTop, padLeft);
}

template <>
void DeConv2d<float>::backwardGPUInputImpl(Device *d, float *dx, const float *filter, const float *dy,
                          int batch, int inputHeight, int inputWidth, int inputChannel,
                          int outputHeight, int outputWidth, int outputChannel,
                          int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
                          int padTop, int padLeft) {
    auto device = (GPUDevice*)d;
    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    auto dyMat = (float*)device->malloc(sizeof(float) * size);

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DeConv2dBackwardKernel<float>, 0, size));

    grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dBackwardKernel<float><<<grideSize, blockSize>>>(dyMat, dy,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth,
            outputHeight, outputWidth, outputChannel,
            forwardStrideY, forwardStrideX,
            padTop, padLeft, size);


    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    float alpha = 1;
    float beta  = 1;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k,m, n, &alpha, filter, k, dyMat, n, &beta, dx, k));

    device->free(dyMat);
}

template <>
void DeConv2d<double>::backwardGPUInputImpl(Device *d, double *dx, const double *filter, const double *dy,
                          int batch, int inputHeight, int inputWidth, int inputChannel,
                          int outputHeight, int outputWidth, int outputChannel,
                          int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
                          int padTop, int padLeft) {
    auto device = (GPUDevice*)d;
    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    auto dyMat = (double*)device->malloc(sizeof(double) * size);

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DeConv2dBackwardKernel<double>, 0, size));

    grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dBackwardKernel<double> << <grideSize, blockSize >> > (dyMat, dy,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth,
            outputHeight, outputWidth, outputChannel,
            forwardStrideY, forwardStrideX,
            padTop, padLeft, size);


    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    double alpha = 1;
    double beta = 1;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, filter, k, dyMat, n, &beta, dx, k));

    device->free(dyMat);
}

#ifdef HAVE_HALF
template <>
void DeConv2d<half>::backwardGPUInputImpl(Device *d, half *dx, const half *filter, const half *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
		int padTop, int padLeft) {
    auto device = (GPUDevice*)d;
    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    auto dyMat = (half*)device->malloc(sizeof(half) * size);

    int blockSize = 1024;
    int grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dBackwardKernel<half> << <grideSize, blockSize >> > (dyMat, dy,
        batch, inputHeight, inputWidth, inputChannel,
        filterHeight, filterWidth,
        outputHeight, outputWidth, outputChannel,
        forwardStrideY, forwardStrideX,
        padTop, padLeft, size);


    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    half alpha = 1;
    half beta = 1;

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, filter, k, dyMat, n, &beta, dx, k));

    device->free(dyMat);
}
#endif

template <>
void DeConv2d<float>::backwardGPUFilterImpl(Device *d, const float *x, float *dw, const float *dy,
                           int batch, int inputHeight, int inputWidth, int inputChannel,
                           int outputHeight, int outputWidth, int outputChannel,
                           int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
                           int padTop, int padLeft) {
    auto device = (GPUDevice*)d;
    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    auto dyMat = (float*)device->malloc(sizeof(float) * size);

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DeConv2dBackwardKernel<float>, 0, size));

    grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dBackwardKernel<float><<<grideSize, blockSize>>>(dyMat, dy,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth,
            outputHeight, outputWidth, outputChannel,
            forwardStrideY, forwardStrideX,
            padTop, padLeft, size);

    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    float alpha = 1;
    float beta  = 1;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, n, m, &alpha, x, k, dyMat, n, &beta, dw, k));

    device->free(dyMat);
}

template <>
void DeConv2d<double>::backwardGPUFilterImpl(Device *d, const double *x, double *dw, const double *dy,
                           int batch, int inputHeight, int inputWidth, int inputChannel,
                           int outputHeight, int outputWidth, int outputChannel,
                           int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
                           int padTop, int padLeft) {
    auto device = (GPUDevice*)d;
    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    auto dyMat = (double*)device->malloc(sizeof(double) * size);

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DeConv2dBackwardKernel<double>, 0, size));

    grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dBackwardKernel<double> << <grideSize, blockSize >> > (dyMat, dy,
            batch, inputHeight, inputWidth, inputChannel,
            filterHeight, filterWidth,
            outputHeight, outputWidth, outputChannel,
            forwardStrideY, forwardStrideX,
            padTop, padLeft, size);

    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    double alpha = 1;
    double beta = 1;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, n, m, &alpha, x, k, dyMat, n, &beta, dw, k));

    device->free(dyMat);
}

#ifdef HAVE_HALF
template <>
void DeConv2d<half>::backwardGPUFilterImpl(Device *d, const half *x, half *dw, const half *dy,
                                    int batch, int inputHeight, int inputWidth, int inputChannel,
                                    int outputHeight, int outputWidth, int outputChannel,
                                    int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
                                    int padTop, int padLeft) {
    auto device = (GPUDevice*)d;
    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    auto dyMat = (half*)device->malloc(sizeof(half) * size);

    int blockSize = 1024;
    int grideSize = (size + blockSize - 1) / blockSize;

    DeConv2dBackwardKernel<half> << <grideSize, blockSize >> > (dyMat, dy,
        batch, inputHeight, inputWidth, inputChannel,
        filterHeight, filterWidth,
        outputHeight, outputWidth, outputChannel,
        forwardStrideY, forwardStrideX,
        padTop, padLeft, size);

    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    half alpha = 1;
    half beta = 1;

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, n, m, &alpha, x, k, dyMat, n, &beta, dw, k));

    device->free(dyMat);
}
#endif // HAVE_HALF

template <typename T>
void DeConv2d<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    auto xShape = inputs[0]->shape;
    auto wShape = inputs[1]->shape;
    auto yShape = output->shape;

    auto batch        = (int)xShape.dim(0);
    auto inputHeight  = (int)xShape.dim(1);
    auto inputWidth   = (int)xShape.dim(2);
    auto inputChannel = (int)xShape.dim(3);

    auto outputHeight  = (int)yShape.dim(1);
    auto outputWidth   = (int)yShape.dim(2);
    auto outputChannel = (int)yShape.dim(3);

    auto filterHeight = (int)wShape.dim(1);
    auto filterWidth  = (int)wShape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * (int)(forwardStrideY) - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * (int)(forwardStrideX) - 2) / 2);

    if (0 == index) {
        backwardGPUInputImpl(device, iGradient->data(), inputs[1]->data(), outputGradient->data(),
        batch, inputHeight, inputWidth, inputChannel,
		outputHeight, outputWidth, outputChannel,
		filterHeight, filterWidth, (int)forwardStrideY, (int)forwardStrideX,
		padTop, padLeft);
    } else if (1 == index) {
        backwardGPUFilterImpl(device, inputs[0]->data(), iGradient->data(), outputGradient->data(),
        batch, inputHeight, inputWidth, inputChannel,
		outputHeight, outputWidth, outputChannel,
		filterHeight, filterWidth, (int)forwardStrideY, (int)forwardStrideX,
		padTop, padLeft);
    }
}

DEEP8_DECLARATION_GPU_FUNC(DeConv2d);

#endif

}