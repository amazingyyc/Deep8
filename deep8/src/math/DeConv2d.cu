#include "math/DeConv2d.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void DeConv2dCol2ImKernel(const T *outputMat, 
                                    T *output,
                                    const int batch, 
                                    const int inputHeight, 
                                    const int inputWidth, 
                                    const int inputChannel, 
                                    const int filterHeight, 
                                    const int filterWidth, 
                                    const int outputHeight, 
                                    const int outputWidth, 
                                    const int outputChannel, 
                                    const int strideY, 
                                    const int strideX, 
                                    const int padTop, 
                                    const int padLeft,
                                    const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int b = i / (outputHeight * outputWidth * outputChannel);

        int outputY = (i % (outputHeight * outputWidth * outputChannel)) / (outputWidth * outputChannel);
        int outputX = (i % (outputWidth * outputChannel)) / outputChannel;
        int outputOffset = i % outputChannel;

        T out = 0;

        for (int y = 0; y < filterHeight; ++y) {
            for (int x = 0; x < filterWidth; ++x) {
                int inputY = outputY + padTop + y;
                int inputX = outputX + padLeft + x;

                if (0 == inputY % strideY && 0 == inputX % strideX) {
                    inputY /= strideY;
                    inputX /= strideX;

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

template <typename T>
void DeConv2dGPUImpl(GPUDevice *device, 
                    const T *x, 
                    const Shape &xshape, 
                    const T *y, 
                    const Shape &yshape, 
                    T *z, 
                    const Shape &zshape,
                    float *zmat,
                    bool convered,
                    int strideY,
                    int strideX) {
    DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
void DeConv2dGPUImpl<float>(GPUDevice *device, 
                            const float *x, 
                            const Shape &xshape, 
                            const float *y, 
                            const Shape &yshape,
                            float *z,
                            const Shape &zshape,
                            float *zmat,
                            bool convered,
                            int strideY,
                            int strideX) {
    auto batch        = (int)xshape.batch;
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int m = batch * inputHeight * inputWidth;
    int k = inputChannel;
    int n = outputChannel * filterHeight * filterWidth;

    float alpha = 1;
    float beta  = 0;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, y, k, x, k, &beta, zmat, n));

    int size = batch * outputHeight * outputWidth * outputChannel;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
    
    DeConv2dCol2ImKernel<float><<<grideSize, blockSize>>>(  zmat, 
                                                            z, 
                                                            batch, 
                                                            inputHeight, 
                                                            inputWidth, 
                                                            inputChannel, 
                                                            filterHeight, 
                                                            filterWidth, 
                                                            outputHeight, 
                                                            outputWidth, 
                                                            outputChannel,
                                                            strideY, 
                                                            strideX, 
                                                            padTop, 
                                                            padLeft, 
                                                            size);
}   

template <>
void DeConv2dGPUImpl<double>(GPUDevice *device, 
                            const double *x, 
                            const Shape &xshape, 
                            const double *y, 
                            const Shape &yshape,
                            double *z,
                            const Shape &zshape,
                            double *zmat,
                            bool convered,
                            int strideY,
                            int strideX) {
    auto batch        = (int)xshape.batch;
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int m = batch * inputHeight * inputWidth;
    int k = inputChannel;
    int n = outputChannel * filterHeight * filterWidth;

    double alpha = 1;
    double beta  = 0;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, y, k, x, k, &beta, zmat, n));

    int size = batch * outputHeight * outputWidth * outputChannel;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
    
    DeConv2dCol2ImKernel<double><<<grideSize, blockSize>>>( zmat, 
                                                            z, 
                                                            batch, 
                                                            inputHeight, 
                                                            inputWidth, 
                                                            inputChannel, 
                                                            filterHeight, 
                                                            filterWidth, 
                                                            outputHeight, 
                                                            outputWidth, 
                                                            outputChannel,
                                                            strideY, 
                                                            strideX, 
                                                            padTop, 
                                                            padLeft, 
                                                            size);
}   

#ifdef HAVE_HALF
template <>
void DeConv2dGPUImpl<half>(GPUDevice *device, 
                            const half *x, 
                            const Shape &xshape, 
                            const half *y, 
                            const Shape &yshape,
                            half *z,
                            const Shape &zshape,
                            half *zmat,
                            bool convered,
                            int strideY,
                            int strideX) {
    auto batch        = (int)xshape.batch;
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int m = batch * inputHeight * inputWidth;
    int k = inputChannel;
    int n = outputChannel * filterHeight * filterWidth;

    half alpha(1.0);
    half beta(0.0);

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, y, k, x, k, &beta, zmat, n));

    int size = batch * outputHeight * outputWidth * outputChannel;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
    
    DeConv2dCol2ImKernel<half><<<grideSize, blockSize>>>( zmat,
                                                            z, 
                                                            batch, 
                                                            inputHeight, 
                                                            inputWidth, 
                                                            inputChannel, 
                                                            filterHeight, 
                                                            filterWidth, 
                                                            outputHeight, 
                                                            outputWidth, 
                                                            outputChannel,
                                                            strideY, 
                                                            strideX, 
                                                            padTop, 
                                                            padLeft, 
                                                            size);
}   
#endif

void DeConv2dGPU(   const Tensor &x, 
                    const Tensor &y, 
                    Tensor &z,
                    void *zmat,
                    bool convered,
                    int strideY,
                    int strideX) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        DeConv2dGPUImpl<float>(device, 
                                x.data<float>(), 
                                x.shape, 
                                y.data<float>(), 
                                y.shape, 
                                z.data<float>(), 
                                z.shape,
                                zmat,
                                convered,
                                strideY,
                                strideX);
        break;
    case DType::Float64:
        DeConv2dGPUImpl<double>(device, 
                                x.data<double>(), 
                                x.shape, 
                                y.data<double>(), 
                                y.shape, 
                                z.data<double>(), 
                                z.shape,
                                zmat,
                                convered,
                                strideY,
                                strideX);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        DeConv2dGPUImpl<half>(device, 
                                x.data<half>(), 
                                x.shape, 
                                y.data<half>(), 
                                y.shape, 
                                z.data<half>(), 
                                z.shape,
                                zmat,
                                convered,
                                strideY,
                                strideX);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
__global__ void DeConv2dIm2ColKernel(T *dymat, 
                                    const T *dy, 
                                    const int batch, 
                                    const int inputHeight, 
                                    const int inputWidth, 
                                    const int inputChannel,
                                    const int filterHeight, 
                                    const int filterWidth, 
                                    const int outputHeight, 
                                    const int outputWidth, 
                                    const int outputChannel,
                                    const int strideY, 
                                    const int strideX,
                                    const int padTop, 
                                    const int padLeft, 
                                    const int N) {
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

        int outputY = inputY * strideY - padTop - filterY;
        int outputX = inputX * strideX - padLeft - filterX;

        if (0 <= outputY && outputY < outputHeight && 0 <= outputX && outputX < outputWidth) {
            dymat[i] = dy[((b * outputHeight + outputY) * outputWidth + outputX) * outputChannel + outputOffset];
        } else {
            dymat[i] = 0;
        }
    }
}

template <typename T>
void DeConv2dGradXGPUImpl(GPUDevice *device,
                        const T *x, 
                        T *dx,
                        const Shape &xshape,
                        const T *y,
                        const Shape &yshape,
                        const T *z, 
                        const T *dz,
                        const Shape &zshape,
                        T *dzmat,
                        bool convered,
                        int strideY,
                        int strideX) {
    DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
void DeConv2dGradXGPUImpl<float>(GPUDevice *device,
                                const float *x, 
                                float *dx,
                                const Shape &xshape,
                                const float *y,
                                const Shape &yshape,
                                const float *z, 
                                const float *dz,
                                const Shape &zshape,
                                float *dzmat,
                                bool convered,
                                int strideY,
                                int strideX) {
    auto batch        = (int)xshape.batch;
    
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    DeConv2dIm2ColKernel<float><<<grideSize, blockSize>>>(dzmat, 
                                                        dz,
                                                        batch, 
                                                        inputHeight, 
                                                        inputWidth, 
                                                        inputChannel, 
                                                        filterHeight, 
                                                        filterWidth, 
                                                        outputHeight, 
                                                        outputWidth, 
                                                        outputChannel,
                                                        strideY, 
                                                        strideX,
                                                        padTop, 
                                                        padLeft, 
                                                        size);
    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    float alpha = 1;
    float beta  = 1;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k,m, n, &alpha, y, k, dzmat, n, &beta, dx, k));
}

template <>
void DeConv2dGradXGPUImpl<double>(GPUDevice *device,
                                const double *x, 
                                double *dx,
                                const Shape &xshape,
                                const double *y,
                                const Shape &yshape,
                                const double *z, 
                                const double *dz,
                                const Shape &zshape,
                                double *dzmat,
                                bool convered,
                                int strideY,
                                int strideX) {
    auto batch        = (int)xshape.batch;
    
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    DeConv2dIm2ColKernel<double><<<grideSize, blockSize>>>(dzmat, 
                                                        dz,
                                                        batch, 
                                                        inputHeight, 
                                                        inputWidth, 
                                                        inputChannel, 
                                                        filterHeight, 
                                                        filterWidth, 
                                                        outputHeight, 
                                                        outputWidth, 
                                                        outputChannel,
                                                        strideY, 
                                                        strideX,
                                                        padTop, 
                                                        padLeft, 
                                                        size);

    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    double alpha = 1;
    double beta = 1;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, y, k, dzmat, n, &beta, dx, k));
}

#ifdef HAVE_HALF
template <>
void DeConv2dGradXGPUImpl<half>(GPUDevice *device,
                                const half *x, 
                                half *dx,
                                const Shape &xshape,
                                const half *y,
                                const Shape &yshape,
                                const half *z, 
                                const half *dz,
                                const Shape &zshape,
                                half *dzmat,
                                bool convered,
                                int strideY,
                                int strideX) {
    auto batch        = (int)xshape.batch;
    
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    DeConv2dIm2ColKernel<half><<<grideSize, blockSize>>>(dzmat, 
                                                        dz,
                                                        batch, 
                                                        inputHeight, 
                                                        inputWidth, 
                                                        inputChannel, 
                                                        filterHeight, 
                                                        filterWidth, 
                                                        outputHeight, 
                                                        outputWidth, 
                                                        outputChannel,
                                                        strideY, 
                                                        strideX,
                                                        padTop, 
                                                        padLeft, 
                                                        size);

    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    half alpha(1.0);
    half beta(1.0);

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, y, k, dzmat, n, &beta, dx, k));
}
#endif

void DeConv2dGradXGPU(  const Tensor& x, 
                        Tensor& dx,
                        const Tensor& y,
                        const Tensor& z, 
                        const Tensor& dz,
                        void *dzmat,
                        bool convered,
                        int strideY,
                        int strideX) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        DeConv2dGradXGPUImpl<float>(device, 
            x.data<float>(), 
            dx.data<float>(), 
            x.shape,
            y.data<float>(),
            y.shape,
            z.data<float>(),
            dz.data<float>(),
            z.shape,
            dzmat,
            convered,
            strideY,
            strideX);
        break;
    case DType::Float64:
        DeConv2dGradXGPUImpl<double>(device, 
            x.data<double>(), 
            dx.data<double>(), 
            x.shape,
            y.data<double>(),
            y.shape,
            z.data<double>(),
            dz.data<double>(),
            z.shape,
            dzmat,
            convered,
            strideY,
            strideX);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        DeConv2dGradXGPUImpl<half>(device, 
            x.data<half>(), 
            dx.data<half>(), 
            x.shape,
            y.data<half>(),
            y.shape,
            z.data<half>(),
            dz.data<half>(),
            z.shape,
            dzmat,
            convered,
            strideY,
            strideX);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
void DeConv2dGradYGPUImpl(GPUDevice *device,
                            const T *x, 
                            const Shape &xshape,
                            const T *y,
                            T *dy,
                            const Shape &yshape,
                            const T *z, 
                            const T *dz,
                            const Shape &zshape,
                            T *dzmat,
                            bool convered,
                            int strideY,
                            int strideX) {
    DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
void DeConv2dGradYGPUImpl<float>(GPUDevice *device,
                            const float *x, 
                            const Shape &xshape,
                            const float *y,
                            float *dy,
                            const Shape &yshape,
                            const float *z, 
                            const float *dz,
                            const Shape &zshape,
                            float *dzmat,
                            bool convered,
                            int strideY,
                            int strideX) {
    auto batch        = (int)xshape.batch;
    
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    DeConv2dIm2ColKernel<float><<<grideSize, blockSize>>>(dyzat, 
                                                            dz,
                                                            batch, 
                                                            inputHeight, 
                                                            inputWidth, 
                                                            inputChannel,
                                                            filterHeight, 
                                                            filterWidth,
                                                            outputHeight, 
                                                            outputWidth, 
                                                            outputChannel,
                                                            strideY, 
                                                            strideX,
                                                            padTop, 
                                                            padLeft, 
                                                            size);

    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    float alpha = 1;
    float beta  = 1;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, n, m, &alpha, x, k, dzmat, n, &beta, dy, k));
}

template <>
void DeConv2dGradYGPUImpl<double>(GPUDevice *device,
                            const double *x, 
                            const Shape &xshape,
                            const double *y,
                            double *dy,
                            const Shape &yshape,
                            const double *z, 
                            const double *dz,
                            const Shape &zshape,
                            double *dzmat,
                            bool convered,
                            int strideY,
                            int strideX) {
    auto batch        = (int)xshape.batch;
    
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    DeConv2dIm2ColKernel<double><<<grideSize, blockSize>>>(dyzat, 
                                                            dz,
                                                            batch, 
                                                            inputHeight, 
                                                            inputWidth, 
                                                            inputChannel,
                                                            filterHeight, 
                                                            filterWidth,
                                                            outputHeight, 
                                                            outputWidth, 
                                                            outputChannel,
                                                            strideY, 
                                                            strideX,
                                                            padTop, 
                                                            padLeft, 
                                                            size);

    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    double alpha = 1;
    double beta  = 1;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, n, m, &alpha, x, k, dzmat, n, &beta, dy, k));
}

#ifdef HAVE_HALF
template <>
void DeConv2dGradYGPUImpl<half>(GPUDevice *device,
                            const half *x, 
                            const Shape &xshape,
                            const half *y,
                            half *dy,
                            const Shape &yshape,
                            const half *z, 
                            const half *dz,
                            const Shape &zshape,
                            half *dzmat,
                            bool convered,
                            int strideY,
                            int strideX) {
    auto batch        = (int)xshape.batch;
    
    auto inputHeight  = (int)xshape.dim(0);
    auto inputWidth   = (int)xshape.dim(1);
    auto inputChannel = (int)xshape.dim(2);

    auto outputHeight  = (int)zshape.dim(0);
    auto outputWidth   = (int)zshape.dim(1);
    auto outputChannel = (int)zshape.dim(2);

    auto filterHeight = (int)yshape.dim(1);
    auto filterWidth  = (int)yshape.dim(2);

    auto padTop  = -(std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto padLeft = -(std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2) / 2);

    int size = batch * inputHeight * inputWidth * outputChannel * filterHeight * filterWidth;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    DeConv2dIm2ColKernel<half><<<grideSize, blockSize>>>(dyzat, 
                                                            dz,
                                                            batch, 
                                                            inputHeight, 
                                                            inputWidth, 
                                                            inputChannel,
                                                            filterHeight, 
                                                            filterWidth,
                                                            outputHeight, 
                                                            outputWidth, 
                                                            outputChannel,
                                                            strideY, 
                                                            strideX,
                                                            padTop, 
                                                            padLeft, 
                                                            size);

    int m = batch * inputHeight * inputWidth;
    int n = outputChannel * filterHeight * filterWidth;
    int k = inputChannel;

    half alpha(1.0);
    half beta(1.0);

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, n, m, &alpha, x, k, dzmat, n, &beta, dy, k));
}
#endif

void DeConv2dGradYGPU(  const Tensor &x,
                        const Tensor &y, 
                        Tensor &dy,
                        const Tensor &z, 
                        const Tensor& dz,
                        void *dzmat,
                        bool convered,
                        int strideY,
                        int strideX) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        DeConv2dGradYGPUImpl<float>(device, 
            x.data<float>(), 
            x.shape,
            y.data<float>(),
            dy.data<float>(),
            y.shape,
            z.data<float>(),
            dz.data<float>(),
            z.shape,
            dzmat,
            convered,
            strideY,
            strideX);
        break;
    case DType::Float64:
        DeConv2dGradYGPUImpl<double>(device, 
            x.data<double>(), 
            x.shape,
            y.data<double>(),
            dy.data<double>(),
            y.shape,
            z.data<double>(),
            dz.data<double>(),
            z.shape,
            dzmat,
            convered,
            strideY,
            strideX);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        DeConv2dGradYGPUImpl<half>(device, 
            x.data<half>(), 
            x.shape,
            y.data<half>(),
            dy.data<half>(),
            y.shape,
            z.data<half>(),
            dz.data<half>(),
            z.shape,
            dzmat,
            convered,
            strideY,
            strideX);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

}
}