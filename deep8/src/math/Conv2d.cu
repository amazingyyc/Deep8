#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/Conv2d.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void Conv2dIm2ColKernel(const T* im, 
                                   T* col,
                                   const int batch, 
                                   const int inputHeight, 
                                   const int inputWidth, 
                                   const int inputChannel,
                                   const int outputHeight,
                                   const int outputWidth,
                                   const int filterHeight, 
                                   const int filterWidth,
                                   const int strideY, 
                                   const int strideX, 
                                   const int dilationY, 
                                   const int dilationX,
                                   const int padTop,
                                   const int padLeft,
                                   const int N) {
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

        int inputY = padTop + outputY * strideY + filterY * dilationY;
        int inputX = padLeft + outputX * strideX + filterX * dilationX;

        T* colPtr = col + colY * filterHeight * filterWidth * inputChannel + colX * inputChannel;

        if (0 > inputY || inputY >= inputHeight || 0 > inputX || inputX >= inputWidth) {
            for (int k = 0; k < inputChannel; ++k) {
                colPtr[k] = 0;
            }
        } else {
            const T* imPtr = im + b * inputHeight * inputWidth * inputChannel + inputY * inputWidth * inputChannel + inputX * inputChannel;

            for (int k = 0; k < inputChannel; ++k) {
                colPtr[k] = imPtr[k];
            }
        }
    }
}

template <typename T>
__global__ void Conv2dCol2ImKernel(const T* col, 
                                   T* im,
                                   const int batch,
                                   const int inputHeight,
                                   const int inputWidth,
                                   const int inputChannel,
                                   const int outputHeight,
                                   const int outputWidth,
                                   const int filterHeight, 
                                   const int filterWidth, 
                                   const int strideY, 
                                   const int strideX, 
                                   const int dilationY, 
                                   const int dilationX,
                                   const int padTop,
                                   const int padLeft,
                                   const int N) {

    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int colWidth = filterHeight * filterWidth * inputChannel;

    for (int i = start; i < N; i += stride) {
        int b = i / (inputHeight * inputWidth);

        int inputIndex = i % (inputHeight * inputWidth);

        int inputY = inputIndex / inputWidth;
        int inputX = inputIndex % inputWidth;

        T* imPtr = im + b * inputHeight * inputWidth * inputChannel + inputY * inputWidth * inputChannel + inputX * inputChannel;

        for (int filterY = 0; filterY < filterHeight; ++filterY) {
            for (int filterX = 0; filterX < filterWidth; ++filterX) {
                int outputY = inputY - padTop - filterY * dilationY;
                int outputX = inputX - padLeft - filterX * dilationX;

                if (0 == (outputY % strideY) && 0 == (outputX % strideX)) {
                    outputY /= strideY;
                    outputX /= strideX;

                    if (0 <= outputY && outputY < outputHeight && 0 <= outputX && outputX < outputWidth) {
                        const T* colPtr = col + (b * outputHeight * outputWidth + outputY * outputWidth + outputX) * colWidth
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

template <typename T>
void Conv2dGPUImpl(GPUDevice* device,
                   const T *x,
                   const T *y,
                   T *z,
                   T *xcol,
                   const int batch,
                   const int inputHeight,
                   const int inputWidth,
                   const int inputChannel,
                   const int outputHeight,
                   const int outputWidth,
                   const int outputChannel,
                   const int filterHeight,
                   const int filterWidth,
                   const int strideY,
                   const int strideX,
                   const int dilationY,
                   const int dilationX,
                   const int padTop,
                   const int padLeft) {
    DEEP8_RUNTIME_ERROR("the type is error!");
}

template <>
void Conv2dGPUImpl<float>(GPUDevice* device,
                          const float* x,
                          const float* y,
                          float* z,
                          float* xcol,
                          const int batch,
                          const int inputHeight,
                          const int inputWidth,
                          const int inputChannel,
                          const int outputHeight,
                          const int outputWidth,
                          const int outputChannel,
                          const int filterHeight,
                          const int filterWidth,
                          const int strideY,
                          const int strideX,
                          const int dilationY,
                          const int dilationX,
                          const int padTop,
                          const int padLeft) {
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    Conv2dIm2ColKernel<float> << <grideSize, blockSize >> > (x, 
                                                             xcol,
                                                             batch, 
                                                             inputHeight, 
                                                             inputWidth, 
                                                             inputChannel, 
                                                             outputHeight,
                                                             outputWidth,
                                                             filterHeight, 
                                                             filterWidth, 
                                                             padTop, 
                                                             padLeft,
                                                             strideY, 
                                                             strideX, 
                                                             dilationY,
                                                             dilationX,
                                                             size);

    int m = batch * outputHeight * outputWidth;
    int k = filterHeight * filterWidth * inputChannel;
    int n = outputChannel;

    float alpha = 1;
    float beta  = 0;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, y, k, xcol, k, &beta, z, n));
}

template <>
void Conv2dGPUImpl<double>(GPUDevice* device,
                           const double* x,
                           const double* y,
                           double* z,
                           double* xcol,
                           const int batch,
                           const int inputHeight,
                           const int inputWidth,
                           const int inputChannel,
                           const int outputHeight,
                           const int outputWidth,
                           const int outputChannel,
                           const int filterHeight,
                           const int filterWidth,
                           const int strideY,
                           const int strideX,
                           const int dilationY,
                           const int dilationX,
                           const int padTop,
                           const int padLeft) {
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    Conv2dIm2ColKernel<double> << <grideSize, blockSize >> > (x,
                                                              xcol,
                                                              batch,
                                                              inputHeight,
                                                              inputWidth,
                                                              inputChannel,
                                                              outputHeight,
                                                              outputWidth,
                                                              filterHeight,
                                                              filterWidth,
                                                              padTop,
                                                              padLeft,
                                                              strideY,
                                                              strideX,
                                                              dilationY,
                                                              dilationX,
                                                              size);

    int m = batch * outputHeight * outputWidth;
    int k = filterHeight * filterWidth * inputChannel;
    int n = outputChannel;

    double alpha = 1;
    double beta  = 0;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, y, k, xcol, k, &beta, z, n));
}

#ifdef HAVE_HALF
template <>
void Conv2dGPUImpl<half>(GPUDevice* device,
                         const half *x,
                         const half *y,
                         half *z,
                         half *xcol,
                         const int batch,
                         const int inputHeight,
                         const int inputWidth,
                         const int inputChannel,
                         const int outputHeight,
                         const int outputWidth,
                         const int outputChannel,
                         const int filterHeight,
                         const int filterWidth,
                         const int strideY,
                         const int strideX,
                         const int dilationY,
                         const int dilationX,
                         const int padTop,
                         const int padLeft) {
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    Conv2dIm2ColKernel<half> << <grideSize, blockSize >> > (x,
                                                            xcol,
                                                            batch,
                                                            inputHeight,
                                                            inputWidth,
                                                            inputChannel,
                                                            outputHeight,
                                                            outputWidth,
                                                            filterHeight,
                                                            filterWidth,
                                                            padTop,
                                                            padLeft,
                                                            strideY,
                                                            strideX,
                                                            dilationY,
                                                            dilationX,
                                                            size);

    int m = batch * outputHeight * outputWidth;
    int k = filterHeight * filterWidth * inputChannel;
    int n = outputChannel;

    half alpha(1.0);
    half beta(0.0);

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, y, k, xcol, k, &beta, z, n));
}
#endif

void Conv2dGPU( const Tensor &x, 
                const Tensor &y, 
                Tensor &z,
                void *xcol,
                bool convered,
                int strideY,
                int strideX,
                int dilationY,
                int dilationX) {
    auto device = (GPUDevice*) x.device();

    auto batch = (int)x.batch();

    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    auto outputHeight  = (int)z.dim(0);
    auto outputWidth   = (int)z.dim(1);
    auto outputChannel = (int)z.dim(2);

    auto filterHeight = (int)y.dim(1);
    auto filterWidth  = (int)y.dim(2);

    auto realFilterHeight = filterHeight + (filterHeight - 1) * (dilationY - 1);
    auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * (dilationX - 1);

    auto padTop  = -(std::max<int>(0, (outputHeight - 1) * strideY + realFilterHeight - inputHeight) / 2);
    auto padLeft = -(std::max<int>(0, (outputWidth  - 1) * strideX + realFilterWidth  - inputWidth)  / 2);

    switch (x.elementType.id) {
    case DType::Float32:
        Conv2dGPUImpl<float>(device,
                          x.data<float>(),
                          y.data<float>(),
                          z.data<float>(),
                          (float*)xcol,
                          batch,
                          inputHeight,
                          inputWidth,
                          inputChannel,
                          outputHeight,
                          outputWidth,
                          outputChannel,
                          filterHeight,
                          filterWidth,
                          strideY,
                          strideX,
                          dilationY,
                          dilationX,
                          padTop,
                          padLeft);
        break;
    case DType::Float64:
        Conv2dGPUImpl<double>(device,
                          x.data<double>(),
                          y.data<double>(),
                          z.data<double>(),
                          (double*)xcol,
                          batch,
                          inputHeight,
                          inputWidth,
                          inputChannel,
                          outputHeight,
                          outputWidth,
                          outputChannel,
                          filterHeight,
                          filterWidth,
                          strideY,
                          strideX,
                          dilationY,
                          dilationX,
                          padTop,
                          padLeft);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        Conv2dGPUImpl<half>(device,
                          x.data<half>(),
                          y.data<half>(),
                          z.data<half>(),
                          (half*)xcol,
                          batch,
                          inputHeight,
                          inputWidth,
                          inputChannel,
                          outputHeight,
                          outputWidth,
                          outputChannel,
                          filterHeight,
                          filterWidth,
                          strideY,
                          strideX,
                          dilationY,
                          dilationX,
                          padTop,
                          padLeft);
    break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

/**grad of x*/
template <typename T>
void Conv2dGradXGPUImpl(GPUDevice* device,
                        const T* x, T* dx,
                        const T* y,
                        const T* z, const T* dz,
                        T *dxcol,
                        const int batch,
                        const int inputHeight,
                        const int inputWidth,
                        const int inputChannel,
                        const int outputHeight,
                        const int outputWidth,
                        const int outputChannel,
                        const int filterHeight,
                        const int filterWidth,
                        const bool convered,
                        const int strideY,
                        const int strideX,
                        const int dilationY,
                        const int dilationX,
                        const int padTop,
                        const int padLeft) {
    DEEP8_RUNTIME_ERROR("the type is error!");
}

template <>
void Conv2dGradXGPUImpl<float>(GPUDevice* device,
                        const float* x, float* dx,
                        const float* y,
                        const float* z, const float* dz,
                        float* dxcol,
                        const int batch,
                        const int inputHeight,
                        const int inputWidth,
                        const int inputChannel,
                        const int outputHeight,
                        const int outputWidth,
                        const int outputChannel,
                        const int filterHeight,
                        const int filterWidth,
                        const bool convered,
                        const int strideY,
                        const int strideX,
                        const int dilationY,
                        const int dilationX,
                        const int padTop,
                        const int padLeft) {
    int size = batch * inputHeight * inputWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    int m = filterHeight * filterWidth * inputChannel;
    int k = outputChannel;
    int n = batch * outputHeight * outputWidth;

    float alpha = 1;
    float beta = 0;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, y, m, dz, k, &beta, dxcol, m));

    Conv2dCol2ImKernel<float> << <grideSize, blockSize >> > (dxcol,
                                                             dx,
                                                             batch,
                                                             inputHeight,
                                                             inputWidth,
                                                             inputChannel,
                                                             outputHeight,
                                                             outputWidth,
                                                             filterHeight,
                                                             filterWidth,
                                                             strideY,
                                                             strideX,
                                                             dilationY,
                                                             dilationX,
                                                             padTop,
                                                             padLeft,
                                                             size);
}

template <>
void Conv2dGradXGPUImpl<double>(GPUDevice* device,
                               const double* x, double* dx,
                               const double* y,
                               const double* z, const double* dz,
                               double* dxcol,
                               const int batch,
                               const int inputHeight,
                               const int inputWidth,
                               const int inputChannel,
                               const int outputHeight,
                               const int outputWidth,
                               const int outputChannel,
                               const int filterHeight,
                               const int filterWidth,
                               const bool convered,
                               const int strideY,
                               const int strideX,
                               const int dilationY,
                               const int dilationX,
                               const int padTop,
                               const int padLeft) {
    int size = batch * inputHeight * inputWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    int m = filterHeight * filterWidth * inputChannel;
    int k = outputChannel;
    int n = batch * outputHeight * outputWidth;

    double alpha = 1;
    double beta = 0;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, y, m, dz, k, &beta, dxcol, m));

    Conv2dCol2ImKernel<double> << <grideSize, blockSize >> > (dxcol,
                                                              dx,
                                                              batch,
                                                              inputHeight,
                                                              inputWidth,
                                                              inputChannel,
                                                              outputHeight,
                                                              outputWidth,
                                                              filterHeight,
                                                              filterWidth,
                                                              strideY,
                                                              strideX,
                                                              dilationY,
                                                              dilationX,
                                                              padTop,
                                                              padLeft,
                                                              size);
}

#ifdef HAVE_HALF
template <>
void Conv2dGradXGPUImpl<half>(GPUDevice* device,
                                const half* x, half* dx,
                                const half* y,
                                const half* z, const half* dz,
                                half* dxcol,
                                const int batch,
                                const int inputHeight,
                                const int inputWidth,
                                const int inputChannel,
                                const int outputHeight,
                                const int outputWidth,
                                const int outputChannel,
                                const int filterHeight,
                                const int filterWidth,
                                const bool convered,
                                const int strideY,
                                const int strideX,
                                const int dilationY,
                                const int dilationX,
                                const int padTop,
                                const int padLeft) {
    int size = batch * inputHeight * inputWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    int m = filterHeight * filterWidth * inputChannel;
    int k = outputChannel;
    int n = batch * outputHeight * outputWidth;

    half alpha(1.0);
    half beta(0.0);

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, y, m, dz, k, &beta, dxcol, m));

    Conv2dCol2ImKernel<half> << <grideSize, blockSize >> > (dxcol,
                                                            dx,
                                                            batch,
                                                            inputHeight,
                                                            inputWidth,
                                                            inputChannel,
                                                            outputHeight,
                                                            outputWidth,
                                                            filterHeight,
                                                            filterWidth,
                                                            strideY,
                                                            strideX,
                                                            dilationY,
                                                            dilationX,
                                                            padTop,
                                                            padLeft,
                                                            size);
}
#endif

/**gradient for x (input)*/
void Conv2dGradXGPU(const Tensor& x, 
                    Tensor& dx,
                    const Tensor& y,
                    const Tensor& z, 
                    const Tensor& dz,
                    void *dxcol,
                    bool convered,
                    int strideY,
                    int strideX,
                    int dilationY,
                    int dilationX) {
    auto device = (GPUDevice*) x.device();

    auto batch = (int)x.batch();

    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    auto outputHeight  = (int)z.dim(0);
    auto outputWidth   = (int)z.dim(1);
    auto outputChannel = (int)z.dim(2);

    auto filterHeight = (int)y.dim(1);
    auto filterWidth  = (int)y.dim(2);

    auto realFilterHeight = filterHeight + (filterHeight - 1) * ((int)dilationY - 1);
    auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * ((int)dilationX - 1);

    auto padTop  = -(std::max<int>(0, (outputHeight - 1) * strideY + realFilterHeight - inputHeight) / 2);
    auto padLeft = -(std::max<int>(0, (outputWidth  - 1) * strideX + realFilterWidth  - inputWidth)  / 2);

    switch (x.elementType.id) {
    case DType::Float32:
        Conv2dGradXGPUImpl<float>(device,
                        x.data<float>(), 
                        dx.data<float>(),
                        y.data<float>(),
                        z.data<float>(), 
                        dz.data<float>(),
                        (float*)dxcol,
                        batch,
                        inputHeight,
                        inputWidth,
                        inputChannel,
                        outputHeight,
                        outputWidth,
                        outputChannel,
                        filterHeight,
                        filterWidth,
                        convered,
                        strideY,
                        strideX,
                        dilationY,
                        dilationX,
                        padTop,
                        padLeft);
        break;
    case DType::Float64:
        Conv2dGradXGPUImpl<double>(device,
                        x.data<double>(), 
                        dx.data<double>(),
                        y.data<double>(),
                        z.data<double>(), 
                        dz.data<double>(),
                        (double*)dxcol,
                        batch,
                        inputHeight,
                        inputWidth,
                        inputChannel,
                        outputHeight,
                        outputWidth,
                        outputChannel,
                        filterHeight,
                        filterWidth,
                        convered,
                        strideY,
                        strideX,
                        dilationY,
                        dilationX,
                        padTop,
                        padLeft);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        Conv2dGradXGPUImpl<half>(device,
                        x.data<half>(), 
                        dx.data<half>(),
                        y.data<half>(),
                        z.data<half>(), 
                        dz.data<half>(),
                        (half*)dxcol,
                        batch,
                        inputHeight,
                        inputWidth,
                        inputChannel,
                        outputHeight,
                        outputWidth,
                        outputChannel,
                        filterHeight,
                        filterWidth,
                        convered,
                        strideY,
                        strideX,
                        dilationY,
                        dilationX,
                        padTop,
                        padLeft);
    break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

/**grad for y (filter)*/
template <typename T>
void Conv2dGradYGPUImpl(GPUDevice* device,
                        const T* x,
                        const T* y, T *dy,
                        const T* z, const T* dz,
                        T* xcol,
                        const int batch,
                        const int inputHeight,
                        const int inputWidth,
                        const int inputChannel,
                        const int outputHeight,
                        const int outputWidth,
                        const int outputChannel,
                        const int filterHeight,
                        const int filterWidth,
                        const bool convered,
                        const int strideY,
                        const int strideX,
                        const int dilationY,
                        const int dilationX,
                        const int padTop,
                        const int padLeft) {
    DEEP8_RUNTIME_ERROR("the type is error!");
}

template <>
void Conv2dGradYGPUImpl<float>(GPUDevice* device,
                        const float* x,
                        const float* y, float* dy,
                        const float* z, const float* dz,
                        float* xcol,
                        const int batch,
                        const int inputHeight,
                        const int inputWidth,
                        const int inputChannel,
                        const int outputHeight,
                        const int outputWidth,
                        const int outputChannel,
                        const int filterHeight,
                        const int filterWidth,
                        const bool convered,
                        const int strideY,
                        const int strideX,
                        const int dilationY,
                        const int dilationX,
                        const int padTop,
                        const int padLeft) {
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    Conv2dIm2ColKernel<float> << <grideSize, blockSize >> > (x,
                                                             xcol,
                                                             batch,
                                                             inputHeight,
                                                             inputWidth,
                                                             inputChannel,
                                                             outputHeight,
                                                             outputWidth,
                                                             filterHeight,
                                                             filterWidth,
                                                             strideY,
                                                             strideX,
                                                             dilationY,
                                                             dilationX,
                                                             padTop,
                                                             padLeft,
                                                             size);

    int m = filterHeight * filterWidth * inputChannel;
    int k = batch * outputHeight * outputWidth;
    int n = outputChannel;

    float alpha = 1;
    float beta = 1;

    CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, xcol, m, dz, n, &beta, dy, m));
}

template <>
void Conv2dGradYGPUImpl<double>(GPUDevice* device,
                                const double* x,
                                const double* y, double* dy,
                                const double* z, const double* dz,
                                double* xcol,
                                const int batch,
                                const int inputHeight,
                                const int inputWidth,
                                const int inputChannel,
                                const int outputHeight,
                                const int outputWidth,
                                const int outputChannel,
                                const int filterHeight,
                                const int filterWidth,
                                const bool convered,
                                const int strideY,
                                const int strideX,
                                const int dilationY,
                                const int dilationX,
                                const int padTop,
                                const int padLeft) {
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    Conv2dIm2ColKernel<double> << <grideSize, blockSize >> > (x,
                                                              xcol,
                                                              batch,
                                                              inputHeight,
                                                              inputWidth,
                                                              inputChannel,
                                                              outputHeight,
                                                              outputWidth,
                                                              filterHeight,
                                                              filterWidth,
                                                              strideY,
                                                              strideX,
                                                              dilationY,
                                                              dilationX,
                                                              padTop,
                                                              padLeft,
                                                              size);

    int m = filterHeight * filterWidth * inputChannel;
    int k = batch * outputHeight * outputWidth;
    int n = outputChannel;

    double alpha = 1;
    double beta = 1;

    CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, xcol, m, dz, n, &beta, dy, m));
}

#ifdef HAVE_HALF
template <>
void Conv2dGradYGPUImpl<half>(GPUDevice* device,
                              const half* x,
                              const half* y, half* dy,
                              const half* z, const half* dz,
                              half* xcol,
                              const int batch,
                              const int inputHeight,
                              const int inputWidth,
                              const int inputChannel,
                              const int outputHeight,
                              const int outputWidth,
                              const int outputChannel,
                              const int filterHeight,
                              const int filterWidth,
                              const bool convered,
                              const int strideY,
                              const int strideX,
                              const int dilationY,
                              const int dilationX,
                              const int padTop,
                              const int padLeft) {
    int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    Conv2dIm2ColKernel<half> << <grideSize, blockSize >> > (x,
                                                            xcol,
                                                            batch,
                                                            inputHeight,
                                                            inputWidth,
                                                            inputChannel,
                                                            outputHeight,
                                                            outputWidth,
                                                            filterHeight,
                                                            filterWidth,
                                                            strideY,
                                                            strideX,
                                                            dilationY,
                                                            dilationX,
                                                            padTop,
                                                            padLeft,
                                                            size);

    int m = filterHeight * filterWidth * inputChannel;
    int k = batch * outputHeight * outputWidth;
    int n = outputChannel;

    half alpha(1.0);
    half beta(1.0);

    CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, xcol, m, dz, n, &beta, dy, m));
}
#endif


void Conv2dGradYGPU(const Tensor &x,
                    const Tensor &y, 
                    Tensor &dy,
                    const Tensor &z, 
                    const Tensor& dz,
                    void *xcol,
                    bool convered,
                    int strideY,
                    int strideX,
                    int dilationY,
                    int dilationX) {
    auto device = (GPUDevice*) x.device();

    auto batch = (int)x.batch();

    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    auto outputHeight  = (int)z.dim(0);
    auto outputWidth   = (int)z.dim(1);
    auto outputChannel = (int)z.dim(2);

    auto filterHeight = (int)y.dim(1);
    auto filterWidth  = (int)y.dim(2);

    auto realFilterHeight = filterHeight + (filterHeight - 1) * ((int)dilationY - 1);
    auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * ((int)dilationX - 1);

    auto padTop  = -(std::max<int>(0, (outputHeight - 1) * strideY + realFilterHeight - inputHeight) / 2);
    auto padLeft = -(std::max<int>(0, (outputWidth  - 1) * strideX + realFilterWidth  - inputWidth)  / 2);

    switch (x.elementType.id) {
    case DType::Float32:
        Conv2dGradYGPUImpl<float>(device,
                        x.data<float>(),
                        y.data<float>(), 
                        dy.data<float>(),
                        z.data<float>(), 
                        dz.data<float>(),
                        (float*)xcol,
                        batch,
                        inputHeight,
                        inputWidth,
                        inputChannel,
                        outputHeight,
                        outputWidth,
                        outputChannel,
                        filterHeight,
                        filterWidth,
                        convered,
                        strideY,
                        strideX,
                        dilationY,
                        dilationX,
                        padTop,
                        padLeft);
        break;
    case DType::Float64:
        Conv2dGradYGPUImpl<double>(device,
                x.data<double>(),
                y.data<double>(), 
                dy.data<double>(),
                z.data<double>(), 
                dz.data<double>(),
                (double*)xcol,
                batch,
                inputHeight,
                inputWidth,
                inputChannel,
                outputHeight,
                outputWidth,
                outputChannel,
                filterHeight,
                filterWidth,
                convered,
                strideY,
                strideX,
                dilationY,
                dilationX,
                padTop,
                padLeft);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        Conv2dGradYGPUImpl<half>(device,
                x.data<half>(),
                y.data<half>(), 
                dy.data<half>(),
                z.data<half>(), 
                dz.data<half>(),
                (half*)xcol,
                batch,
                inputHeight,
                inputWidth,
                inputChannel,
                outputHeight,
                outputWidth,
                outputChannel,
                filterHeight,
                filterWidth,
                convered,
                strideY,
                strideX,
                dilationY,
                dilationX,
                padTop,
                padLeft);
    break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}