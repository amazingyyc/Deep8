#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "utils/GPUMathUtils.h"
#include "math/MaxPooling2d.h"

namespace Deep8 {
namespace Math {

#ifdef HAVE_CUDNN
template <typename T>
void MaxPooling2dGPUImpl(GPUDevice* device, 
                         const T *x, 
                         const Shape &xshape, 
                         T *y, 
                         const Shape &yshape,
                         const int windowsHeight,   
                         const int windowsWidth,
                         const int verticalPadding, 
                         const int horizontalPadding,
                         const int verticalStride,  
                         const int horizontalStride) {
    DEEP8_RUNTIME_ERROR("the type is not support");      
}

template <>
void MaxPooling2dGPUImpl<float>(GPUDevice* device, 
                                const float *x, 
                                const Shape &xshape, 
                                float *y, 
                                const Shape &yshape,
                                const int windowsHeight,   
                                const int windowsWidth,
                                const int verticalPadding, 
                                const int horizontalPadding,
                                const int verticalStride,  
                                const int horizontalStride) {
    float alpha = 1;
    float beta  = 0;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, 
                                            CUDNN_POOLING_MAX, 
                                            CUDNN_PROPAGATE_NAN,
                                            windowsHeight, 
                                            windowsWidth, 
                                            verticalPadding, 
                                            horizontalPadding, 
                                            verticalStride, 
                                            horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, 
                                            CUDNN_TENSOR_NHWC, 
                                            CUDNN_DATA_FLOAT, 
                                            (int)xshape.batch, 
                                            (int)xshape.dim(2), 
                                            (int)xshape.dim(0), 
                                            (int)xshape.dim(1)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, 
                                            CUDNN_TENSOR_NHWC, 
                                            CUDNN_DATA_FLOAT, 
                                            (int)yshape.batch, 
                                            (int)yshape.dim(2), 
                                            (int)yshape.dim(0), 
                                            (int)yshape.dim(1)));

    CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

template <>
void MaxPooling2dGPUImpl<double>(GPUDevice* device, 
                                const double *x, 
                                const Shape &xshape, 
                                double *y, 
                                const Shape &yshape,
                                const int windowsHeight,   
                                const int windowsWidth,
                                const int verticalPadding, 
                                const int horizontalPadding,
                                const int verticalStride,  
                                const int horizontalStride) {
    double alpha = 1;
    double beta  = 0;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xshape.batch, (int)xshape.dim(2), (int)xshape.dim(0), (int)xshape.dim(1)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yshape.batch, (int)yshape.dim(2), (int)yshape.dim(0), (int)yshape.dim(1)));

    CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

#ifdef HAVE_HALF
template <>
void MaxPooling2dGPUImpl<half>(GPUDevice* device, 
                                const half *x, 
                                const Shape &xshape, 
                                half *y, 
                                const Shape &yshape,
                                const int windowsHeight,   
                                const int windowsWidth,
                                const int verticalPadding, 
                                const int horizontalPadding,
                                const int verticalStride,  
                                const int horizontalStride) {
    half alpha(1.0);
    half beta(0.0);

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
        windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xshape.batch, (int)xshape.dim(2), (int)xshape.dim(0), (int)xshape.dim(1)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yshape.batch, (int)yshape.dim(2), (int)yshape.dim(0), (int)yshape.dim(1)));

    CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}
#endif
#endif

void MaxPooling2dGPU(const Tensor &x, Tensor &y, bool coverd, int filterHeight, int filterWidth, int strideY, int strideX) {
    auto inputHeight = (int)(x.shape.dim(0));
    auto inputWidth  = (int)(x.shape.dim(1));

    auto outputHeight = (int)(y.shape.dim(0));
    auto outputWidth  = (int)(y.shape.dim(1));

    int padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    int padX = std::max<int>(0, (outputWidth  - 1) * strideX + filterWidth  - inputWidth);

    int padTop  = (padY / 2);
    int padLeft = (padX / 2);

    auto device = x.device();

#ifdef HAVE_CUDNN

    switch (x.type.id) {
        case DType::Float32:
            MaxPooling2dGPUImpl<float>(device, 
                                    x.data<float>(), 
                                    x.shape, 
                                    y.data<float>(), 
                                    y.shape,
                                    filterHeight,   
                                    filterWidth,
                                    padTop, 
                                    padLeft, 
                                    strideY, 
                                    strideX);
            break;
        case DType::Float64:
            MaxPooling2dGPUImpl<double>(device, 
                                    x.data<double>(), 
                                    x.shape, 
                                    y.data<double>(), 
                                    y.shape,
                                    filterHeight,   
                                    filterWidth,
                                    padTop, 
                                    padLeft, 
                                    strideY, 
                                    strideX);
            break;
#ifdef HAVE_HALF
        case DType::Float16:
            MaxPooling2dGPUImpl<half>(device, 
                                    x.data<half>(), 
                                    x.shape, 
                                    y.data<half>(), 
                                    y.shape,
                                    filterHeight,   
                                    filterWidth,
                                    padTop, 
                                    padLeft, 
                                    strideY, 
                                    strideX);
        break;
#endif
        default:
            DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
            break;
    }
#else
    DEEP8_RUNTIME_ERROR("the MaxPooling2d needs CUDNN");
#endif
}

#ifdef HAVE_CUDNN
template <typename T>
void MaxPooling2dGradGPUImpl(GPUDevice *device, 
                         const T *x,
                         T *dx,
                         const Shape &xshape,
                         const T *y,
                         const T *dy,
                         const Shape &yshape,
                         int windowsHeight, 
                         int windowsWidth,
                         int verticalPadding, 
                         int horizontalPadding,
                         int verticalStride, 
                         int horizontalStride) {
    DEEP8_RUNTIME_ERROR("the type in not support");
}

template <>
void MaxPooling2dGradGPUImpl<float>(GPUDevice *device, 
                                const float *x,
                                float *dx,
                                const Shape &xshape,
                                const float *y,
                                const float *dy,
                                const Shape &yshape,
                                int windowsHeight, 
                                int windowsWidth,
                                int verticalPadding, 
                                int horizontalPadding,
                                int verticalStride, 
                                int horizontalStride) {
    float alpha = 1;
    float beta = 1;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)xshape.batch, (int)xshape.dim(2), (int)xshape.dim(0), (int)xshape.dim(1)));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)xshape.batch, (int)xshape.dim(2), (int)xshape.dim(0), (int)xshape.dim(1)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)yshape.batch, (int)yshape.dim(2), (int)yshape.dim(0), (int)yshape.dim(1)));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)yshape.batch, (int)yshape.dim(2), (int)yshape.dim(0), (int)yshape.dim(1)));

    CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

template <>
void MaxPooling2dGradGPUImpl<double>(GPUDevice *device, 
                                const double *x,
                                double *dx,
                                const Shape &xshape,
                                const double *y,
                                const double *dy,
                                const Shape &yshape,
                                int windowsHeight, 
                                int windowsWidth,
                                int verticalPadding, 
                                int horizontalPadding,
                                int verticalStride, 
                                int horizontalStride) {
    double alpha = 1;
    double beta = 1;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xshape.batch, (int)xshape.dim(2), (int)xshape.dim(0), (int)xshape.dim(1)));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xshape.batch, (int)xshape.dim(2), (int)xshape.dim(0), (int)xshape.dim(1)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yshape.batch, (int)yshape.dim(2), (int)yshape.dim(0), (int)yshape.dim(1)));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yshape.batch, (int)yshape.dim(2), (int)yshape.dim(0), (int)yshape.dim(1)));

    CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

#ifdef HAVE_HALF
template <>
void MaxPooling2dGradGPUImpl<half>(GPUDevice *device, 
                                const half *x,
                                half *dx,
                                const Shape &xshape,
                                const half *y,
                                const half *dy,
                                const Shape &yshape,
                                int windowsHeight, 
                                int windowsWidth,
                                int verticalPadding, 
                                int horizontalPadding,
                                int verticalStride, 
                                int horizontalStride) {
    half alpha(1.0);
    half beta(1.0);

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
        windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xshape.batch, (int)xshape.dim(2), (int)xshape.dim(0), (int)xshape.dim(1)));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xshape.batch, (int)xshape.dim(2), (int)xshape.dim(0), (int)xshape.dim(1)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yshape.batch, (int)yshape.dim(2), (int)yshape.dim(0), (int)yshape.dim(1)));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yshape.batch, (int)yshape.dim(2), (int)yshape.dim(0), (int)yshape.dim(1)));

    CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}
#endif
#endif


void MaxPooling2dGradGPU(const Tensor &x, 
                        Tensor &dx, 
                        const Tensor &y, 
                        const Tensor &dy, 
                        bool coverd, 
                        int filterHeight, 
                        int filterWidth, 
                        int strideY, 
                        int strideX) {
    auto inputHeight = (int)(x.shape.dim(0));
    auto inputWidth  = (int)(x.shape.dim(1));

    auto outputHeight = (int)(y.shape.dim(0));
    auto outputWidth  = (int)(y.shape.dim(1));

    int padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    int padX = std::max<int>(0, (outputWidth  - 1) * strideX + filterWidth  - inputWidth);

    int padTop  = (padY / 2);
    int padLeft = (padX / 2);

    auto device = x.device();

#ifdef HAVE_CUDNN
    switch (x.type.id) {
        case DType::Float32:
            MaxPooling2dGradGPUImpl<float>(device, 
                                x.data<float>(),
                                dx.data<float>(),
                                x.shape,
                                y.data<float>(),
                                dy.data<float>(),
                                y.shape,
                                filterHeight, 
                                filterWidth, 
                                padTop, 
                                padLeft, 
                                strideY, 
                                strideX);
            break;
        case DType::Float64:
            MaxPooling2dGradGPUImpl<double>(device, 
                    x.data<double>(),
                    dx.data<double>(),
                    x.shape,
                    y.data<double>(),
                    dy.data<double>(),
                    y.shape,
                    filterHeight, 
                    filterWidth, 
                    padTop, 
                    padLeft, 
                    strideY, 
                    strideX);
            break;
#ifdef HAVE_HALF
        case DType::Float16:
            MaxPooling2dGradGPUImpl<half>(device, 
                    x.data<half>(),
                    dx.data<half>(),
                    x.shape,
                    y.data<half>(),
                    dy.data<half>(),
                    y.shape,
                    filterHeight, 
                    filterWidth, 
                    padTop, 
                    padLeft, 
                    strideY, 
                    strideX);
        break;
#endif
        default:
            DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
            break;
    }
#else
    DEEP8_RUNTIME_ERROR("the MaxPooling2d needs CUDNN");
#endif
}










}
}