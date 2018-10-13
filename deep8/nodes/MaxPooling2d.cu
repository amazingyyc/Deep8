#include "MaxPooling2d.h"

namespace Deep8 {

#ifdef HAVE_CUDA

#ifdef HAVE_CUDNN
template <>
void MaxPooling2d<float>::forwardGPUCUDNNImpl(Device *d, const float *x, const Shape &xShape, float *y, const Shape &yShape,
                         int windowsHeight,   int windowsWidth,
                         int verticalPadding, int horizontalPadding,
                         int verticalStride,  int horizontalStride) {
    auto device = (GPUDevice*)d;

    float alpha = 1;
    float beta  = 0;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

template <>
void MaxPooling2d<double>::forwardGPUCUDNNImpl(Device *d, const double *x, const Shape &xShape, double *y, const Shape &yShape,
                         int windowsHeight,   int windowsWidth,
                         int verticalPadding, int horizontalPadding,
                         int verticalStride,  int horizontalStride) {
    auto device = (GPUDevice*)d;

    double alpha = 1;
    double beta  = 0;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

#ifdef HAVE_HALF
template <>
void MaxPooling2d<half>::forwardGPUCUDNNImpl(Device *d, const half *x, const Shape &xShape, half *y, const Shape &yShape,
                        int windowsHeight, int windowsWidth,
                        int verticalPadding, int horizontalPadding,
                        int verticalStride, int horizontalStride) {
    auto device = (GPUDevice*)d;

    half alpha = 1;
    half beta = 0;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
        windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}
#endif
#endif

template <typename T>
void MaxPooling2d<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output)  {
#ifdef HAVE_CUDNN

    auto inputH = static_cast<int>(inputs[0]->shape.dim(1));
    auto inputW = static_cast<int>(inputs[0]->shape.dim(2));

    auto outputH = static_cast<int>(output->shape.dim(1));
    auto outputW = static_cast<int>(output->shape.dim(2));

    int padY = std::max<int>(0, (outputH - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputH);
    int padX = std::max<int>(0, (outputW - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth) - inputW);

    int padTop  = (padY / 2);
    int padLeft = (padX / 2);

    forwardGPUCUDNNImpl(output->device(), inputs[0]->data(), inputs[0]->shape, output->data(), output->shape,
        (int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);
#else
    DEEP8_RUNTIME_ERROR("the MaxPooling2d needs CUDNN");
#endif
}


#ifdef HAVE_CUDNN

template <>
void MaxPooling2d<float>::backwardGPUCUDNNImpl(Device *d, const float *x, float *dx, const Shape &xShape, const float *y, const float *dy, const Shape &yShape,
                          int windowsHeight, int windowsWidth,
                          int verticalPadding, int horizontalPadding,
                          int verticalStride, int horizontalStride) {
    auto device = (GPUDevice*)d;

    float alpha = 1;
    float beta = 1;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

template <>
void MaxPooling2d<double>::backwardGPUCUDNNImpl(Device *d, const double *x, double *dx, const Shape &xShape, const double *y, const double *dy, const Shape &yShape,
                          int windowsHeight, int windowsWidth,
                          int verticalPadding, int horizontalPadding,
                          int verticalStride, int horizontalStride) {
    auto device = (GPUDevice*)d;

    double alpha = 1;
    double beta = 1;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

#ifdef HAVE_HALF
template <>
void MaxPooling2d<half>::backwardGPUCUDNNImpl(Device *d, const half *x, half *dx, const Shape &xShape, const half *y, const half *dy, const Shape &yShape,
                                    int windowsHeight, int windowsWidth,
                                    int verticalPadding, int horizontalPadding,
                                    int verticalStride, int horizontalStride) {
    auto device = (GPUDevice*)d;

    half alpha = 1;
    half beta = 1;

    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
        windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

    CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}
#endif
#endif

template <typename T>
void MaxPooling2d<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {
#ifdef HAVE_CUDNN

    auto inputH = static_cast<int>(iGradient->shape.dim(1));
    auto inputW = static_cast<int>(iGradient->shape.dim(2));

    auto outputH = static_cast<int>(outputGradient->shape.dim(1));
    auto outputW = static_cast<int>(outputGradient->shape.dim(2));

    int padY = std::max<int>(0, (outputH - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputH);
    int padX = std::max<int>(0, (outputW - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth) - inputW);

    int padTop  = (padY / 2);
    int padLeft = (padX / 2);

    backwardGPUCUDNNImpl(output->device(),
        inputs[0]->data(), iGradient->data(), iGradient->shape,
        output->data(), outputGradient->data(), output->shape,
        (int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);

#else
    DEEP8_RUNTIME_ERROR("the MaxPooling2d needs CUDNN");
#endif
}

DEEP8_DECLARATION_GPU_FUNC(MaxPooling2d);

#endif

}