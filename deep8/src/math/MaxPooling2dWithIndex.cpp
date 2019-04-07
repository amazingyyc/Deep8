#include "math/MaxPooling2dWithIndex.h"

namespace Deep8 {
namespace Math {

void MaxPooling2dWithIndex(const Tensor &x,
                           const Tensor &index,
                           Tensor &y,
                           bool covered,
                           int filterHeight,
                           int filterWidth,
                           int strideY,
                           int strideX) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType() && x.deviceType() == index.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType, "the data type must be same");
    DEEP8_ARGUMENT_CHECK(index.elementType.id == DType::Int32, "the index element type must be int");
    DEEP8_ARGUMENT_CHECK(3 == x.nDims() && 3 == y.nDims(), "the tensor dim must be 3");
    DEEP8_ARGUMENT_CHECK(filterHeight >= 1 && filterWidth >= 1 && strideY >= 1 && strideX >= 1, "the params is error");

    auto batch        = (int)x.batch();
    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    int outputHeight;
    int outputWidth;

    if (!covered) {
        outputHeight = (inputHeight - filterHeight) / strideY + 1;
        outputWidth  = (inputWidth  - filterWidth) / strideX + 1;
    } else {
        outputHeight = (inputHeight - 1) / strideY + 1;
        outputWidth  = (inputWidth  - 1) / strideX + 1;
    }

    DEEP8_ARGUMENT_CHECK(batch        == (int)y.batch() &&
                         outputHeight == (int)y.dim(0) &&
                         outputWidth  == (int)y.dim(1) &&
                         inputChannel == (int)y.dim(2), "the shape is error");
    DEEP8_ARGUMENT_CHECK(index.shape == y.shape, "the index and y shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        MaxPooling2dWithIndexCPU(x, index, y, covered, filterHeight, filterWidth, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        MaxPooling2dWithIndexGPU(x, index, y, covered, filterHeight, filterWidth, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void MaxPooling2dWithIndexGrad(const Tensor &x,
                               Tensor &dx,
                               const Tensor &index,
                               const Tensor &y,
                               const Tensor &dy,
                               bool covered,
                               int filterHeight,
                               int filterWidth,
                               int strideY,
                               int strideX) {
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape, "the x and dx shape must be same");
    DEEP8_ARGUMENT_CHECK(y.shape == dy.shape, "the y and dy shape must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == dx.elementType && x.elementType == y.elementType && x.elementType == dy.elementType, "the type must be same");
    DEEP8_ARGUMENT_CHECK(index.elementType.id == DType::Int32, "the index element type must be int");
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() &&
                         x.deviceType() == index.deviceType() &&
                         x.deviceType() == y.deviceType() &&
                         x.deviceType() == dy.deviceType(), "the device type must be same");

    auto batch        = (int)x.batch();
    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    int outputHeight;
    int outputWidth;

    if (!covered) {
        outputHeight = (inputHeight - filterHeight) / strideY + 1;
        outputWidth  = (inputWidth - filterWidth) / strideX + 1;
    } else {
        outputHeight = (inputHeight - 1) / strideY + 1;
        outputWidth  = (inputWidth - 1) / strideX + 1;
    }

    DEEP8_ARGUMENT_CHECK(outputHeight == (int)y.dim(0) &&
                         outputWidth  == (int)y.dim(1) &&
                         inputChannel == (int)y.dim(2), "the shape is error");
    DEEP8_ARGUMENT_CHECK(index.shape == y.shape, "the index and y shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        MaxPooling2dWithIndexGradCPU(x, dx, index, y, dy, covered, filterHeight, filterWidth, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        MaxPooling2dWithIndexGradGPU(x, dx, index, y, dy, covered, filterHeight, filterWidth, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void MaxPooling2dWithIndexCPUImpl(const T *x,
                              int *index,
                              T *y,
                              int batch,
                              int startChannel,
                              int endChannel,
                              int inputHeight,
                              int inputWidth,
                              int outputHeight,
                              int outputWidth,
                              int channel,
                              int filterHeight,
                              int filterWidth,
                              int strideY,
                              int strideX,
                              int padTop,
                              int padLeft) {
    for (int b = 0; b < batch; ++b) {
        auto xptr     = x     + b * inputHeight  * inputWidth  * channel;
        auto indexptr = index + b * outputHeight * outputWidth * channel;
        auto yptr     = y     + b * outputHeight * outputWidth * channel;

        for (int c = startChannel; c < endChannel; ++c) {
            for (int h = 0; h < outputHeight; ++h) {
                for (int w = 0; w < outputWidth; ++w) {
                    auto starth = std::max<int>(0, padTop + h * strideY);
                    auto endh   = std::min<int>(inputHeight, padTop + h * strideY + filterHeight);

                    auto startw = std::max<int>(0, padLeft + w * strideX);
                    auto endw   = std::min<int>(inputWidth, padLeft + w * strideX + filterWidth);

                    DEEP8_ARGUMENT_CHECK(endh >= starth && endw >= startw, "the input or output dim is error");

                    int maxh = starth;
                    int maxw = startw;

                    auto maxvalue = xptr[starth * inputWidth * channel + startw * channel + c];

                    for (int xh = starth; xh < endh; ++xh) {
                        for (int xw = startw; xw < endw; ++xw) {
                            auto value = xptr[xh * inputWidth * channel + xw * channel + c];

                            if (value > maxvalue) {
                                maxvalue = value;

                                maxh = xh;
                                maxw = xw;
                            }
                        }
                    }

                    indexptr[h * outputWidth * channel + w * channel + c] = ((b * inputHeight + maxh) * inputWidth + maxw) * channel + c;
                    yptr[h * outputWidth * channel + w * channel + c] = maxvalue;
                }
            }
        }
    }
}

void MaxPooling2dWithIndexCPU(const Tensor &x,
                              const Tensor &index,
                              Tensor &y,
                              bool covered,
                              int filterHeight,
                              int filterWidth,
                              int strideY,
                              int strideX) {
    auto device = (CPUDevice*)x.device();
    auto eigenDevice = device->eigenDevice;

    auto batch       = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth  = (int)x.dim(1);
    auto channel     = (int)x.dim(2);

    auto outputHeight = (int)y.dim(0);
    auto outputWidth  = (int)y.dim(1);

    auto padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    auto padX = std::max<int>(0, (outputWidth  - 1) * strideX + filterWidth - inputWidth);

    int padTop  = -(padY / 2);
    int padLeft = -(padX / 2);

    int threadNum = (int)eigenDevice->numThreads();
    int blockSize = (channel + threadNum - 1) / threadNum;

    Eigen::Barrier barrier((unsigned int)(threadNum));

    if (DType::Float32 == x.elementType.id) {
        auto blockFunc = [&barrier](const float *x,
                                    int *index,
                                    float *y,
                                    int batch,
                                    int startChannel,
                                    int endChannel,
                                    int inputHeight,
                                    int inputWidth,
                                    int outputHeight,
                                    int outputWidth,
                                    int channel,
                                    int filterHeight,
                                    int filterWidth,
                                    int strideY,
                                    int strideX,
                                    int padTop,
                                    int padLeft) {
            MaxPooling2dWithIndexCPUImpl<float>(x,
                                            index,
                                            y,
                                            batch,
                                            startChannel,
                                            endChannel,
                                            inputHeight,
                                            inputWidth,
                                            outputHeight,
                                            outputWidth,
                                            channel,
                                            filterHeight,
                                            filterWidth,
                                            strideY,
                                            strideX,
                                            padTop,
                                            padLeft);

            barrier.Notify();
        };

        for (int i = 0; i < threadNum; ++i) {
            int startChannel = i * blockSize;
            int endChannel   = std::min<int>(startChannel + blockSize, channel);

            eigenDevice->enqueueNoNotification(blockFunc,
                                               x.data<float>(),
                                               index.data<int>(),
                                               y.data<float>(),
                                               batch,
                                               startChannel,
                                               endChannel,
                                               inputHeight,
                                               inputWidth,
                                               outputHeight,
                                               outputWidth,
                                               channel,
                                               filterHeight,
                                               filterWidth,
                                               strideY,
                                               strideX,
                                               padTop,
                                               padLeft);
        }

        barrier.Wait();
    } else if (DType::Float64 == x.elementType.id) {
        auto blockFunc = [&barrier](const double *x,
                                    int *index,
                                    double *y,
                                    int batch,
                                    int startChannel,
                                    int endChannel,
                                    int inputHeight,
                                    int inputWidth,
                                    int outputHeight,
                                    int outputWidth,
                                    int channel,
                                    int filterHeight,
                                    int filterWidth,
                                    int strideY,
                                    int strideX,
                                    int padTop,
                                    int padLeft) {
            MaxPooling2dWithIndexCPUImpl<double>(x,
                                                index,
                                                y,
                                                batch,
                                                startChannel,
                                                endChannel,
                                                inputHeight,
                                                inputWidth,
                                                outputHeight,
                                                outputWidth,
                                                channel,
                                                filterHeight,
                                                filterWidth,
                                                strideY,
                                                strideX,
                                                padTop,
                                                padLeft);

            barrier.Notify();
        };

        for (int i = 0; i < threadNum; ++i) {
            int startChannel = i * blockSize;
            int endChannel = std::min<int>(startChannel + blockSize, channel);

            eigenDevice->enqueueNoNotification(blockFunc,
                                               x.data<double>(),
                                               index.data<int>(),
                                               y.data<double>(),
                                               batch,
                                               startChannel,
                                               endChannel,
                                               inputHeight,
                                               inputWidth,
                                               outputHeight,
                                               outputWidth,
                                               channel,
                                               filterHeight,
                                               filterWidth,
                                               strideY,
                                               strideX,
                                               padTop,
                                               padLeft);
        }

        barrier.Wait();
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
    }
}

template <typename T>
void MaxPooling2dWithIndexGradCPUImpl(T *x,
                                     T *dx,
                                     int *index,
                                     T *dy,
                                     int batch,
                                     int startChannel,
                                     int endChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int outputHeight,
                                     int outputWidth,
                                     int channel,
                                     int filterHeight,
                                     int filterWidth,
                                     int strideY,
                                     int strideX,
                                     int padTop,
                                     int padLeft) {
    int maxxi = batch * inputHeight * inputWidth * channel;

    for (int b = 0; b < batch; ++b) {
        auto dyptr = dy + b * outputHeight * outputWidth * channel;
        auto indexptr = index + b * outputHeight * outputWidth * channel;

        for (int c = startChannel; c < endChannel; ++c) {
            for (int h = 0; h < outputHeight; ++h) {
                for (int w = 0; w < outputWidth; ++w) {
                    auto yi = h * outputWidth * channel + w * channel + c;
                    auto xi = indexptr[yi];

                    DEEP8_ARGUMENT_CHECK(0 <= xi && xi < maxxi, "the index is error");

                    dx[xi] += dyptr[yi];
                }
            }
        }
    }
}

void MaxPooling2dWithIndexGradCPU(const Tensor &x,
                                  Tensor &dx,
                                  const Tensor &index,
                                  const Tensor &y,
                                  const Tensor &dy,
                                  bool covered,
                                  int filterHeight,
                                  int filterWidth,
                                  int strideY,
                                  int strideX) {
    auto device = (CPUDevice*)x.device();
    auto eigenDevice = device->eigenDevice;

    auto batch       = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth  = (int)x.dim(1);
    auto channel     = (int)x.dim(2);

    auto outputHeight = (int)y.dim(0);
    auto outputWidth  = (int)y.dim(1);

    auto padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    auto padX = std::max<int>(0, (outputWidth  - 1) * strideX + filterWidth  - inputWidth);

    int padTop  = -(padY / 2);
    int padLeft = -(padX / 2);

    int threadNum = (int)eigenDevice->numThreads();
    int blockSize = (channel + threadNum - 1) / threadNum;

    Eigen::Barrier barrier((unsigned int)(threadNum));

    if (DType::Float32 == x.elementType.id) {
        auto blockFunc = [&barrier](float *x,
                                    float *dx,
                                    int *index,
                                    float *dy,
                                    int batch,
                                    int startChannel,
                                    int endChannel,
                                    int inputHeight,
                                    int inputWidth,
                                    int outputHeight,
                                    int outputWidth,
                                    int channel,
                                    int filterHeight,
                                    int filterWidth,
                                    int strideY,
                                    int strideX,
                                    int padTop,
                                    int padLeft) {
            MaxPooling2dWithIndexGradCPUImpl<float>(x,
                                            dx,
                                            index,
                                            dy,
                                            batch,
                                            startChannel,
                                            endChannel,
                                            inputHeight,
                                            inputWidth,
                                            outputHeight,
                                            outputWidth,
                                            channel,
                                            filterHeight,
                                            filterWidth,
                                            strideY,
                                            strideX,
                                            padTop,
                                            padLeft);

            barrier.Notify();
        };

        for (int i = 0; i < threadNum; ++i) {
            int startChannel = i * blockSize;
            int endChannel = std::min<int>(startChannel + blockSize, channel);

            eigenDevice->enqueueNoNotification(blockFunc,
                                               x.data<float>(),
                                               dx.data<float>(),
                                               index.data<int>(),
                                               dy.data<float>(),
                                               batch,
                                               startChannel,
                                               endChannel,
                                               inputHeight,
                                               inputWidth,
                                               outputHeight,
                                               outputWidth,
                                               channel,
                                               filterHeight,
                                               filterWidth,
                                               strideY,
                                               strideX,
                                               padTop,
                                               padLeft);
        }

        barrier.Wait();
    } else if (DType::Float64 == x.elementType.id) {
        auto blockFunc = [&barrier](double *x,
                                    double *dx,
                                    int *index,
                                    double *dy,
                                    int batch,
                                    int startChannel,
                                    int endChannel,
                                    int inputHeight,
                                    int inputWidth,
                                    int outputHeight,
                                    int outputWidth,
                                    int channel,
                                    int filterHeight,
                                    int filterWidth,
                                    int strideY,
                                    int strideX,
                                    int padTop,
                                    int padLeft) {
            MaxPooling2dWithIndexGradCPUImpl<double>(x,
                                                    dx,
                                                    index,
                                                    dy,
                                                    batch,
                                                    startChannel,
                                                    endChannel,
                                                    inputHeight,
                                                    inputWidth,
                                                    outputHeight,
                                                    outputWidth,
                                                    channel,
                                                    filterHeight,
                                                    filterWidth,
                                                    strideY,
                                                    strideX,
                                                    padTop,
                                                    padLeft);

            barrier.Notify();
        };

        for (int i = 0; i < threadNum; ++i) {
            int startChannel = i * blockSize;
            int endChannel = std::min<int>(startChannel + blockSize, channel);

            eigenDevice->enqueueNoNotification(blockFunc,
                                               x.data<double>(),
                                               dx.data<double>(),
                                               index.data<int>(),
                                               dy.data<double>(),
                                               batch,
                                               startChannel,
                                               endChannel,
                                               inputHeight,
                                               inputWidth,
                                               outputHeight,
                                               outputWidth,
                                               channel,
                                               filterHeight,
                                               filterWidth,
                                               strideY,
                                               strideX,
                                               padTop,
                                               padLeft);
        }

        barrier.Wait();
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
    }
}













}
}