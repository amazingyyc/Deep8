#include "math/MaxUnPooling2d.h"

namespace Deep8 {
namespace Math {

void MaxUnPooling2d(const Tensor &x,
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

    auto batch         = (int)y.batch();
    auto outputHeight  = (int)y.dim(0);
    auto outputWidth   = (int)y.dim(1);
    auto outputChannel = (int)y.dim(2);

    int inputHeight;
    int inputWidth;

    if (!covered) {
        inputHeight = (outputHeight - filterHeight) / strideY + 1;
        inputWidth  = (outputWidth  - filterWidth) / strideX + 1;
    } else {
        inputHeight = (outputHeight - 1) / strideY + 1;
        inputWidth  = (outputWidth  - 1) / strideX + 1;
    }

    DEEP8_ARGUMENT_CHECK(batch         == (int)x.batch() &&
                         inputHeight   == (int)x.dim(0) &&
                         inputWidth    == (int)x.dim(1) &&
                         outputChannel == (int)x.dim(2), "the shape is error");
    DEEP8_ARGUMENT_CHECK(index.shape == x.shape, "the index and y shape must be same");
    
    if (DeviceType::CPU == x.deviceType()) {
        MaxUnPooling2dCPU(x, index, y, covered, filterHeight, filterWidth, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        MaxUnPooling2dGPU(x, index, y, covered, filterHeight, filterWidth, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void MaxUnPooling2dGrad(const Tensor &x,
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

    auto batch         = (int)y.batch();
    auto outputHeight  = (int)y.dim(0);
    auto outputWidth   = (int)y.dim(1);
    auto outputChannel = (int)y.dim(2);

    int inputHeight;
    int inputWidth;

    if (!covered) {
        inputHeight = (outputHeight - filterHeight) / strideY + 1;
        inputWidth = (outputWidth - filterWidth) / strideX + 1;
    } else {
        inputHeight = (outputHeight - 1) / strideY + 1;
        inputWidth = (outputWidth - 1) / strideX + 1;
    }

    DEEP8_ARGUMENT_CHECK(batch         == (int)x.batch() &&
                         inputHeight   == (int)x.dim(0) &&
                         inputWidth    == (int)x.dim(1) &&
                         outputChannel == (int)x.dim(2), "the shape is error");
    DEEP8_ARGUMENT_CHECK(index.shape == x.shape, "the index and y shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        MaxUnPooling2dGradCPU(x, dx, index, y, dy, covered, filterHeight, filterWidth, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        MaxUnPooling2dGradGPU(x, dx, index, y, dy, covered, filterHeight, filterWidth, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void MaxUnPooling2dCPUImpl(const T *x,
                           int *index,
                           T *y,
                           int batch,
                           int startChannel,
                           int endChannel,
                           int inputHeight,
                           int inputWidth,
                           int outputHeight,
                           int outputWidth,
                           int channel) {
    int maxj = batch * outputHeight * outputWidth * channel;

    /**first set y be 0*/
    for (int b = 0; b < batch; ++b) {
        for (int c = startChannel; c < endChannel; ++c) {
            for (int h = 0; h < outputHeight; ++h) {
                for (int w = 0; w < outputWidth; ++w) {
                    y[((b * outputHeight + h) * outputWidth + w) * channel + c] = T(0);
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int c = startChannel; c < endChannel; ++c) {
            for (int h = 0; h < inputHeight; ++h) {
                for (int w = 0; w < inputWidth; ++w) {
                    int i = ((b * inputHeight + h) * inputWidth + w) * channel + c;
                    int j = index[i];

                    if (0 <= j && j < maxj) {
                        y[j] = x[i];
                    }
                }
            }
        }
    }
}

void MaxUnPooling2dCPU(const Tensor &x,
                       const Tensor &index,
                       Tensor &y,
                       bool covered,
                       int filterHeight,
                       int filterWidth,
                       int strideY,
                       int strideX) {
    auto device = (CPUDevice*)x.device();
    auto eigenDevice = device->eigenDevice;

    auto batch = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth = (int)x.dim(1);
    auto channel = (int)x.dim(2);

    auto outputHeight = (int)y.dim(0);
    auto outputWidth = (int)y.dim(1);

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
                                    int channel) {
            MaxUnPooling2dCPUImpl<float>(x,
                                        index,
                                        y,
                                        batch,
                                        startChannel,
                                        endChannel,
                                        inputHeight,
                                        inputWidth,
                                        outputHeight,
                                        outputWidth,
                                        channel);

            barrier.Notify();
        };

        for (int i = 0; i < threadNum; ++i) {
            int startChannel = i * blockSize;
            int endChannel = std::min<int>(startChannel + blockSize, channel);

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
                                               channel);
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
                                    int channel) {
            MaxUnPooling2dCPUImpl<double>(x,
                                        index,
                                        y,
                                        batch,
                                        startChannel,
                                        endChannel,
                                        inputHeight,
                                        inputWidth,
                                        outputHeight,
                                        outputWidth,
                                        channel);

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
                                               channel);
        }

        barrier.Wait();
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
    }
}

template <typename T>
void MaxUnPooling2dGradCPUImpl(T *x,
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
                               int channel) {
    int maxj = batch * outputHeight * outputWidth * channel;

    for (int b = 0; b < batch; ++b) {
        for (int c = startChannel; c < endChannel; ++c) {
            for (int h = 0; h < inputHeight; ++h) {
                for (int w = 0; w < inputWidth; ++w) {
                    int i = ((b * inputHeight + h) * inputWidth + w) * channel + c;
                    int j = index[i];

                    if (0 <= j && j < maxj) {
                        dx[i] += dy[j];
                    }
                }
            }
        }
    }
}

void MaxUnPooling2dGradCPU(const Tensor &x,
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

    auto batch = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth = (int)x.dim(1);
    auto channel = (int)x.dim(2);

    auto outputHeight = (int)y.dim(0);
    auto outputWidth = (int)y.dim(1);

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
                                    int channel) {
            MaxUnPooling2dGradCPUImpl<float>(x,
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
                                            channel);

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
                                               channel);
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
                                    int channel) {
            MaxUnPooling2dGradCPUImpl<double>(x,
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
                        channel);

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
                                               channel);
        }

        barrier.Wait();
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
    }
}











}
}