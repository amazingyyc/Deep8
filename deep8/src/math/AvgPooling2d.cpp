#include "math/AvgPooling2d.h"

namespace Deep8 {
namespace Math {

/**avg pooling 2d*/
void AvgPooling2d(const Tensor &x, Tensor &y, 
                  bool covered, 
                  int filterHeight, 
                  int filterWidth, 
                  int strideY, 
                  int strideX) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType, "the data type must be same");
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
        outputWidth  = (inputWidth  - filterWidth)  / strideX + 1;
    } else {
        outputHeight = (inputHeight - 1) / strideY + 1;
        outputWidth  = (inputWidth  - 1) / strideX + 1;
    }

    DEEP8_ARGUMENT_CHECK(batch        == (int)y.batch() &&
                         outputHeight == (int)y.dim(0) && 
                         outputWidth  == (int)y.dim(1) && 
                         inputChannel == (int)y.dim(2), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        AvgPooling2dCPU(x, y, covered, filterHeight, filterWidth, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        AvgPooling2dGPU(x, y, covered, filterHeight, filterWidth, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

/**grad of AvgPooling*/
void AvgPooling2dGrad(const Tensor &x,
                      Tensor &dx,
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
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && 
                         x.deviceType() ==  y.deviceType() && 
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
        outputWidth  = (inputWidth  - 1) / strideX + 1;
    }

    DEEP8_ARGUMENT_CHECK(outputHeight == (int)y.dim(0) &&
                         outputWidth  == (int)y.dim(1) &&
                         inputChannel == (int)y.dim(2), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        AvgPooling2dGradCPU(x, dx, y, dy, covered, filterHeight, filterWidth, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        AvgPooling2dGradGPU(x, dx, y, dy, covered, filterHeight, filterWidth, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void AvgPooling2dCPUImpl(CPUDevice *device,
                        T *x, 
                        T *y,
                        int batch,
                        int inputHeight, 
                        int inputWidth,
                        int outputHeight, 
                        int outputWidth,
                        int channel,
                        int filterHeight, 
                        int filterWidth,
                        int strideY, 
                        int strideX) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        xtensor(x, batch, inputHeight, inputWidth, channel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        ytensor(y, batch, outputHeight, outputWidth, channel);

    Eigen::DSizes<int, 4> preDims;
    preDims[0] = batch * outputHeight * outputWidth;
    preDims[1] = filterHeight;
    preDims[2] = filterWidth;
    preDims[3] = channel;

    Eigen::DSizes<int, 2> reductionDims;
    reductionDims[0] = 1;
    reductionDims[1] = 2;

    auto padY = std::max<int>(0, (outputHeight - 1) * strideY + filterHeight - inputHeight);
    auto padX = std::max<int>(0, (outputWidth  - 1) * strideX + filterWidth  - inputWidth);

    auto padTop    = padY / 2;
    auto padBottom = padY - padTop;
    auto padLeft   = padX / 2;
    auto padRight  = padX - padLeft;

    ytensor.device(*eigenDevice) = xtensor.extract_image_patches(
        filterWidth, filterHeight, strideX, strideY, 1, 1, 1, 1, padTop, padBottom, padLeft, padRight, 0)
        .reshape(preDims)
        .mean(reductionDims)
        .reshape(ytensor.dimensions());
}

void AvgPooling2dCPU(const Tensor &x, Tensor &y, 
                     bool covered, 
                     int filterHeight, 
                     int filterWidth, 
                     int strideY, 
                     int strideX) {
    auto device = (CPUDevice*)x.device();

    auto batch       = (int)x.batch();
    auto inputHeight = (int)x.dim(0);
    auto inputWidth  = (int)x.dim(1);
    auto channel     = (int)x.dim(2);

    auto outputHeight = (int)y.dim(0);
    auto outputWidth  = (int)y.dim(1);

    switch (x.elementType.id) {
    case DType::Float32:
        AvgPooling2dCPUImpl<float>(device,
                            x.data<float>(),
                            y.data<float>(),
                            batch,
                            inputHeight,
                            inputWidth,
                            outputHeight, 
                            outputWidth,
                            channel,
                            filterHeight, 
                            filterWidth,
                            strideY, 
                            strideX);
        break;
    case DType::Float64:
        AvgPooling2dCPUImpl<double>(device,
                                   x.data<double>(), 
                                   y.data<double>(),
                                   batch,
                                   inputHeight, 
                                   inputWidth,
                                   outputHeight, 
                                   outputWidth,
                                   channel,
                                   filterHeight, 
                                   filterWidth,
                                   strideY, 
                                   strideX);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


template <typename T>
void AvgPooling2dGradCPUImpl(T *dx, 
                             const T *dy,
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
    T ratio = T(1) / (T(filterHeight) * T(filterWidth));

    for (int b = 0; b < batch; ++b) {
        auto dxptr = dx + b * inputHeight  * inputWidth  * channel;
        auto dyptr = dy + b * outputHeight * outputWidth * channel;

        for (int k = startChannel; k < endChannel; ++k) {
            for (int y = 0; y < outputHeight; ++y) {
                for (int x = 0; x < outputWidth; ++x) {
                    auto startH = std::max<int>(0, padTop + y * strideY);
                    auto endH   = std::min<int>(inputHeight, padTop + y * strideY + filterHeight);

                    auto startW = std::max<int>(0, padLeft + x * strideX);
                    auto endW   = std::min<int>(inputWidth, padLeft + x * strideX + filterWidth);

                    if (startH >= endH || startW >= endW) {
                        continue;
                    }

                    auto grad = dyptr[y * outputWidth * channel + x * channel + k];

                    for (int inH = startH; inH < endH; ++inH) {
                        for (int inW = startW; inW < endW; ++inW) {
                            dxptr[inH * inputWidth * channel + inW * channel + k] += (ratio * grad);
                        }
                    }
                }
            }
        }
    }
}

void AvgPooling2dGradCPU(const Tensor &x, Tensor &dx,
                         const Tensor &y, const Tensor &dy,
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

    int threadNum = (int) eigenDevice->numThreads();
    int blockSize = (channel + threadNum - 1) / threadNum;

    Eigen::Barrier barrier((unsigned int)(threadNum));

    if (DType::Float32 == x.elementType.id) {
        auto blockFunc = [&barrier](float *dx,
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
             AvgPooling2dGradCPUImpl<float>(dx,
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
                                          dx.data<float>(),
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
        auto blockFunc = [&barrier](double *dx,
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
            AvgPooling2dGradCPUImpl<double>(dx,
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
            int endChannel   = std::min<int>(startChannel + blockSize, channel);

            eigenDevice->enqueueNoNotification(blockFunc,
                                               dx.data<double>(),
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
