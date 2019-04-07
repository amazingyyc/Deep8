#include "math/MaxIndex2d.h"

namespace Deep8 {
namespace Math {

void MaxIndex2d(const Tensor &x,
                Tensor &index,
                bool covered, 
                int filterHeight, 
                int filterWidth, 
                int strideY, 
                int strideX) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == index.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(index.elementType.id == DType::Int32, "the index element type must be int");
    DEEP8_ARGUMENT_CHECK(3 == x.nDims() && 3 == index.nDims(), "the tensor dim must be 3");
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

    DEEP8_ARGUMENT_CHECK(batch       == (int)index.batch() &&
                        outputHeight == (int)index.dim(0) &&
                        outputWidth  == (int)index.dim(1) &&
                        inputChannel == (int)index.dim(2), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        MaxIndex2dCPU(x, index, covered, filterHeight, filterWidth, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        MaxIndex2dGPU(x, index, covered, filterHeight, filterWidth, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void MaxIndex2dCPUImpl(const T *x,
                        int *index,
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
                }
            }
        }
    }
}

void MaxIndex2dCPU(const Tensor &x,
                    Tensor &index,
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

    auto outputHeight = (int)index.dim(0);
    auto outputWidth  = (int)index.dim(1);

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
            MaxIndex2dCPUImpl<float>(x,
                                    index,
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
            MaxIndex2dCPUImpl<double>(x,
                                    index,
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
                                               x.data<double>(),
                                               index.data<int>(),
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