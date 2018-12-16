#include "AvgPooling2d.h"

namespace Deep8 {

template <typename T>
AvgPooling2d<T>::AvgPooling2d(std::vector<Node *> &inputs, bool covered , size_t filterH, size_t filterW, size_t strideH, size_t strideW):
Function<T>(inputs), covered(covered), filterHeight(filterH), filterWidth(filterW), strideY(strideH), strideX(strideW) {
    check();
}

template <typename T>
void AvgPooling2d<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the AvgPooling2d only need 1 input");
    DEEP8_ARGUMENT_CHECK(filterHeight >= 1 && filterWidth >= 1 && strideY >= 1 && strideX >= 1,
                         "the filter size or stride is error");
    DEEP8_ARGUMENT_CHECK(3 == this->inputs[0]->outputShape.nDims,
                         "AvgPooling2d needs inputs nDims is 3");

    auto inputShape = this->inputs[0]->outputShape;

    if (!covered) {
        DEEP8_ARGUMENT_CHECK(filterHeight <= inputShape.dim(0) && filterWidth <= inputShape.dim(1),
                             "the not forwardCovered mode type needs filter smaller than input");
    }

    auto inputH = static_cast<int64_t>(inputShape.dim(0));
    auto inputW = static_cast<int64_t>(inputShape.dim(1));

    std::vector<size_t> outputDim(3);
    outputDim[2] = inputShape.dim(2);

    if (!covered) {
        int64_t outputH =
                (inputH - static_cast<int64_t>(filterHeight)) / static_cast<int64_t>(strideY) + 1;
        int64_t outputW =
                (inputW - static_cast<int64_t>(filterWidth)) / static_cast<int64_t>(strideX) + 1;

        DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

        outputDim[0] = static_cast<size_t>(outputH);
        outputDim[1] = static_cast<size_t>(outputW);
    } else {
        int64_t outputH =
                (inputH - static_cast<int64_t>(filterHeight) + static_cast<int64_t>(strideY) - 1) /
                static_cast<int64_t>(strideY) + 1;
        int64_t outputW =
                (inputW - static_cast<int64_t>(filterWidth) + static_cast<int64_t>(strideX) - 1) /
                static_cast<int64_t>(strideX) + 1;

        DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

        outputDim[0] = static_cast<size_t>(outputH);
        outputDim[1] = static_cast<size_t>(outputW);
    }

    this->outputShape = Shape(inputShape.batch, outputDim);
}

template <typename T>
void AvgPooling2d<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    typedef typename Eigen::internal::traits<Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    auto input = inputs[0];

    auto batch  = static_cast<TensorIndex>(input->batch());
    auto inputH = static_cast<TensorIndex>(input->shape.dim(0));
    auto inputW = static_cast<TensorIndex>(input->shape.dim(1));
    auto inputC = static_cast<TensorIndex>(input->shape.dim(2));

    auto outputH = static_cast<TensorIndex>(output->shape.dim(0));
    auto outputW = static_cast<TensorIndex>(output->shape.dim(1));
    auto outputC = static_cast<TensorIndex>(output->shape.dim(2));

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            inputTensor(input->data(), batch, inputH, inputW, inputC);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            outputTensor(output->data(), batch, outputH, outputW, outputC);

    Eigen::DSizes<TensorIndex, 4> preDims;
    preDims[0] = batch * outputH * outputW;
    preDims[1] = filterHeight;
    preDims[2] = filterWidth;
    preDims[3] = inputC;

    Eigen::DSizes<TensorIndex, 2> reductionDims;
    reductionDims[0] = 1;
    reductionDims[1] = 2;

    auto padY = std::max<TensorIndex>(0, (outputH - 1) * static_cast<TensorIndex>(strideY) + static_cast<TensorIndex>(filterHeight) - inputH);
    auto padX = std::max<TensorIndex>(0, (outputW - 1) * static_cast<TensorIndex>(strideX) + static_cast<TensorIndex>(filterWidth) - inputW);

    auto padTop    = padY / 2;
    auto padBottom = padY - padTop;
    auto padLeft   = padX / 2;
    auto padRight  = padX - padLeft;

    outputTensor.device(*device) = inputTensor.extract_image_patches(filterWidth, filterHeight,
                                                                     strideX, strideY, 1, 1, 1, 1,
                                                                     padTop, padBottom, padLeft,
                                                                     padRight, 0)
            .reshape(preDims)
            .mean(reductionDims)
            .reshape(outputTensor.dimensions());
}

template <typename T>
void AvgPooling2d<T>::backwardCPUImpl(T *inputGrad,
                     T *outputGrad,
                     int64_t batch,
                     int64_t startChannel,
                     int64_t endChannel,
                     int64_t inputHeight,
                     int64_t inputWidth,
                     int64_t outputHeight,
                     int64_t outputWidth,
                     int64_t channel,
                     int64_t filterH,
                     int64_t filterW,
                     int64_t strideH,
                     int64_t strideW,
                     int64_t padTop,
                     int64_t padLeft) {
    T ratio = T(1) / (T(filterH) * T(filterW));

    for (int64_t b = 0; b < batch; ++b) {
        auto inputGradPtr  = inputGrad  + b * inputHeight  * inputWidth  * channel;
        auto outputGradPtr = outputGrad + b * outputHeight * outputWidth * channel;

        for (int64_t k = startChannel; k < endChannel; ++k) {
            for (int64_t y = 0; y < outputHeight; ++y) {
                for (int64_t x = 0; x < outputWidth; ++x) {
                    auto startH = std::max<int64_t>(0, padTop + y * strideH);
                    auto endH   = std::min<int64_t>(inputHeight, padTop + y * strideH + filterH);

                    auto startW = std::max<int64_t>(0, padLeft + x * strideW);
                    auto endW   = std::min<int64_t>(inputWidth, padLeft + x * strideW + filterW);

                    if (startH >= endH || startW >= endW) {
                        continue;
                    }

                    auto grad = outputGradPtr[y * outputWidth * channel + x * channel + k];

                    for (int64_t inH = startH; inH < endH; ++inH) {
                        for (int64_t inW = startW; inW < endW; ++inW) {
                            inputGradPtr[inH * inputWidth * channel + inW * channel + k] += (ratio * grad);
                        }
                    }
                }
            }
        }
    }
}

#ifdef HAVE_HALF
template <>
void AvgPooling2d<half>::backwardCPUImpl(half *inputGrad,
										half *outputGrad,
										int64_t batch,
										int64_t startChannel,
										int64_t endChannel,
										int64_t inputHeight,
										int64_t inputWidth,
										int64_t outputHeight,
										int64_t outputWidth,
										int64_t channel,
										int64_t filterH,
										int64_t filterW,
										int64_t strideH,
										int64_t strideW,
										int64_t padTop,
										int64_t padLeft) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif


template <typename T>
void AvgPooling2d<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<CPUDevice *>(iGradient->device())->eigenDevice;

    auto batch  = static_cast<int64_t>(iGradient->shape.batch);
    auto inputH = static_cast<int64_t>(iGradient->shape.dim(0));
    auto inputW = static_cast<int64_t>(iGradient->shape.dim(1));
    auto inputC = static_cast<int64_t>(iGradient->shape.dim(2));

    auto outputH = static_cast<int64_t>(outputGradient->shape.dim(0));
    auto outputW = static_cast<int64_t>(outputGradient->shape.dim(1));
    auto outputC = static_cast<int64_t>(outputGradient->shape.dim(2));

    int64_t padY = std::max<int64_t>(0, (outputH - 1) * static_cast<int64_t>(strideY) +
                                        static_cast<int64_t>(filterHeight) - inputH);
    int64_t padX = std::max<int64_t>(0, (outputW - 1) * static_cast<int64_t>(strideX) +
                                        static_cast<int64_t>(filterWidth) - inputW);

    int64_t padTop = -(padY / 2);
    int64_t padLeft = -(padX / 2);

    /**
     * use the Eigen ThreadPool
     */
    int64_t threadNum = device->numThreads();
    int64_t blockSize = (outputC + threadNum - 1) / threadNum;

    Eigen::Barrier barrier(static_cast<unsigned int>(threadNum));

    auto blockFunc = [this, &barrier](T *inputGrad,
                                      T *outputGrad,
                                      int64_t batch,
                                      int64_t startChannel,
                                      int64_t endChannel,
                                      int64_t inputHeight,
                                      int64_t inputWidth,
                                      int64_t outputHeight,
                                      int64_t outputWidth,
                                      int64_t channel,
                                      int64_t filterH,
                                      int64_t filterW,
                                      int64_t strideH,
                                      int64_t strideW,
                                      int64_t padTop,
                                      int64_t padLeft) {
        this->backwardCPUImpl(inputGrad,
                              outputGrad,
                              batch,
                              startChannel,
                              endChannel,
                              inputHeight,
                              inputWidth,
                              outputHeight,
                              outputWidth,
                              channel,
                              filterH,
                              filterW,
                              strideH,
                              strideW,
                              padTop,
                              padLeft);

        barrier.Notify();
    };

    for (int64_t i = 0; i < threadNum; ++i) {
        int64_t startChannel = i * blockSize;
        int64_t endChannel = std::min<int64_t>(startChannel + blockSize, outputC);

        device->enqueueNoNotification(blockFunc,
                                      iGradient->data(),
                                      outputGradient->data(),
                                      batch,
                                      startChannel,
                                      endChannel,
                                      inputH,
                                      inputW,
                                      outputH,
                                      outputW,
                                      outputC,
                                      filterHeight,
                                      filterWidth,
                                      strideY,
                                      strideX,
                                      padTop,
                                      padLeft);
    }

    barrier.Wait();
}

DEEP8_RE_DECLARATION_HALF_FUNC(AvgPooling2d);
DEEP8_DECLARATION_INSTANCE(AvgPooling2d)

}