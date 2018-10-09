#include "Conv2d.h"

namespace Deep8 {

template <typename T>
void Conv2d<T>::check() {
    Function < T > ::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "need 2 inputs node");
    DEEP8_ARGUMENT_CHECK(strideY >= 1 && strideX >= 1, "the stride can not smaller than 1");
    DEEP8_ARGUMENT_CHECK(dilationY >= 1 && dilationX >= 1, "the dilation can not smaller than 1");

    auto inputShape = static_cast<Variable<T> *>(this->inputs[0])->value.shape;
    auto filterShape = static_cast<Variable<T> *>(this->inputs[1])->value.shape;

    DEEP8_ARGUMENT_CHECK(4 == inputShape.nDims() && 4 == filterShape.nDims(),
                         "Conv2d needs inputs nDims is 4");
    DEEP8_ARGUMENT_CHECK(inputShape.dim(3) == filterShape.dim(3), "the inputs dimension is error");
    DEEP8_ARGUMENT_CHECK(filterShape.dim(1) > 0 && filterShape.dim(2) > 0,
                         "the filter must bigger than 0");

    if (!covered) {
        DEEP8_ARGUMENT_CHECK(
                filterShape.dim(1) <= inputShape.dim(1) && filterShape.dim(2) <= inputShape.dim(2),
                "the not forwardCovered mode Padding type needs filter smaller than input");
    }

    auto filterH = static_cast<int64_t>(filterShape.dim(1));
    auto filterW = static_cast<int64_t>(filterShape.dim(2));

    auto inputH = static_cast<int64_t>(inputShape.dim(1));
    auto inputW = static_cast<int64_t>(inputShape.dim(2));

    auto realFilterH = filterH + (filterH - 1) * (static_cast<int64_t>(dilationY) - 1);
    auto realFilterW = filterW + (filterW - 1) * (static_cast<int64_t>(dilationX) - 1);

    std::vector<size_t> outputDim(4);
    outputDim[0] = inputShape.dim(0);
    outputDim[3] = filterShape.dim(0);

/**
 * the input dimension is (batch, inputHeight, inputWidth, inputChannel)
 * filter dimension is (outputChannel, filterHeight, filterWidth, inputChannel)
 * output dimension is (batch, outputHeight, outputWidth, outputChannel)
 */
    if (!covered) {
        int64_t outputH = (inputH - realFilterH) / static_cast<int64_t>(strideY) + 1;
        int64_t outputW = (inputW - realFilterW) / static_cast<int64_t>(strideX) + 1;

        DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

        outputDim[1] = static_cast<size_t>(outputH);
        outputDim[2] = static_cast<size_t>(outputW);
    } else {
        int64_t outputH = (inputH - realFilterH + static_cast<int64_t>(strideY) - 1) /
                          static_cast<int64_t>(strideY) + 1;
        int64_t outputW = (inputW - realFilterW + static_cast<int64_t>(strideX) - 1) /
                          static_cast<int64_t>(strideX) + 1;

        DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

        outputDim[1] = static_cast<size_t>(outputH);
        outputDim[2] = static_cast<size_t>(outputW);
    }

    this->outputShape = Shape(outputDim);
}

template <typename T>
void Conv2d<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    /**
     * the input dimension in Deep8 is NHWC (batch, inputHeight, intputWidth, inputChannel)
     * the filter dimension is NHWC (outputChannel, filterHeight, filterWidth, inputChannel)
     * the output dimension is NHWC (batch, outputHeight, outputWidth, outputChannel)
     */
    typedef typename Eigen::internal::traits<Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    auto input = inputs[0];
    auto filter = inputs[1];

    auto batch = (TensorIndex) input->shape.batch();

    auto inputHeight = (TensorIndex) input->shape.dim(1);
    auto inputWidth = (TensorIndex) input->shape.dim(2);
    auto inputChannel = (TensorIndex) input->shape.dim(3);

    auto outputHeight = (TensorIndex) output->shape.dim(1);
    auto outputWidth = (TensorIndex) output->shape.dim(2);
    auto outputChannel = (TensorIndex) output->shape.dim(3);

    auto filterHeight = (TensorIndex) filter->shape.dim(1);
    auto filterWidth = (TensorIndex) filter->shape.dim(2);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            inputTensor(input->data(), batch, inputHeight, inputWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            filterTensor(filter->data(), outputChannel, filterHeight, filterWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            outputTensor(output->data(), batch, outputHeight, outputWidth, outputChannel);

    Eigen::DSizes<TensorIndex, 2> preContractDims;
    preContractDims[0] = batch * outputHeight * outputWidth;
    preContractDims[1] = filterHeight * filterWidth * inputChannel;

    Eigen::DSizes<TensorIndex, 2> shuffleDims;
    shuffleDims[0] = 1;
    shuffleDims[1] = 0;

    Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contractDims;
    contractDims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

    Eigen::DSizes<TensorIndex, 2> kernelDims;
    kernelDims[0] = outputChannel;
    kernelDims[1] = filterHeight * filterWidth * inputChannel;

    auto realFilterHeight = filterHeight + (filterHeight - 1) * ((TensorIndex) (dilationY) - 1);
    auto realFilterWidth = filterWidth + (filterWidth - 1) * ((TensorIndex) (dilationX) - 1);

    auto padY = std::max<TensorIndex>(0, (outputHeight - 1) * (TensorIndex) (strideY) +
                                         realFilterHeight - inputHeight);
    auto padX = std::max<TensorIndex>(0, (outputWidth - 1) * (TensorIndex) (strideX) +
                                         realFilterWidth - inputWidth);

    auto padTop = padY / 2;
    auto padBottom = padY - padTop;
    auto padLeft = padX / 2;
    auto padRight = padX - padLeft;

    outputTensor.device(*device) = inputTensor.extract_image_patches(filterWidth, filterHeight,
                                                                     strideX, strideY, dilationX,
                                                                     dilationY, 1, 1, padTop,
                                                                     padBottom, padLeft, padRight,
                                                                     0)
            .reshape(preContractDims)
            .contract(filterTensor.reshape(kernelDims).shuffle(shuffleDims), contractDims)
            .reshape(outputTensor.dimensions());
}

template <typename T>
void Conv2d<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

    typedef typename Eigen::internal::traits<Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

    auto device = static_cast<CPUDevice *>(iGradient->device())->eigenDevice;

    auto inputShape = inputs[0]->shape;
    auto filterShape = inputs[1]->shape;
    auto outputShape = output->shape;

    auto batch = (TensorIndex) inputShape.batch();

    auto inputHeight = (TensorIndex) inputShape.dim(1);
    auto inputWidth = (TensorIndex) inputShape.dim(2);
    auto inputChannel = (TensorIndex) inputShape.dim(3);

    auto outputHeight = (TensorIndex) outputShape.dim(1);
    auto outputWidth = (TensorIndex) outputShape.dim(2);
    auto outputChannel = (TensorIndex) outputShape.dim(3);

    auto filterHeight = (TensorIndex) filterShape.dim(1);
    auto filterWidth = (TensorIndex) filterShape.dim(2);

    if (0 == index) {
        /**
         * calculate the grad for input
         */
        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
                outputGradTensor(outputGradient->data(), batch, outputHeight, outputWidth,
                                 outputChannel);

        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
                inputGradTensor(iGradient->data(), batch, inputHeight, inputWidth, inputChannel);

        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
                filterTensor(inputs[1]->data(), outputChannel, filterHeight, filterWidth,
                             inputChannel);

        Eigen::DSizes<TensorIndex, 2> preContractDims;
        preContractDims[0] = batch * inputHeight * inputWidth;
        preContractDims[1] = filterHeight * filterWidth * outputChannel;

        Eigen::internal::conditional<false, Eigen::array<bool, 4>, Eigen::array<bool, 4>>::type filterReverse;
        filterReverse[0] = false;
        filterReverse[1] = true;
        filterReverse[2] = true;
        filterReverse[3] = false;

        Eigen::DSizes<TensorIndex, 4> filterShuffle;
        filterShuffle[0] = 1;
        filterShuffle[1] = 2;
        filterShuffle[2] = 0;
        filterShuffle[3] = 3;

        Eigen::DSizes<TensorIndex, 2> filterDim;
        filterDim[0] = filterHeight * filterWidth * outputChannel;
        filterDim[1] = inputChannel;

        Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contractDims;
        contractDims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

        auto realFilterHeight = filterHeight + (filterHeight - 1) * ((TensorIndex) (dilationY) - 1);
        auto realFilterWidth = filterWidth + (filterWidth - 1) * ((TensorIndex) (dilationX) - 1);

        auto forwardPadTop = std::max<TensorIndex>(0,
                                                   ((outputHeight - 1) * (TensorIndex) (strideY) +
                                                    realFilterHeight - inputHeight) / 2);
        auto forwardPadLeft = std::max<TensorIndex>(0,
                                                    ((outputWidth - 1) * (TensorIndex) (strideX) +
                                                     realFilterWidth - inputWidth) / 2);

        auto padTop = realFilterHeight - 1 - forwardPadTop;
        auto padLeft = realFilterWidth - 1 - forwardPadLeft;
        auto padBottom = inputHeight - (outputHeight - 1) * (TensorIndex) (strideY) - 2 - padTop +
                         realFilterHeight;
        auto padRight = inputWidth - (outputWidth - 1) * (TensorIndex) (strideX) - 2 - padLeft +
                        realFilterWidth;

        inputGradTensor.device(*device) += outputGradTensor.extract_image_patches(filterWidth,
                                                                                  filterHeight, 1,
                                                                                  1, dilationX,
                                                                                  dilationY,
                                                                                  strideX, strideY,
                                                                                  padTop, padBottom,
                                                                                  padLeft, padRight,
                                                                                  0)
                .reshape(preContractDims)
                .contract(filterTensor.reverse(filterReverse).shuffle(filterShuffle).reshape(
                        filterDim), contractDims)
                .reshape(inputGradTensor.dimensions());
    } else if (1 == index) {
        /**
         * calculate the filter gradient
         */
        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
                outputGradTensor(outputGradient->data(), batch, outputHeight, outputWidth,
                                 outputChannel);

        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
                filterGradTensor(iGradient->data(), outputChannel, filterHeight, filterWidth,
                                 inputChannel);

        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
                inputTensor(inputs[0]->data(), batch, inputHeight, inputWidth, inputChannel);

        Eigen::DSizes<TensorIndex, 2> preContractDims;
        preContractDims[0] = batch * outputHeight * outputWidth;
        preContractDims[1] = filterHeight * filterWidth * inputChannel;

        Eigen::DSizes<TensorIndex, 2> outputGradDim;
        outputGradDim[0] = batch * outputHeight * outputWidth;
        outputGradDim[1] = outputChannel;

        Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contractDims;
        contractDims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

        Eigen::DSizes<TensorIndex, 2> shuffleDims;
        shuffleDims[0] = 1;
        shuffleDims[1] = 0;

        TensorIndex realFilterH =
                filterHeight + (filterHeight - 1) * ((TensorIndex) (dilationY) - 1);
        TensorIndex realFilterW = filterWidth + (filterWidth - 1) * ((TensorIndex) (dilationX) - 1);

        TensorIndex padY = std::max<TensorIndex>(0, (outputHeight - 1) * (TensorIndex) (strideY) +
                                                    realFilterH - inputHeight);
        TensorIndex padX = std::max<TensorIndex>(0, (outputWidth - 1) * (TensorIndex) (strideX) +
                                                    realFilterW - inputWidth);

        TensorIndex padTop = padY / 2;
        TensorIndex padBottom = padY - padTop;
        TensorIndex padLeft = padX / 2;
        TensorIndex padRight = padX - padLeft;

        filterGradTensor.device(*device) += outputGradTensor.reshape(outputGradDim).shuffle(shuffleDims)
                .contract(inputTensor.extract_image_patches(filterWidth, filterHeight, strideX,
                                                            strideY, dilationX, dilationY, 1, 1,
                                                            padTop, padBottom, padLeft, padRight,
                                                            0).reshape(preContractDims), contractDims)
                .reshape(filterGradTensor.dimensions());
    }
}

DEEP8_DECLARATION_INSTANCE(Conv2d)

}
