#include "DeConv2d.h"

namespace Deep8 {

template <typename T>
DeConv2d<T>::DeConv2d(std::vector<Node *> &inputs, bool covered, size_t strideY, size_t strideX)
        :Function<T>(inputs), forwardCovered(covered), forwardStrideY(strideY), forwardStrideX(strideX) {
    check();
}

template <typename T>
void DeConv2d<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "need 2 inputs node");
    DEEP8_ARGUMENT_CHECK(forwardStrideY >= 1 && forwardStrideX >= 1, "the stride is error");

    auto inputShape  = this->inputs[0]->outputShape;
    auto filterShape = this->inputs[1]->outputShape;

    DEEP8_ARGUMENT_CHECK(3 == inputShape.nDims, "Conv2d needs inputs nDims is 3");
    DEEP8_ARGUMENT_CHECK(4 == filterShape.nDims && 1 == filterShape.batch, "Conv2d needs filter nDims is 4, the batch must be 1");

    DEEP8_ARGUMENT_CHECK(inputShape.dim(2) == filterShape.dim(3), "the inputs dimension is error");
    DEEP8_ARGUMENT_CHECK(filterShape.dim(1) > 0 && filterShape.dim(2) > 0, "the filter must bigger than 0");

    auto filterH = static_cast<int64_t>(filterShape.dim(1));
    auto filterW = static_cast<int64_t>(filterShape.dim(2));

    auto inputH = static_cast<int64_t>(inputShape.dim(0));
    auto inputW = static_cast<int64_t>(inputShape.dim(1));

    std::vector<size_t> outputDim(3);
    outputDim[2] = filterShape.dim(0);

    /**
     * calculate the output dimension is the reverse of the forward Conv2d
     */
    if (!forwardCovered) {
        int64_t outputH = (inputH - 1) * static_cast<int64_t>(forwardStrideY) + filterH;
        int64_t outputW = (inputW - 1) * static_cast<int64_t>(forwardStrideX) + filterW;

        DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height/width must > 0")

        outputDim[0] = static_cast<size_t>(outputH);
        outputDim[1] = static_cast<size_t>(outputW);
    } else {
        int64_t outputH = (inputH - 1) * static_cast<int64_t>(forwardStrideY) + 1 -
                          static_cast<int64_t>(forwardStrideY) + filterH;
        int64_t outputW = (inputW - 1) * static_cast<int64_t>(forwardStrideX) + 1 -
                          static_cast<int64_t>(forwardStrideX) + filterW;

        DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height/width must > 0")

        outputDim[0] = static_cast<size_t>(outputH);
        outputDim[1] = static_cast<size_t>(outputW);
    }

    this->outputShape = Shape(inputShape.batch, outputDim);
}

template <typename T>
void DeConv2d<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    typedef typename Eigen::internal::traits<Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    auto input = inputs[0];
    auto filter = inputs[1];

    auto batch = (TensorIndex) input->shape.batch;

    auto inputHeight  = (TensorIndex) input->shape.dim(0);
    auto inputWidth   = (TensorIndex) input->shape.dim(1);
    auto inputChannel = (TensorIndex) input->shape.dim(2);

    auto outputHeight  = (TensorIndex) output->shape.dim(0);
    auto outputWidth   = (TensorIndex) output->shape.dim(1);
    auto outputChannel = (TensorIndex) output->shape.dim(2);

    auto filterHeight = (TensorIndex) filter->shape.dim(1);
    auto filterWidth  = (TensorIndex) filter->shape.dim(2);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            inputTensor(input->data(), batch, inputHeight, inputWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            filterTensor(filter->data(), outputChannel, filterHeight, filterWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            outputTensor(output->data(), batch, outputHeight, outputWidth, outputChannel);

    Eigen::DSizes<TensorIndex, 2> preContractDims;
    preContractDims[0] = batch * outputHeight * outputWidth;
    preContractDims[1] = filterHeight * filterWidth * inputChannel;

    Eigen::DSizes<TensorIndex, 2> kernelDims;
    kernelDims[0] = outputChannel;
    kernelDims[1] = filterHeight * filterWidth * inputChannel;

    Eigen::DSizes<TensorIndex, 2> filterShuffle;
    filterShuffle[0] = 1;
    filterShuffle[1] = 0;

    Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contractDims;
    contractDims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

    TensorIndex padH = std::max<TensorIndex>(0, outputHeight + filterHeight -
                                                (inputHeight - 1) * (TensorIndex) (forwardStrideY) -
                                                2);
    TensorIndex padW = std::max<TensorIndex>(0, outputWidth + filterWidth -
                                                (inputWidth - 1) * (TensorIndex) (forwardStrideX) -
                                                2);

    TensorIndex padTop = padH / 2;
    TensorIndex padBottom = padH - padTop;
    TensorIndex padLeft = padW / 2;
    TensorIndex padRight = padW - padLeft;

    outputTensor.device(*device) = inputTensor.extract_image_patches(filterWidth, filterHeight, 1,
                                                                     1, 1, 1, forwardStrideX,
                                                                     forwardStrideY, padTop,
                                                                     padBottom, padLeft, padRight,
                                                                     0)
            .reshape(preContractDims)
            .contract(filterTensor.reshape(kernelDims).shuffle(filterShuffle), contractDims)
            .reshape(outputTensor.dimensions());
}

template <typename T>
void DeConv2d<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    typedef typename Eigen::internal::traits<Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

    auto device = static_cast<CPUDevice *>(iGradient->device())->eigenDevice;

    auto batch = (TensorIndex) inputs[0]->shape.batch;

    auto inputHeight = (TensorIndex) inputs[0]->shape.dim(0);
    auto inputWidth  = (TensorIndex) inputs[0]->shape.dim(1);
    auto inputChannel = (TensorIndex) inputs[0]->shape.dim(2);

    auto outputHeight = (TensorIndex) output->shape.dim(0);
    auto outputWidth  = (TensorIndex) output->shape.dim(1);
    auto outputChannel = (TensorIndex) output->shape.dim(2);

    auto filterHeight = (TensorIndex) inputs[1]->shape.dim(1);
    auto filterWidth  = (TensorIndex) inputs[1]->shape.dim(2);

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

        auto forwardPadTop = std::max<TensorIndex>(0, (outputHeight + filterHeight -
                                                       (inputHeight - 1) *
                                                       (TensorIndex) (forwardStrideY) - 2) / 2);
        auto forwardPadBottom = std::max<TensorIndex>(0, (outputWidth + filterWidth -
                                                          (inputWidth - 1) *
                                                          (TensorIndex) (forwardStrideX) - 2) / 2);

        auto padTop = filterHeight - 1 - forwardPadTop;
        auto padLeft = filterWidth - 1 - forwardPadBottom;
        auto padBottom =
                (inputHeight - 1) * (TensorIndex) (forwardStrideY) + filterHeight - outputHeight -
                padTop;
        auto padRight =
                (inputWidth - 1) * (TensorIndex) (forwardStrideX) + filterWidth - outputWidth -
                padLeft;

        inputGradTensor.device(*device) += outputGradTensor.extract_image_patches(filterWidth,
                                                                                  filterHeight,
                                                                                  forwardStrideX,
                                                                                  forwardStrideY, 1,
                                                                                  1, 1, 1, padTop,
                                                                                  padBottom,
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

        auto padH = std::max<TensorIndex>(0, outputHeight + filterHeight -
                                             (inputHeight - 1) * (TensorIndex) (forwardStrideY) -
                                             2);
        auto padW = std::max<TensorIndex>(0, outputWidth + filterWidth -
                                             (inputWidth - 1) * (TensorIndex) (forwardStrideX) - 2);

        auto padTop = padH / 2;
        auto padBottom = padH - padTop;
        auto padLeft = padW / 2;
        auto padRight = padW - padLeft;

        filterGradTensor.device(*device) +=
                outputGradTensor.reshape(outputGradDim).shuffle(shuffleDims)
                        .contract(inputTensor.extract_image_patches(filterWidth, filterHeight, 1, 1,
                                                                    1, 1, forwardStrideX,
                                                                    forwardStrideY, padTop,
                                                                    padBottom, padLeft, padRight,
                                                                    0).reshape(preContractDims),
                                  contractDims)
                        .reshape(filterGradTensor.dimensions());
    } else {
        DEEP8_RUNTIME_ERROR("the index of DeConv2d backward is error");
    }
}

DEEP8_RE_DECLARATION_HALF_FUNC(DeConv2d);
DEEP8_DECLARATION_INSTANCE(DeConv2d)

}