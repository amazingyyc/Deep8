#include "L2Norm.h"

namespace Deep8 {

template<typename T>
void L2Norm<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L2Norm Function needs only 1 input");

    this->outputShape = Shape({this->inputs[0]->outputShape.batch(), 1});
}

template<typename T>
void L2Norm<T>::forwardCPU(const std::vector<const Tensor <T> *> &inputs, Tensor <T> *output) {
    auto eigenDevice = static_cast<CPUDevice *>(output->device())->eigenDevice;

    auto input = inputs[0];
    auto batch = input->batch();
    auto size = input->size() / batch;

    Eigen::array<size_t, 2> reshapeDims = {batch, size};
    Eigen::array<size_t, 1> sumDims = {1};

    eTVec(output).device(*eigenDevice) = eTVec(input).square().reshape(reshapeDims).sum(sumDims).sqrt();
}

template<typename T>
void L2Norm<T>::backwardCPU(const std::vector<const Tensor <T> *> &inputs,
                            const Tensor <T> *output,
                            const Tensor <T> *outputGradient,
                            size_t index,
                            Tensor <T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of L2Norm backwardCPU is error");

    auto eigenDevice = static_cast<CPUDevice *>(iGradient->device())->eigenDevice;

    auto batch = iGradient->batch();
    auto size = iGradient->batchSize();

    Eigen::array<size_t, 2> outputDims = {batch, 1};
    Eigen::array<size_t, 2> outputBroad = {1, size};
    Eigen::array<size_t, 2> iGradientDims = {batch, size};

    eTVec(iGradient).reshape(iGradientDims).device(*eigenDevice) +=
            (eTVec(outputGradient).reshape(outputDims) / eTVec(output).reshape(outputDims)).broadcast(outputBroad) * eTVec(inputs[0]).reshape(iGradientDims);
}

DEEP8_DECLARATION_INSTANCE(L2Norm)

}