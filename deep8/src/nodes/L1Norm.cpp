#include "L1Norm.h"

namespace Deep8 {

template <typename T>
struct L1NormBackwardExpr {
    inline T operator()(T outputGrad, T input) const {
        if (input > T(0)) {
            return outputGrad;
        } else if (input < T(0)) {
            return -outputGrad;
        } else {
            return 0;
        }
    }
};

template <typename T>
L1Norm<T>::L1Norm(std::vector<Node *> &inputs): Function<T>(inputs) {
    check();
}

template <typename T>
void L1Norm<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L1Norm Function needs only 1 input");

    this->outputShape = Shape({this->inputs[0]->outputShape.batch(), 1});
}

template <typename T>
void L1Norm<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto eigenDevice = static_cast<CPUDevice *>(output->device())->eigenDevice;

    auto input = inputs[0];
    auto batch = input->batch();
    auto size = input->size() / batch;

    Eigen::array<size_t, 2> reshapeDims = {batch, size};
    Eigen::array<size_t, 1> sumDims = {1};

    eTVec(output).device(*eigenDevice) = eTVec(input).abs().reshape(reshapeDims).sum(sumDims);
}

template <typename T>
void L1Norm<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backwardCPU is error");

    auto eigenDevice = static_cast<CPUDevice *>(iGradient->device())->eigenDevice;

    auto batch = iGradient->batch();
    auto size = iGradient->size() / batch;

    Eigen::array<size_t, 2> iGradientDims = {batch, size};
    Eigen::array<size_t, 2> outputGradientDims = {batch, 1};
    Eigen::array<size_t, 2> broadDims = {1, size};

    eTVec(iGradient).reshape(iGradientDims).device(*eigenDevice) +=
            eTVec(outputGradient).reshape(outputGradientDims).broadcast(broadDims).binaryExpr(eTVec(inputs[0]).reshape(iGradientDims), L1NormBackwardExpr<T>());
}

DEEP8_RE_DECLARATION_HALF_FUNC(L1Norm)
DEEP8_DECLARATION_INSTANCE(L1Norm)

}

