#include "Exp.h"

namespace Deep8 {

template <typename T>
Exp<T>::Exp(std::vector<Node *> &inputs): Function<T>(inputs) {
    check();
}

template <typename T>
void Exp<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Exp Function needs only 1 input");

    this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Exp<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    eTVec(output).device(*device) = eTVec(inputs[0]).exp();
}

template <typename T>
void Exp<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Exp backwardCPU is error");

    auto device = static_cast<CPUDevice *>(outputGradient->device())->eigenDevice;

    eTVec(iGradient).device(*device) += (eTVec(output) * eTVec(outputGradient));
}

DEEP8_RE_DECLARATION_HALF_FUNC(Exp);
DEEP8_DECLARATION_INSTANCE(Exp)

}