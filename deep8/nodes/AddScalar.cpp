#include "AddScalar.h"

namespace Deep8 {

template <typename T>
AddScalar<T>::AddScalar(std::vector<Node *> &inputs, T scalar) : Function<T>(inputs), scalar(scalar) {
    check();
}

template <typename T>
void AddScalar<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in AddScalar Function");

    this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void AddScalar<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    eTVec(output).device(*device) = eTVec(inputs[0]) + scalar;
}

template <typename T>
void AddScalar<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<CPUDevice *>(outputGradient->device())->eigenDevice;

    eTVec(iGradient).device(*device) += eTVec(outputGradient);
}

DEEP8_RE_DECLARATION_HALF_FUNC(AddScalar);
DEEP8_DECLARATION_INSTANCE(AddScalar)

}
