#include "DivideScalar.h"

namespace Deep8 {

template <typename T>
DivideScalar<T>::DivideScalar(std::vector<Node *> &inputs, T scalar) : Function<T>(inputs), scalar(scalar) {
    check();
}

template <typename T>
void DivideScalar<T>::check()  {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in DivideScalar Function");
    DEEP8_ARGUMENT_CHECK(0 != scalar, "the divide scalar can no be 0");

    this->outputShape = this->inputs[0]->outputShape;
}

#ifdef HAVE_HALF
template <>
void DivideScalar<half>::check()  {
    Function<half>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in DivideScalar Function");
    DEEP8_ARGUMENT_CHECK(0 != __half2float(scalar), "the divide scalar can no be 0");

    this->outputShape = this->inputs[0]->outputShape;
}
#endif

template <typename T>
void DivideScalar<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output)  {
    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    eTVec(output).device(*device) = eTVec(inputs[0]) / scalar;
}

template <typename T>
void DivideScalar<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<CPUDevice *>(outputGradient->device())->eigenDevice;

    eTVec(iGradient).device(*device) += eTVec(outputGradient) / scalar;
}

DEEP8_RE_DECLARATION_HALF_FUNC(DivideScalar);
DEEP8_DECLARATION_INSTANCE(DivideScalar)

}