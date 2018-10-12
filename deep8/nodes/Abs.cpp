#include "Abs.h"

namespace Deep8 {

template <typename T>
struct AbsBackwardExpr {
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
Abs<T>::Abs(std::vector<Node *> &inputs) : Function<T>(inputs) {
	check();
}

template <typename T>
void Abs<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Abs Function needs only 1 input");

    this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Abs<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output)  {
    auto eigenDevice = static_cast<CPUDevice *>(output->device())->eigenDevice;

    eTVec(output).device(*eigenDevice) = eTVec(inputs[0]).abs();
}

template <typename T>
void Abs<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Abs backwardCPU is error");

    auto eigenDevice = static_cast<CPUDevice *>(output->device())->eigenDevice;

    eTVec(iGradient).device(*eigenDevice) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), AbsBackwardExpr<T>());
}

DEEP8_RE_DECLARATION_HALF_FUNC(Abs)
DEEP8_DECLARATION_INSTANCE(Abs);

}