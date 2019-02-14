#include "math/Divide.h"
#include "Divide.h"

namespace Deep8 {

Divide::Divide(std::vector<Node *> &inputs) : Function(inputs) {
        check();
}

void Divide::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs size must be 2 in Divide Function");

    /**
     * the Add Function apply to Broadcasting rule: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
     */
    this->outputShape = broadcastShape(this->inputs[0]->outputShape, this->inputs[1]->outputShape);
}

void Divide::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::Divide(*(inputs[0]), *(inputs[1]), *output);
}

void Divide::backward(const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    if (0 == index) {
        Math::DivideGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::DivideGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}



}