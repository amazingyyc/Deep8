#include "math/PReLu.h"
#include "nodes/PReLu.h"

namespace Deep8 {

PReLu::PReLu(std::vector<Node*>& inputs) : Function(inputs) {
    check();
}

void PReLu::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the PReLu Function needs 2 input");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape == this->inputs[1]->shape, "the inputs shap must be same");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->elementType == this->inputs[1]->elementType, "the input elementType must be same");

    this->shape       = this->inputs[0]->shape;
    this->elementType = this->inputs[0]->elementType;
}

void PReLu::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
    Math::PReLu(*(inputs[0]), *(inputs[1]), *output);
}

void PReLu::backward(const std::vector<const Tensor*>& inputs, const Tensor* output, const Tensor* outputGradient, size_t index, Tensor* iGradient) {
    if (0 == index) {
        Math::PReLuGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::PReLuGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}

}
