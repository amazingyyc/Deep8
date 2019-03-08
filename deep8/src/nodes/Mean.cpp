#include "math/Mean.h"
#include "nodes/Mean.h"

namespace Deep8 {

Mean::Mean(std::vector<Node*>& inputs) : Function(inputs) {
    check();
}

void Mean::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L1NormLoss Function needs only 1 input");

    this->shape = Shape(1, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void Mean::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
    Math::Mean(*(inputs[0]), *output);
}

void Mean::backward(const std::vector<const Tensor*>& inputs,
                          const Tensor* output,
                          const Tensor* outputGradient,
                          size_t index,
                          Tensor* iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1NormLoss backward is error");

    Math::MeanGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}