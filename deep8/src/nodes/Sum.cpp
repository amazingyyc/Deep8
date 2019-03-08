#include "math/Sum.h"
#include "nodes/Sum.h"

namespace Deep8 {

Sum::Sum(std::vector<Node*>& inputs) : Function(inputs) {
    check();
}

void Sum::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L1NormLoss Function needs only 1 input");

    this->shape = Shape(1, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void Sum::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
    Math::Sum(*(inputs[0]), *output);
}

void Sum::backward(const std::vector<const Tensor*>& inputs,
                          const Tensor* output,
                          const Tensor* outputGradient,
                          size_t index,
                          Tensor* iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1NormLoss backward is error");

    Math::SumGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}