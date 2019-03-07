#include "math/MeanLoss.h"
#include "nodes/MeanLoss.h"

namespace Deep8 {

MeanLoss::MeanLoss(std::vector<Node*>& inputs) : Function(inputs) {
    check();
}

void MeanLoss::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L1NormLoss Function needs only 1 input");

    this->shape = Shape(1, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void MeanLoss::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
    Math::MeanLoss(*(inputs[0]), *output);
}

void MeanLoss::backward(const std::vector<const Tensor*>& inputs,
                          const Tensor* output,
                          const Tensor* outputGradient,
                          size_t index,
                          Tensor* iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1NormLoss backward is error");

    Math::MeanLossGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}