#include "math/CrossEntropy.h"
#include "nodes/CrossEntropy.h"

namespace Deep8 {

CrossEntropy::CrossEntropy(std::vector<Node *> &inputs) : Function(inputs) {
}

void CrossEntropy::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the CrossEntropy Function needs only 2 input");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->elementType == this->inputs[1]->elementType, "the inputs elementtype must be same");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape.size() == this->inputs[1]->shape.size(), "inputs's shape size must be equal");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape.batch == this->inputs[1]->shape.batch, "inputs's batch must be equal");

    this->shape       = Shape(1, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void CrossEntropy::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::CrossEntropy(*(inputs[0]), *(inputs[1]), *output);
}

void CrossEntropy::backward(const std::vector<const Tensor*> &inputs, 
                            const Tensor *output, 
                            const Tensor *outputGradient, 
                            size_t index, 
                            Tensor *iGradient) {
    if (0 == index) {
        Math::CrossEntropyGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::CrossEntropyGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}



}