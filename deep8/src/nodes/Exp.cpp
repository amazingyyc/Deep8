#include "math/Exp.h"
#include "nodes/Exp.h"

namespace Deep8 {

Exp::Exp(std::vector<Node *> &inputs): Function(inputs) {
    check();
}

void Exp::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Exp Function needs only 1 input");

    this->shape       = this->inputs[0]->shape;
    this->elementType = this->inputs[0]->elementType;
}

void Exp::forward(const std::vector<const Tensor*> &inputs, Tensor *output)  {
    Math::Exp(*(inputs[0]), *output);
}

void Exp::backward( const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Exp backward is error");

    Math::ExpGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}