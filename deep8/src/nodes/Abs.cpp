#include "math/Abs.h"
#include "Abs.h"
#include "AutoBatchCodeHelper.h"

namespace Deep8 {

Abs::Abs(std::vector<Node *> &inputs) : Function(inputs) {
	check();
}

void Abs::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Abs Function needs only 1 input");

    this->outputShape = this->inputs[0]->outputShape;
}

/**
 * for Unary Function it can be auto bateched but default set it to not support auto-batch
 */
int Abs::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
size_t Abs::autoBatchCode() {
    AutoBatchCodeHelper helper;

    helper.functionType(FunctionType::Abs);

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
Shape Abs::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error!");

    /**simple set it to a 1 batch shape*/
    size_t size = 0;

    for (auto item : shapes) {
        size += item.size();
    }

    return Shape({size});
}

/**
 * return the inputs's index that can be auto batched
 */
std::vector<size_t> Abs::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
Node* Abs::autoBatchClone(std::vector<Node*> &inputs) {
    return new Abs(inputs);
} 

void Abs::forward(const std::vector<const Tensor*> &inputs, Tensor *output)  {
    Math::Abs(*(inputs[0]), *output);
}

void Abs::backward( const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Abs backwardCPU is error");

    Math::AbsGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}