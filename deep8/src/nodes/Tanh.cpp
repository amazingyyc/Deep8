#include "math/Tanh.h"
#include "nodes/Tanh.h"
#include "model/AutoBatchCodeHelper.h"

namespace Deep8 {

Tanh::Tanh(std::vector<Node *> &inputs) : Function(inputs) {
		check();
}

void Tanh::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Tanh Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

int Tanh::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
size_t Tanh::autoBatchCode() {
    AutoBatchCodeHelper helper;

    helper.functionType(FunctionType::Tanh);

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
Shape Tanh::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error!");

    /**simple set it to a 1 batch shape*/
    size_t size = 0;

    for (auto item : shapes) {
        size += item.size();
    }

    return Shape({ size });
}

/**
 * return the inputs's index that can be auto batched
 */
std::vector<size_t> Tanh::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
Node* Tanh::autoBatchClone(std::vector<Node*> &inputs) {
	return new Tanh(inputs);
}

void Tanh::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Tanh(*(inputs[0]), *output);
}

void Tanh::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	Math::TanhGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}