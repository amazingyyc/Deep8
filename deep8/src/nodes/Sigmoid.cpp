#include "math/Sigmoid.h"
#include "nodes/Sigmoid.h"
#include "model/AutoBatchCodeHelper.h"

namespace Deep8 {

Sigmoid::Sigmoid(std::vector<Node*> &inputs): Function(inputs) {
		check();
}

void Sigmoid::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Sigmoid Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

int Sigmoid::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
size_t Sigmoid::autoBatchCode() {
    AutoBatchCodeHelper helper;

    helper.functionType(FunctionType::Sigmoid);

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
Shape Sigmoid::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
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
std::vector<size_t> Sigmoid::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
Node* Sigmoid::autoBatchClone(std::vector<Node*> &inputs) {
	return new Sigmoid(inputs);
}

void Sigmoid::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Sigmoid(*(inputs[0]), *output);
}

void Sigmoid::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	Math::SigmoidGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}