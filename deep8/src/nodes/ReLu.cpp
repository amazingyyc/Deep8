#include "math/ReLu.h"
#include "nodes/ReLu.h"
#include "model/AutoBatchCodeHelper.h"

namespace Deep8 {

ReLu::ReLu(std::vector<Node *> &inputs) : Function(inputs) {
	check();
}

void ReLu::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the ReLu Function needs only 1 input");

	/**the ReLu output shape equal the input*/
	this->outputShape = this->inputs[0]->outputShape;
}

int ReLu::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
size_t ReLu::autoBatchCode() {
    AutoBatchCodeHelper helper;

    helper.functionType(FunctionType::ReLu);

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
Shape ReLu::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
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
std::vector<size_t> ReLu::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
Node* ReLu::autoBatchClone(std::vector<Node*> &inputs) {
	return new ReLu(inputs);
}

void ReLu::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::ReLu(*(inputs[0]), *output);
}

void ReLu::backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) {
	Math::ReLuGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}