#include "model/AutoBatchCodeHelper.h"
#include "math/Linear.h"
#include "nodes/Linear.h"

namespace Deep8 {

Linear::Linear(std::vector<Node*> &inputs, float a, float b):Function(inputs), a(a), b(b) {
	check();
}

void Linear::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Linear Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

int Linear::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
size_t Linear::autoBatchCode() {
    AutoBatchCodeHelper helper;

	// todo: aupport half
    helper.functionType(FunctionType::Linear);
	helper.attachBegin();
	helper.put("a", a);
	helper.put("b", a);
	helper.attachEnd();

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
Shape Linear::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
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
std::vector<size_t> Linear::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
Node* Linear::autoBatchClone(std::vector<Node*> &inputs) {
	return new Linear(inputs, a, b);
} 

void Linear::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Linear(*(inputs[0]), a, b, *output);
}

void Linear::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");
	
	Math::LinearGrad(*(inputs[0]), *iGradient, a, b, *output, *outputGradient);
}

}