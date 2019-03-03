#include "math/LReLu.h"
#include "nodes/LReLu.h"
#include "model/AutoBatchCodeHelper.h"

namespace Deep8 {

LReLu::LReLu(std::vector<Node*> &inputs, float a): Function(inputs), a(a) {
    check();
}

void LReLu::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the LReLu Function needs only 1 input");

    this->shape = this->inputs[0]->shape;
    this->elementType = this->inputs[0]->elementType;
}

int LReLu::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
size_t LReLu::autoBatchCode() {
    AutoBatchCodeHelper helper;

	// todo: aupport half
    helper.functionType(FunctionType::LReLu);
	helper.attachBegin();
	helper.put("a", a);
	helper.attachEnd();

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
Shape LReLu::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
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
std::vector<size_t> LReLu::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
Node* LReLu::autoBatchClone(std::vector<Node*> &inputs) {
	return new LReLu(inputs, a);
}

void LReLu::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::LReLu(*(inputs[0]), a, *output);
}

void LReLu::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	Math::LReLuGrad(*(inputs[0]), *iGradient, a, *output, *outputGradient);
}



}