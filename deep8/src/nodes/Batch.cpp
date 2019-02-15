#include "math/Add.h"
#include "nodes/Batch.h"

namespace Deep8 {

Batch::Batch(std::vector<Node *> &inputs, Shape &shape) : Function(inputs), continuous(false) {
	this->outputShape = shape;
	check();
}

void Batch::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(!this->inputs.empty(), "the input can not be empty");

	size_t totalSize = 0;
	for (auto item : this->inputs) {
		totalSize += item->outputShape.size();
	}

	DEEP8_ARGUMENT_CHECK(totalSize == this->outputShape.size(), "the inputs's output shape is error");
}

void Batch::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	auto device = output->device();

	size_t offset = 0;

	for (auto item : inputs) {
		device->copy(item->raw(), (byte*)(output->raw()) + offset, item->byteCount());

		offset += item->byteCount();
	}
}

void Batch::backward(const std::vector<const Tensor*> &inputs, const Tensor *output, const Tensor *outputGradient, size_t index, Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 <= index && index < inputs.size(), "the index is error");
	
	size_t size   = iGradient->size();
	size_t offset = 0;

	for (size_t i = 0; i < index; ++i) {
		offset += inputs[i]->byteCount();
	}

	Tensor dx;
	Tensor dy;

	dx.storage = iGradient->storage;
	dx.offset  = iGradient->offset;
	dx.shape   = Shape(1, {size});
	dx.type    = iGradient->type;

	dy.storage = outputGradient->storage;
	dy.offset  = outputGradient->offset + offset;
	dy.shape   = Shape(1, {size});
	dy.type    = outputGradient->type;

	Math::Add(dy, dx, dx);
}

void Batch::forward() {
	for (auto item : this->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be Variable type");
	}

	DEEP8_ARGUMENT_CHECK(1 == this->outputs.size(), "the outputs size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be Variable type");
	DEEP8_ARGUMENT_CHECK(this->outputShape == this->outputs.first()->outputShape, "the output shape is error");

	/**
	 * 2 condition the continuous is true
	 * 1: all inputs's updateGradient is false, the the inputs value's memory is continuous
	 * 2: all inputs's updateGradient is true, the inpus value and gradient's memory is continuous
	 */
	bool allUpdateGradient = true;
	for (auto item : this->inputs) {
		if (false == ((Variable*)item)->updateGradient) {
			allUpdateGradient = false;
			break;
		}
	}

	bool anyUpdateGradient = false;
	for (auto item : this->inputs) {
		if (((Variable*)item)->updateGradient) {
			anyUpdateGradient = true;
			break;
		}
	}

	this->continuous = false;

	if (allUpdateGradient || !anyUpdateGradient) {
		this->continuous = true;

		/**check if value is continuous*/
		for (size_t i = 1; i < this->inputs.size(); ++i) {
			auto preVar  = (Variable*) this->inputs[i - 1];
			auto curVar  = (Variable*) this->inputs[i];

			void *prePtr = (byte*)(preVar->value.raw()) + preVar->value.byteCount();

			/**the memory must be in same storage and continuous*/
			if (preVar->value.storage.ptr != curVar->value.storage.ptr ||
				prePtr != curVar->value.raw()) {
				this->continuous = false;
				break;
			}
		}

		if (continuous && allUpdateGradient) {
			/**check if gradient memory is continuous*/
			for (size_t i = 1; i < this->inputs.size(); ++i) {
				auto preVar = (Variable*) this->inputs[i - 1];
				auto curVar = (Variable*) this->inputs[i];

				void *prePtr = (byte*)(preVar->gradient.raw()) + preVar->gradient.byteCount();

				if (preVar->gradient.storage.ptr != curVar->gradient.storage.ptr ||
					prePtr != curVar->gradient.raw()) {
					this->continuous = false;
					break;
				}
			}
		}
	}

	if (this->continuous) {
		/**if continuous is true just set the output and inputs point to same memory*/
		auto x = (Variable*)this->inputs[0];
		auto y = (Variable*)this->outputs.first();

		y->outputShape    = this->outputShape;
		y->updateGradient = allUpdateGradient;

		y->value.storage = x->value.storage;
		y->value.offset  = x->value.offset;
		y->value.type    = x->value.type;
		y->value.shape   = this->outputShape;

		if (allUpdateGradient) {
			y->gradient.storage = x->gradient.storage;
			y->gradient.offset  = x->gradient.offset;
			y->gradient.type    = x->gradient.type;
			y->gradient.shape   = this->outputShape;
		} else {
			/**release the gradient memory*/
			y->releaseGradient();
		}
	} else {
		/**copy the inputs memory to output*/
		auto outputVariable = (Variable*)(this->outputs.first());

		auto outputValue = &(outputVariable->value);
		auto deviceType  = outputVariable->deviceType();

		std::vector<const Tensor*> inputValues;

		for (auto item : this->inputs) {
			auto inputVariable = (Variable*)(item);

			DEEP8_ARGUMENT_CHECK(deviceType == inputVariable->deviceType(), "the device of input must be same with output");

			inputValues.emplace_back(&(inputVariable->value));
		}

		/**copy memory*/
		this->forward(inputValues, outputValue);
	}
}

void Batch::backward() {
	if (this->continuous) {
		/**do nothing*/
		return;
	}

	/**the inputs and outputs must be Variable*/
	for (auto item : this->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be a Variable type");
	}

	DEEP8_ARGUMENT_CHECK(1 == this->outputs.size(), "the output size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be Variable type");

	auto outputVariable = (Variable*)(this->outputs.first());

	auto outputValue    = &(outputVariable->value);
	auto outputGradient = &(outputVariable->gradient);

	auto deviceType = outputVariable->deviceType();

	std::vector<const Tensor*> inputValues;

	for (auto item : this->inputs) {
		auto inputVariable = (Variable*)(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVariable->deviceType(), "the device of the input and output must have same device type");

		inputValues.emplace_back(&(inputVariable->value));
	}

	for (size_t i = 0; i < this->inputs.size(); ++i) {
		auto inputVariable = (Variable*)(this->inputs[i]);

		if (inputVariable->updateGradient) {
			this->backward(inputValues, outputValue, outputGradient, i, &(inputVariable->gradient));
		}
	}
}




}