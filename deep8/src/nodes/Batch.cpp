#include "Batch.h"

namespace Deep8 {

template <typename T>
Batch<T>::Batch(std::vector<Node *> &inputs, Shape &outputShape) : Function<T>(inputs), continuous(false) {
	this->outputShape = outputShape;
	check();
}

template <typename T>
void Batch<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(!this->inputs.empty(), "the input can not be empty");

	size_t totalSize = 0;
	for (auto item : this->inputs) {
		totalSize += item->outputShape.size();
	}

	DEEP8_ARGUMENT_CHECK(totalSize == this->outputShape.size(), "the inputs's output shape is error");
}

template <typename T>
void Batch<T>::forward() {
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
		if (false == ((Variable<T>*)item)->updateGradient) {
			allUpdateGradient = false;
			break;
		}
	}

	bool anyUpdateGradient = false;
	for (auto item : this->inputs) {
		if (((Variable<T>*)item)->updateGradient) {
			anyUpdateGradient = true;
			break;
		}
	}

	this->continuous = false;

	if (allUpdateGradient || !anyUpdateGradient) {
		this->continuous = true;

		/**this place have a bug: the GPU memory may be continuous in different storage*/
		for (size_t i = 1; i < this->inputs.size(); ++i) {
			auto preVar  = (Variable<T>*) this->inputs[i - 1];
			auto curVar  = (Variable<T>*) this->inputs[i];

			void *prePtr = (byte*)(preVar->value.raw()) + preVar->value.size() * sizeof(T);

			/**the memory must be in same storage and continuous*/
			if (curVar->value.storage.ptr != preVar->value.storage.ptr ||
				prePtr != ((Variable<T>*) this->inputs[i])->value.raw()) {
				this->continuous = false;
				break;
			}
		}

		if (continuous && allUpdateGradient) {
			/**check if gradient memory is continuous*/
			for (size_t i = 1; i < this->inputs.size(); ++i) {
				auto preVar = (Variable<T>*) this->inputs[i - 1];
				auto curVar = (Variable<T>*) this->inputs[i];

				void *prePtr = (byte*)(preVar->gradient.raw()) + preVar->gradient.size() * sizeof(T);

				if (curVar->gradient.storage.ptr != preVar->gradient.storage.ptr ||
					prePtr != ((Variable<T>*) this->inputs[i])->gradient.raw()) {
					this->continuous = false;
					break;
				}
			}
		}
	}

	if (this->continuous) {
		/**if continuous is true just set the output and inputs point to same memory*/
		auto x = (Variable<T>*)this->inputs[0];
		auto y = (Variable<T>*)this->outputs.first();

		y->outputShape    = this->outputShape;
		y->updateGradient = allUpdateGradient;

		y->value.storage = x->value.storage;
		y->value.offset  = x->value.offset;
		y->value.shape   = this->outputShape;

		if (allUpdateGradient) {
			y->gradient.storage = x->gradient.storage;
			y->gradient.offset  = x->gradient.offset;
			y->gradient.shape   = this->outputShape;
		} else {
			/**release the gradient memory*/
			y->releaseGradient();
		}
	} else {
		/**copy the inputs memory to output*/
		auto outputVar = static_cast<Variable<T>*>(this->outputs.first());

		auto outputValue = &(outputVar->value);
		auto deviceType  = outputVar->deviceType();

		std::vector<const Tensor<T>*> inputValues;

		for (auto item : this->inputs) {
			auto inputVar = static_cast<Variable<T>*>(item);

			DEEP8_ARGUMENT_CHECK(deviceType == inputVar->deviceType(), "the device of input must be same with output");

			inputValues.emplace_back(&(inputVar->value));
		}

		if (DeviceType::CPU == deviceType) {
			this->forwardCPU(inputValues, outputValue);
		} else {
#ifdef HAVE_CUDA
			this->forwardGPU(inputValues, outputValue);
#else
			DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
		}
	}
}

template <typename T>
void Batch<T>::backward() {
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

	auto outputVar = static_cast<Variable<T>*>(this->outputs.first());

	auto outputValue    = &(outputVar->value);
	auto outputGradient = &(outputVar->gradient);

	auto deviceType = outputVar->deviceType();

	std::vector<const Tensor<T>*> inputValues;

	for (auto item : this->inputs) {
		auto inputVar = static_cast<Variable<T>*>(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVar->deviceType(), "the device of the input and output must have same device type");

		inputValues.emplace_back(&(inputVar->value));
	}

	if (DeviceType::CPU == deviceType) {
		for (size_t i = 0; i < this->inputs.size(); ++i) {
			auto inputVar = static_cast<Variable<T>*>(this->inputs[i]);

			if (inputVar->updateGradient) {
				this->backwardCPU(inputValues, outputValue, outputGradient, i, &(inputVar->gradient));
			}
		}
	} else {
		for (size_t i = 0; i < this->inputs.size(); ++i) {
			auto inputVar = static_cast<Variable<T>*>(this->inputs[i]);

			if (inputVar->updateGradient) {
#ifdef HAVE_CUDA
				this->backwardGPU(inputValues, outputValue, outputGradient, i, &(inputVar->gradient));
#else
				DEEP8_RUNTIME_ERROR("not have a GPU");
#endif
			}
		}
	}
}

template <typename T>
void Batch<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice *>(output->device());

	size_t offset = 0;

	for (auto item : inputs) {
		device->copy(item->raw(), (byte*)(output->raw()) + offset, sizeof(T) * item->size());

		offset += sizeof(T) * item->size();
	}
}

template <typename T>
void Batch<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 <= index && index < inputs.size(), "the index is error");

	auto device = static_cast<CPUDevice *>(iGradient->device())->eigenDevice;

	size_t offset = 0;

	for (size_t i = 0; i < index; ++i) {
		offset += sizeof(T) * inputs[i]->size();
	}

	Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dx( (T*)(iGradient->raw()), iGradient->size() );
	Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dy( (T*)((byte*)(outputGradient->raw()) + offset), iGradient->size() );

	dx.device(*device) += dy;
}

DEEP8_RE_DECLARATION_HALF_FUNC(Batch);
DEEP8_DECLARATION_INSTANCE(Batch)

}