#include "SumElements.h"

namespace Deep8 {

template <typename T>
void SumElements<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the SumElements Function needs only 1 input");

	this->outputShape = Shape({ this->inputs[0]->outputShape.batch(), 1 });
}

template <typename T>
void SumElements<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto eigenDevice = static_cast<CPUDevice*>(output->device())->eigenDevice;

	auto input = inputs[0];
	auto batch = input->batch();
	auto size = input->size() / batch;

	Eigen::array<size_t, 2> reshapeDims = { batch, size };
	Eigen::array<size_t, 1> sumDims = { 1 };

	eTVec(output).device(*eigenDevice) = eTVec(input).reshape(reshapeDims).sum(sumDims);
}

template <typename T>
void SumElements<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
								const Tensor<T> *output,
								const Tensor<T> *outputGradient,
								size_t index,
								Tensor<T> *iGradient) {
	if (0 != index) {
		DEEP8_RUNTIME_ERROR("the index of SumElements backwardCPU is error");
	}

	auto eigenDevice = static_cast<CPUDevice*>(iGradient->device())->eigenDevice;

	auto batch = iGradient->batch();
	auto size = iGradient->size() / batch;

	Eigen::array<size_t, 2> iGradientDims = { batch, size };
	Eigen::array<size_t, 2> outputGradientDims = { batch, 1 };
	Eigen::array<size_t, 2> broadDims = { 1, size };

	eTVec(iGradient).reshape(iGradientDims).device(*eigenDevice) += eTVec(outputGradient).reshape(outputGradientDims).broadcast(broadDims);
}

DEEP8_DECLARATION_INSTANCE(SumElements)

}