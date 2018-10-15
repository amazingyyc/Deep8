#include "Softmax.h"

namespace Deep8 {

template <typename T>
Softmax<T>::Softmax(std::vector<Node *> &inputs): Function<T>(inputs) {
		check();
}

template <typename T>
void Softmax<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Softmax Function needs only 1 input");
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.nDims() >= 2, "the input dimension must be >= 2");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Softmax<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto cpuDevice = static_cast<CPUDevice*>(output->device());
	auto eigenDevice = cpuDevice->eigenDevice;

	auto shape = output->shape;

	auto batch = (int)shape.batch();
	auto size = (int)shape.batchSize();

	auto tempPtr = (T*)cpuDevice->malloc(sizeof(T) * batch);

	Eigen::array<int, 1> reduceDims = { 1 };
	Eigen::array<int, 2> reshape = { batch, 1 };
	Eigen::array<int, 2> broad = { 1, size };

	Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> t(tempPtr, batch, 1);
	Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> x(inputs[0]->data(), batch, size);
	Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> y(output->data(), batch, size);

	t.device(*eigenDevice) = x.maximum(reduceDims).reshape(reshape);
	y.device(*eigenDevice) = (x - t.broadcast(broad)).exp();
	t.device(*eigenDevice) = y.sum(reduceDims).reshape(reshape);
	y.device(*eigenDevice) = y / t.broadcast(broad);

	cpuDevice->free(tempPtr);
}

template <typename T>
void Softmax<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
				const Tensor<T> *output,
				const Tensor<T> *outputGradient,
				size_t index,
				Tensor<T> *iGradient)  {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of Softmax backwardCPU is error");

	auto cpuDevice = static_cast<CPUDevice*>(iGradient->device());
	auto eigenDevice = cpuDevice->eigenDevice;

	auto shape = outputGradient->shape;

	auto batch = (int)shape.batch();
	auto size = (int)shape.batchSize();

	Eigen::array<int, 1> sumDims = { 1 };
	Eigen::array<int, 2> reshape = { batch, 1 };
	Eigen::array<int, 2> broad = { 1, size };

	auto sptr = (T*)cpuDevice->malloc(sizeof(T) * batch);

	Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> s(sptr, batch, 1);
	Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dx(iGradient->data(), batch, size);
	Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> y(output->data(), batch, size);
	Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dy(outputGradient->data(), batch, size);

	s.device(*eigenDevice) = (y * dy).sum(sumDims).reshape(reshape);
	dx.device(*eigenDevice) += (dy - s.broadcast(broad)) * y;

	cpuDevice->free(sptr);
}

DEEP8_RE_DECLARATION_HALF_FUNC(Softmax);
DEEP8_DECLARATION_INSTANCE(Softmax)

}