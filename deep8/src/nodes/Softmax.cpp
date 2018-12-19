#include "Softmax.h"

namespace Deep8 {

template <typename T>
Softmax<T>::Softmax(std::vector<Node *> &inputs, size_t a): Function<T>(inputs), axis(a) {
	check();
}

template <typename T>
void Softmax<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Softmax Function needs only 1 input");

	auto inputShape = this->inputs[0]->outputShape;

	DEEP8_ARGUMENT_CHECK(0 <= axis && axis < inputShape.nDims, "the axis is error");

	std::vector<size_t> list;
	for (int i = 0; i < inputShape.nDims; ++i) {
		if (i != axis) {
			list.emplace_back(inputShape.dim(i));
		} else {
			list.emplace_back(1);
		}
	}

	this->outputShape = Shape(inputShape.batch, list);
}

template <typename T>
void Softmax<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto cpuDevice = static_cast<CPUDevice *>(output->device());
    auto eigenDevice = cpuDevice->eigenDevice;

    auto xshape = inputs[0]->shape;

    int dim0 = xshape.batch;
    int dim1 = xshape.dim(axis);
    int dim2 = 1;

    for (int i = 0; i < axis; ++i) {
      dim0 *= xshape.dim(i);
    }

    for (int i = axis + 1; i < xshape.nDims; ++i) {
      dim2 *= xshape.dim(i);
    }

    auto tempptr = (T*)cpuDevice->malloc(sizeof(T) * dim0 * dim2);

    Eigen::array<int, 1> reduceDims = { 1 };
    Eigen::array<int, 3> reshape    = { dim0, 1, dim2 };
    Eigen::array<int, 3> broad      = { 1, dim1, 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> x(inputs[0]->data(), dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> y(output->data(), dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> t(tempptr, dim0, 1, dim2);

    t.device(*eigenDevice) = x.maximum(reduceDims).reshape(reshape);
    y.device(*eigenDevice) = (x - t.broadcast(broad)).exp();
    t.device(*eigenDevice) = y.sum(reduceDims).reshape(reshape);
    y.device(*eigenDevice) = y / t.broadcast(broad);

    cpuDevice->free(tempptr);



	/*auto cpuDevice = static_cast<CPUDevice*>(output->device());
	auto eigenDevice = cpuDevice->eigenDevice;

	auto shape = output->shape;

	auto batch = (int)shape.batch;
	auto size  = (int)shape.batchSize();

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

	cpuDevice->free(tempPtr);*/
}

template <typename T>
void Softmax<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Softmax backwardCPU is error");

    auto cpuDevice = static_cast<CPUDevice*>(iGradient->device());
    auto eigenDevice = cpuDevice->eigenDevice;

    auto shape = iGradient->shape;

    int dim0 = shape.batch;
    int dim1 = shape.dim(axis);
    int dim2 = 1;

    for (int i = 0; i < axis; ++i) {
        dim0 *= shape.dim(i);
    }

    for (int i = axis + 1; i < shape.nDims; ++i) {
        dim2 *= shape.dim(i);
    }

    auto tempptr = (T*)cpuDevice->malloc(sizeof(T) * dim0 * dim2);

    Eigen::array<int, 1> sumDims = { 1 };
    Eigen::array<int, 3> reshape = { dim0, 1, dim2 };
    Eigen::array<int, 3> broad   = { 1, dim1, 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> t(tempptr, dim0, 1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dx(iGradient->data(),      dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> y(output->data(),          dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dy(outputGradient->data(), dim0, dim1, dim2);

    t.device(*eigenDevice) = (y * dy).sum(sumDims).reshape(reshape);
    dx.device(*eigenDevice) += (dy - t.broadcast(broad)) * y;

    cpuDevice->free(tempptr);

	/*DEEP8_ARGUMENT_CHECK(0 == index, "the index of Softmax backwardCPU is error");

	auto cpuDevice = static_cast<CPUDevice*>(iGradient->device());
	auto eigenDevice = cpuDevice->eigenDevice;

	auto shape = outputGradient->shape;

	auto batch = (int)shape.batch;
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

	cpuDevice->free(sptr);*/
}

DEEP8_RE_DECLARATION_HALF_FUNC(Softmax);
DEEP8_DECLARATION_INSTANCE(Softmax)

}