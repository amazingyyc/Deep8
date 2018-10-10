#include "Multiply.h"

namespace Deep8 {

template <typename T>
void Multiply<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs dim must be 2 in Multiply Function");

	/**
	 * the Add Function apply to Broadcasting rule: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
	 */
	auto xShape = static_cast<Variable<T>*>(this->inputs[0])->value.shape;
	auto yShape = static_cast<Variable<T>*>(this->inputs[1])->value.shape;

	this->outputShape = broadcastShape(xShape, yShape);
}

template <typename T>
void Multiply<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

	auto xShape = inputs[0]->shape;
	auto yShape = inputs[1]->shape;

	auto zShape = output->shape;

	if (zShape == xShape && zShape == yShape) {
		eTVec(output).device(*device) = eTVec(inputs[0]) * eTVec(inputs[1]);
	} else {
		auto xReshape = enlongateShapeToMaxDim(xShape);
		auto yReshape = enlongateShapeToMaxDim(yShape);
		auto zReshape = enlongateShapeToMaxDim(zShape);

		auto xBroad = xReshape;
		auto yBroad = yReshape;

		for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
			if (xBroad[i] < zReshape[i]) {
				xBroad[i] = zReshape[i];
			} else {
				xBroad[i] = 1;
			}

			if (yBroad[i] < zReshape[i]) {
				yBroad[i] = zReshape[i];
			} else {
				yBroad[i] = 1;
			}
		}

		eTVec(output).reshape(zReshape).device(*device) = eTVec(inputs[0]).reshape(xReshape).broadcast(xBroad)
			* eTVec(inputs[1]).reshape(yReshape).broadcast(yBroad);
	}
}

template <typename T>
template <int diffCount>
void Multiply<T>::backwardCPUImpl(Eigen::ThreadPoolDevice *device,
								const Tensor<T> *otherValue,
								const Tensor<T> *outputGradient,
								Tensor<T> *iGradient) {
	auto curElongateDims = enlongateShapeToMaxDim(iGradient->shape);
	auto otherElongateDims = enlongateShapeToMaxDim(otherValue->shape);
	auto outputElongateDims = enlongateShapeToMaxDim(outputGradient->shape);

	Eigen::array<int, diffCount> sumDims;

	for (int i = 0, j = 0; i < MAX_TENSOR_DIMS; ++i) {
		if (curElongateDims[i] != outputElongateDims[i]) {
			sumDims[j++] = i;
		}
	}

	auto otherBroad = otherElongateDims;

	for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
		if (otherBroad[i] < outputElongateDims[i]) {
			otherBroad[i] = outputElongateDims[i];
		} else {
			otherBroad[i] = 1;
		}
	}

	eTVec(iGradient).reshape(curElongateDims).device(*device) +=
		((eTVec(outputGradient).reshape(outputElongateDims)) * (eTVec(*otherValue).reshape(otherElongateDims).broadcast(otherBroad))).sum(sumDims).reshape(curElongateDims);
}

template <typename T>
void Multiply<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
							const Tensor<T> *output,
							const Tensor<T> *outputGradient,
							size_t index,
							Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

	auto curValue = inputs[(0 == index) ? 0 : 1];
	auto otherValue = inputs[(0 == index) ? 1 : 0];

	auto curShape = curValue->shape;
	auto otherShape = otherValue->shape;
	auto outputShape = outputGradient->shape;

	if (outputShape == curShape && outputShape == otherShape) {
		eTVec(iGradient).device(*device) += eTVec(outputGradient) * eTVec(*otherValue);
		return;
	}

	auto curElongateDims = enlongateShapeToMaxDim(curShape);
	auto outputElongateDims = enlongateShapeToMaxDim(outputShape);

	int diffCount = 0;

	for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
		if (curElongateDims[i] != outputElongateDims[i]) {
			diffCount++;
		}
	}

	if (0 == diffCount) {
		backwardCPUImpl<0>(device, otherValue, outputGradient, iGradient);
	} else if (1 == diffCount) {
		backwardCPUImpl<1>(device, otherValue, outputGradient, iGradient);
	} else if (2 == diffCount) {
		backwardCPUImpl<2>(device, otherValue, outputGradient, iGradient);
	} else if (3 == diffCount) {
		backwardCPUImpl<3>(device, otherValue, outputGradient, iGradient);
	} else if (4 == diffCount) {
		backwardCPUImpl<4>(device, otherValue, outputGradient, iGradient);
	} else {
		DEEP8_RUNTIME_ERROR("the shape is error");
	}
}

DEEP8_DECLARATION_INSTANCE(Multiply)

}