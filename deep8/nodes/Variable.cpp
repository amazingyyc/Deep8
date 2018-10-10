#include "Variable.h"

namespace Deep8 {

template <typename T>
void Variable<T>::check() {
	DEEP8_ARGUMENT_CHECK(1 == inputs.size(), "the Variable Node must need 1 input");

	DEEP8_ARGUMENT_CHECK(value.device()->type == gradient.device()->type, "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(value.shape == gradient.shape, "the shape if Value and Gradient must be same");

	for (auto i : inputs) {
		DEEP8_ARGUMENT_CHECK(nullptr != i, "the input can not be null");
		DEEP8_ARGUMENT_CHECK(i->outputShape == value.shape, "the shape of the input, pointer and gradient must be same")
	}

	outputShape = value.shape;
}

/**
 * set the Gradient to be 0
 */
template <typename T>
void Variable<T>::zeroGradient() {
	gradient.zero();
}

/**
 * get the device type
 */
template <typename T>
DeviceType Variable<T>::deviceType() {
	return value.device()->type;
}

/**
 * set the gradient be 1 for backward process
 */
template <typename T>
void Variable<T>::setGradientOne() {
	DEEP8_ARGUMENT_CHECK(gradient.isScalar(), "the gradient is  not scalar");

	if (DeviceType::CPU == gradient.device()->type) {
		gradient.data()[0] = T(1);
	} else {
#ifdef HAVE_CUDA
		auto device = static_cast<GPUDevice*>(gradient.device());

		device->copyFromGPUToGPU(device->gpuOne<T>(), gradient.raw(), sizeof(T));
#else
		DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
	}
}

template <typename T>
bool Variable<T>::isScalar() {
	return value.isScalar() && gradient.isScalar();
}

DEEP8_DECLARATION_INSTANCE(Variable)

}