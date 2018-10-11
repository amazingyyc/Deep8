#include "InputParameter.h"

namespace Deep8 {


/**
 * feed the data into the InputParameter Node
 * the pointer's memory must bigger than the value size
 */
template <typename T>
void InputParameter<T>::feed(const T *ptr) {
	DEEP8_ARGUMENT_CHECK(nullptr != ptr, "the pointer can not be null");

	if (this->value.device()->type == DeviceType::CPU) {
		this->value.device()->copy(ptr, this->value.raw(), sizeof(T) * this->value.size());
	} else {
#ifdef HAVE_CUDA
		value.device()->copyFromCPUToGPU(ptr, value.raw(), sizeof(T) * value.size());
#else
		DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
	}
}

template <typename T>
void InputParameter<T>::zeroGradient() {
}

template <typename T>
bool InputParameter<T>::isScalar() {
	return this->value.isScalar();
}

DEEP8_DECLARATION_INSTANCE(InputParameter)

}