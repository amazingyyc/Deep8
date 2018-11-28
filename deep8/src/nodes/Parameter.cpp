#include "Parameter.h"

namespace Deep8 {

template <typename T>
Parameter<T>::Parameter(): Variable<T>() {
}

template <typename T>
Parameter<T>::Parameter(Tensor<T> &value): Variable<T>(value) {
}

template <typename T>
Parameter<T>::Parameter(Tensor<T> &value, Tensor<T> &gradient): Variable<T>(value, gradient) {
}

/**
 * feed the data into the InputParameter Node
 * the pointer's memory must bigger than the value size
 */
template <typename T>
void Parameter<T>::feed(const void *ptr) {
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
void Parameter<T>::check() {
}

DEEP8_DECLARATION_INSTANCE(Parameter)

}