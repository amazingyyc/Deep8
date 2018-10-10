#include "Parameter.h"

namespace Deep8 {

template <typename T>
void Parameter<T>::check() {
	DEEP8_ARGUMENT_CHECK(this->value.device()->type == this->gradient.device()->type, "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(this->value.shape == this->gradient.shape, "the shape if Value and Gradient must be same");

	this->outputShape = this->value.shape;
}

DEEP8_DECLARATION_INSTANCE(Parameter)

}