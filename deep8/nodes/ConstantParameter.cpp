#include "ConstantParameter.h"

namespace Deep8 {

template <typename T>
void ConstantParameter<T>::zeroGradient() {
}

template <typename T>
bool ConstantParameter<T>::isScalar() {
	return this->value.isScalar();
}

DEEP8_DECLARATION_INSTANCE(ConstantParameter)

}