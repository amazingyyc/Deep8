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

template <typename T>
void Parameter<T>::check() {
}

DEEP8_DECLARATION_INSTANCE(Parameter)

}