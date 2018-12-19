#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.h"
#include "Add.h"

namespace  Deep8 {

template <typename real>
struct AddOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real forward(const real &x, const real &y) {
		return x + y;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real backwardX(const real &x, const real &y, const real &z, const real &dz) {
		return dz;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real backwardY(const real &x, const real &y, const real &z, const real &dz) {
		return dz;
	}
};

template <typename T>
void Add<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	callBinaryElementWiseForward<T, AddOp<T> >(inputs[0]->data(), inputs[0]->shape, inputs[1]->data(), inputs[1]->shape, output->data(), output->shape, AddOp<T>());
}

template <typename T>
void Add<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

	if (0 == index) {
		callBinaryElementWiseBackwardX<T, AddOp<T>>(inputs[0]->data(), iGradient->data(), inputs[0]->shape, inputs[1]->data(), inputs[1]->shape, output->data(), outputGradient->data(), output->shape, AddOp<T>());
	} else if (1 == index ){
		callBinaryElementWiseBackwardY<T, AddOp<T>>(inputs[0]->data(), inputs[0]->shape, inputs[1]->data(), iGradient->data(), inputs[1]->shape, output->data(), outputGradient->data(), output->shape, AddOp<T>());
	} else {
		DEEP8_RUNTIME_ERROR("the index is error");
	}
}

DEEP8_DECLARATION_GPU_FUNC(Add);

}