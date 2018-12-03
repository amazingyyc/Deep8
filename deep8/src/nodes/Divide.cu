#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.cuh"
#include "Divide.h"

namespace Deep8 {

template <typename real>
struct DivideOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real forward(const real &x, const real &y) {
		return x / y;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real backwardX(const real &x, const real &y, const real &z, const real &dz) {
		return dz / y;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real backwardY(const real &x, const real &y, const real &z, const real &dz) {
		return -x * dz / (y * y);
	}
};

template <typename T>
void Divide<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	callBinaryElementWiseForward<T, DivideOp<T> >(inputs[0]->data(), inputs[0]->shape, inputs[1]->data(), inputs[1]->shape, output->data(), output->shape, DivideOp<T>());
}

template <typename T>
void Divide<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                         const Tensor<T> *output,
                         const Tensor<T> *outputGradient,
                         size_t index,
                         Tensor<T> *iGradient) {
    if (0 == index) {
		callBinaryElementWiseBackwardX<T, DivideOp<T>>(
			inputs[0]->data(), 
			iGradient->data(), 
			inputs[0]->shape, 
			inputs[1]->data(), 
			inputs[1]->shape, 
			output->data(), 
			outputGradient->data(), 
			output->shape, 
			DivideOp<T>());
	} else if (1 == index) {
		callBinaryElementWiseBackwardY<T, DivideOp<T>>(
			inputs[0]->data(), 
			inputs[0]->shape, 
			inputs[1]->data(), 
			iGradient->data(), 
			inputs[1]->shape, 
			output->data(), 
			outputGradient->data(), 
			output->shape, 
			DivideOp<T>());
	} else {
		DEEP8_RUNTIME_ERROR("the index is error");
	}
}

DEEP8_DECLARATION_GPU_FUNC(Divide);

}