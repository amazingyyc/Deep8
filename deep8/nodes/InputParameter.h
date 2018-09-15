#ifndef DEEP8_INPUTPARAMETER_H
#define DEEP8_INPUTPARAMETER_H

#include "Parameter.h"

namespace Deep8 {

/**
 * the InputParameter is just store the user input data, so it does not need update the gradient or trained
 */
template <typename T>
class InputParameter: public Parameter<T> {
public:
    explicit InputParameter(Tensor<T> &value): Parameter<T>(value) {
        this->updateGradient = false;
        this->outputShape    = this->value.shape;
    }

public:

    /**
     * feed the data into the InputParameter Node
     * the pointer's memory must bigger than the value size
     */
    void feed(const T *ptr) {
        DEEP8_ARGUMENT_CHECK(nullptr != ptr, "the pointer can not be null");

        if (this->value.device->type == DeviceType::CPU) {
            this->value.device->copy(ptr, this->value.pointer, sizeof(T) * this->value.size());
        } else {
#ifdef HAVE_CUDA
            static_cast<GPUDevice*>(value.device)->copyFromCPUToGPU(ptr, value.pointer, sizeof(T) * value.size());
#else
            DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
        }
    }

    void zeroGradient() override {
    }

	bool isScalar() override {
		return this->value.isScalar();
	}
};

}

#endif //DEEP8_INPUTPARAMETER_H
