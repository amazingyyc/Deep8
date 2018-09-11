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
    ~InputParameter() override {
        this->value.free();
    }

    /**
     * feed the data into the InputParameter Node
     * the pointer's memory must bigger than the value size
     */
    void feed(const T *ptr) {
        DEEP8_ARGUMENT_CHECK(nullptr != ptr, "the pointer can not be null");

        if (this->value.device->type == DeviceType::CPU) {
            this->value.device->copy(ptr, this->value.pointer, sizeof(T) * this->value.size());
        } else {
            DEEP8_RUNTIME_ERROR("the feed does not support the GPU for now");
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
