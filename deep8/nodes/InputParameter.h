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
	void feed(const T *ptr);

	void zeroGradient() override;

	bool isScalar() override;
};

}

#endif //DEEP8_INPUTPARAMETER_H
