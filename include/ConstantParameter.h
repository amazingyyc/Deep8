#ifndef DEEP8_CONSTANTPARAMETER_H
#define DEEP8_CONSTANTPARAMETER_H

#include "Parameter.h"

namespace Deep8 {

/**
 * this Node will not be change when inited
 */
template <typename T>
class ConstantParameter: public Parameter<T> {
public:
    explicit ConstantParameter(Tensor<T> &value): Parameter<T>(value) {
        this->updateGradient = false;
		this->outputShape = value.shape;
    }

	void zeroGradient() override;
	bool isScalar() override;
};


}

#endif //DEEP8_CONSTANTPARAMETER_H
