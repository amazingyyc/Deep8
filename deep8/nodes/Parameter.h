#ifndef DEEP8_PARAMETER_H
#define DEEP8_PARAMETER_H

#include "Variable.h"

namespace Deep8 {

/**
 * the Parameter Node is a special Variable Node that need to be trained
 */
template <typename T>
class Parameter: public Variable<T> {
protected:
    explicit Parameter(): Variable<T>() {
    }

    explicit Parameter(Tensor<T> &value): Variable<T>(value) {
    }

public:
    explicit Parameter(Tensor<T> &value, Tensor<T> &gradient): Variable<T>(value, gradient) {
        check();
    }

protected:
	void check() override;

};

}

#endif //DEEP8_PARAMETER_H
