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
    explicit Parameter();
    explicit Parameter(Tensor<T> &value);

public:
    explicit Parameter(Tensor<T> &value, Tensor<T> &gradient);

protected:
	void check() override;

};

}

#endif //DEEP8_PARAMETER_H
