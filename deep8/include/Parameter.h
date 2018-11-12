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
    
public:
	explicit Parameter(Tensor<T> &value);
    explicit Parameter(Tensor<T> &value, Tensor<T> &gradient);

	/**
	 * feed the data into the InputParameter Node
	 * the pointer's memory must bigger than the value size
	 */
	void feed(const void *ptr);

protected:
	void check() override;
};

}

#endif //DEEP8_PARAMETER_H
