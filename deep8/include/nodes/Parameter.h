#ifndef DEEP8_PARAMETER_H
#define DEEP8_PARAMETER_H

#include "Variable.h"

namespace Deep8 {

/**
 * the Parameter Node is a special Variable Node that need to be trained
 */
class Parameter: public Variable {
protected:
    explicit Parameter();
    
public:
	explicit Parameter(Tensor &value);
    explicit Parameter(Tensor &value, Tensor &gradient);
	
protected:
	void check() override;
};

}

#endif
