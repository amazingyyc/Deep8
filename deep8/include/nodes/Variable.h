#ifndef DEEP8_VARIABLE_H
#define DEEP8_VARIABLE_H

#include "model/Executor.h"
#include "nodes/Node.h"

namespace Deep8 {

class Variable: public Node {
public:
	Tensor value;
	Tensor gradient;

    /**if update the gradient of this Variable*/
    bool updateGradient;

protected:
    explicit Variable(int64_t id, std::string name, Executor *exe);

public:
    explicit Variable(int64_t id, std::string name, Executor *exe, Tensor &v);
    explicit Variable(int64_t id, std::string name, Executor *exe, Tensor &v, Tensor &g);

    explicit Variable(int64_t id, std::string name, Executor *exe, Node *input);
	explicit Variable(int64_t id, std::string name, Executor *exe, Node *input, Tensor &v);
    explicit Variable(int64_t id, std::string name, Executor *exe, Node *input, Tensor &v, Tensor &g);

public:

    /**get the Variable shape*/
    Shape shape();

    /**get the element type*/
    ElementType elementType();

    /**get the device type*/
    DeviceType deviceType();
    
    /**if this is a scalar*/
    bool isScalar();

    /**set the Gradient to be 0*/
	void zeroGradient();

    /**set gradient to one*/
    void oneGradient();

	/**release the gradient*/
	void removeGradient();

	/** the Variable do nothing in forward and backward process*/
	void forward() override;
	void backward() override;

    /****************************************************************************/
    /**function for build computer graph*/
    /****************************************************************************/
    
    /**return a string for print value*/
    std::string valueStr();

    /**feed data to value from CPU memory*/
    Variable& feed(const void*);

    /**copy memory from value to CPU memory*/
    Variable& fetch(void*);

	Variable& constant(float scalar = 0);
	Variable& zero();
	Variable& one();
	Variable& gaussian(float mean = 0.0, float stddev = 0.01);
	Variable& positiveUnitball();
	Variable& random(float lower = 0.0, float upper = 1.0);
	Variable& uniform(float left = 0.0, float right = 1.0);

	Variable& assign(Variable& v);
};

}

#endif //DEEP8_VARIABLE_H
