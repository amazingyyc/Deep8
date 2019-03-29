#ifndef DEEP8_VARIABLE_H
#define DEEP8_VARIABLE_H

#include "Node.h"

namespace Deep8 {

class Variable: public Node {
public:
	Tensor value;
	Tensor gradient;

    /**if update the gradient of this Variable*/
    bool updateGradient;

protected:
    explicit Variable(int64_t id, std::string name);

public:
    explicit Variable(int64_t id, std::string name, Tensor &v);
    explicit Variable(int64_t id, std::string name, Tensor &v, Tensor &g);

	explicit Variable(int64_t id, std::string name, Node *input, Tensor &v);
    explicit Variable(int64_t id, std::string name, Node *input, Tensor &v, Tensor &g);

public:

    /**get the Variable shape*/
    Shape shape();

    /**get the element type*/
    ElementType elementType();

    /**get the device type*/
    DeviceType deviceType();
    
    /**if this is a scalar*/
    bool isScalar();

    /**zero value*/
    void zero();
	
    /**set the Gradient to be 0*/
	void zeroGradient();

    /**set value to one*/
    void one();

    /**set gradient to one*/
    void oneGradient();

	/**release the gradient*/
	void removeGradient();

	/**feed data to value*/
	void feed(const void *);

	/**fetch data from value*/
	void fetch(void *);

	/** the Variable do nothing in forward and backward process*/
	void forward() override;
	void backward() override;

};

}

#endif //DEEP8_VARIABLE_H
