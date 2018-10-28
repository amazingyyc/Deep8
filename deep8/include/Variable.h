#ifndef DEEP8_VARIABLE_H
#define DEEP8_VARIABLE_H

#include "Node.h"

namespace Deep8 {

class VariableBase : public Node {
public:
	/**
	 * @brief if will calculate the Gradient of this Variable in Backward process
	 */
	bool updateGradient;

protected:
	explicit VariableBase();
	explicit VariableBase(bool update);
	explicit VariableBase(std::vector<Node*> &inputs, bool update);

public:
	/**
	 * @brief the Variable do nothing in forward and backward process
	 */
	void forward() override;
	void backward() override;

	/**
	 * set the Gradient to be 0
	 */
	virtual void zeroGradient() = 0;

	/**
	 * get the device type
	 */
	virtual DeviceType deviceType() = 0;

    /**
     * set the gradient be 1 for backward process
     */
	virtual void setGradientOne() = 0;

	/**if the Variable is a Scalar*/
	virtual bool isScalar() = 0;
};

/**
 * @brief the  Variable Node is a Data Node for store the calculate result or the user input data
 * it include 2 property, the Value store the user input or Function result,
 * the gradient store the backward gradient.
 *
 * the Variable have some sub-class like Parameter, InputParameter or ConstantParameter
 * updateGradient means this Variable should be calculate the gradient.
 * some Variable like Function output, weight Parameter need updateGradient be true,
 * but some InputParameter or ConstantParameter needs updateGradient be false.
 *
 * needTrained means this Variable should be trained when after backward, some parameter should be trained like weight Parameter
 * some not like Function output Variable, InputParameter or ConstantParameter
 * some Variable Node can be trained like Parameter but some not like InputParameter ConstantParameter
 */
template <typename T>
class Variable: public VariableBase {
public:
	Tensor<T> value;
	Tensor<T> gradient;

protected:
    explicit Variable();
    explicit Variable(Tensor<T> &v);
    explicit Variable(Tensor<T> &v, Tensor<T> &g);

public:
	explicit Variable(Node *input, Shape &shape);
    explicit Variable(Node *input, Tensor<T> &v, Tensor<T> &g);

protected:
	virtual void check();

public:

	/**
	 * set the Gradient to be 0
	 */
	void zeroGradient() override;

	/**
	 * get the device type
	 */
	DeviceType deviceType() override;

	/**
	 * set the gradient be 1 for backward process
	 */
	void setGradientOne() override;

	bool isScalar() override;

	std::string toString() override;
};

}

#endif //DEEP8_VARIABLE_H
