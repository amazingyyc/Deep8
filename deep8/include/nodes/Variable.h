#ifndef DEEP8_VARIABLE_H
#define DEEP8_VARIABLE_H

#include "Node.h"

namespace Deep8 {

//class VariableBase : public Node {
//protected:
//	explicit VariableBase();
//	explicit VariableBase(Node* input);
//	explicit VariableBase(std::vector<Node*> &inputs);
//
//public:
//	/**
//	 * @brief the Variable do nothing in forward and backward process
//	 */
//	void forward() override;
//	void backward() override;
//
//	/**
//	 * set the Gradient to be 0
//	 */
//	virtual void zeroGradient() = 0;
//
//	/**release the gradient*/
//	virtual void releaseGradient() = 0;
//
//	/**
//	 * get the device type
//	 */
//	virtual DeviceType deviceType() = 0;
//
//    /**
//     * set the gradient be 1 for backward process
//     */
//	virtual void setGradientOne() = 0;
//
//	/**if the Variable is a Scalar*/
//	virtual bool isScalar() = 0;
//
//	/**feed data to value*/
//	virtual void feed(const void *) = 0;
//
//	/**fetch data from value*/
//	virtual void fetch(void *) = 0;
//};

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
class Variable: public Node {
public:
	Tensor value;
	Tensor gradient;

protected:
    explicit Variable();

public:
    explicit Variable(Tensor &v);
    explicit Variable(Tensor &v, Tensor &g);

	explicit Variable(Node *input, Shape &shape);
	explicit Variable(Node *input, Tensor &v);
    explicit Variable(Node *input, Tensor &v, Tensor &g);


public:

	/**
	 * set the Gradient to be 0
	 */
	void zeroGradient();

	/**release the gradient*/
	void releaseGradient();

	/**
	 * get the device type
	 */
	DeviceType deviceType();

	/**
	 * set the gradient be 1 for backward process
	 */
	void setGradientOne();

	/**
	 * if this is a scalar
	 */
	bool isScalar();

	/**feed data to value*/
	void feed(const void *);

	/**fetch data from value*/
	void fetch(void *);

	/**
	 * @brief the Variable do nothing in forward and backward process
	 */
	void forward() override;
	void backward() override;

};

}

#endif //DEEP8_VARIABLE_H
