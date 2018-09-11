#ifndef DEEP8_VARIABLE_H
#define DEEP8_VARIABLE_H

#include "Tensor.h"
#include "Device.h"
#include "Node.h"

namespace Deep8 {

class VariableBase : public Node {
public:
	/**
	 * @brief if will calculate the Gradient of this Variable in Backward process
	 */
	bool updateGradient;

protected:
	explicit VariableBase(): updateGradient(false) {
		this->type = NodeType::Variable;
	}

	explicit VariableBase(bool update) : updateGradient(update) {
		this->type = NodeType::Variable;
	}

	explicit VariableBase(std::vector<Node*> &inputs, bool update) : Node(inputs), updateGradient(update) {
		this->type = NodeType::Variable;
	}
public:
	/**
	 * @brief the Variable do nothing in forward and backward process
	 */
	void forward() override {

	}

	void backward() override {

	}

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
    explicit Variable(): VariableBase(false) {
    }

    explicit Variable(Tensor<T> &v): value(v), VariableBase(false) {
    }

    explicit Variable(Tensor<T> &v, Tensor<T> &g): value(v), gradient(g), VariableBase(true) {
    }

public:
    explicit Variable(Node *input, Tensor<T> &v, Tensor<T> &g): value(v), gradient(g), VariableBase(true) {
        this->inputs.emplace_back(input);

        check();
    }

    explicit Variable(std::vector<Node*> &inputs, Tensor<T> &v, Tensor<T> &g): value(v), gradient(g), VariableBase(inputs, true) {
        check();
    }

protected:
    virtual void check() {
        DEEP8_ARGUMENT_CHECK(1 == inputs.size(), "the Variable Node must need 1 input");

		DEEP8_ARGUMENT_CHECK(value.device->type == gradient.device->type, "the values and gradient must be the same type");
		DEEP8_ARGUMENT_CHECK(value.shape == gradient.shape, "the shape if Value and Gradient must be same");
        
        for (auto i : inputs) {
            DEEP8_ARGUMENT_CHECK(nullptr != i, "the input can not be null");
            DEEP8_ARGUMENT_CHECK(i->outputShape == value.shape, "the shape of the input, pointer and gradient must be same")
        }

        outputShape = value.shape;
    }

public:
	~Variable() {
    	if (!this->shared) {
    		value.free();
    		gradient.free();
    	}
    }

	/**
	 * set the Gradient to be 0
	 */
	void zeroGradient() override {
		gradient.zero();
	}

	/**
	 * get the device type
	 */
	DeviceType deviceType() override {
		return value.device->type;
	}

	/**
	 * set the gradient be 1 for backward process
	 */
	void setGradientOne() override {
		DEEP8_ARGUMENT_CHECK(gradient.isScalar(), "the gradient is  not scalar");

		if (DeviceType::CPU == gradient.device->type) {
			(static_cast<T*>(gradient.pointer))[0] = T(1);
		} else {
			DEEP8_RUNTIME_ERROR("GPU not supported for now");
		}
	}

	bool isScalar() override {
		return value.isScalar() && gradient.isScalar();
	}
};

}

#endif //DEEP8_VARIABLE_H
