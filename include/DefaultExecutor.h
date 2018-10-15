#ifndef DEEP8_DEFAULTEXECUTOR_H
#define DEEP8_DEFAULTEXECUTOR_H

#include "Executor.h"

namespace Deep8 {

/**
 * @brief
 * DefaultExecutor is a special compute graph it will calculate the result immediately after add the Function
 * like a build a graph Z = X + Y
 * First, create 2 Parameter X and Y,
 * Than create a Function Add and the Variable Z.
 * When create the Function Add DefaultExecutor will calculate the result put it into Z immediately.
 */
template <typename T>
class DefaultExecutor : public Executor<T> {
protected:
	/**
	 * @brief if the Executor will delete the Function and Variable after the backward process.
	 * the default is true
	 * in dynamic Deep Learning, it will build a compute graph for every sample, so if we do not delete the Function and Variable Node
	 * the temp Node will be consume the memory
	 * like a graph Z = W * X + B
	 * the X is InputParameter, W and B is Parameter
	 * for every sample if will generate 2 intermediary Variable, one is the output of W * X, another is T1 + B
	 * so if have 400 sample it will generate 800 intermediary Variable and Function in the memory.
	 * and it will consume a lot of memory
	 *
	 * if the user want watch the intermediary result, it could set the clearFlag is false
	 * and clean the intermediary Node by himself using function cleanIntermediaryNodes()
	 */
	bool clearFlag;

public:
	explicit DefaultExecutor(Trainer<T> *tr, DeviceType deviceType = DeviceType::CPU, bool flag = true);

	Node *addFunction(FunctionBase *function) override;

	void clearIntermediaryNodes();

	void forward(Expression<T> &e) override;
	void forward(Node *) override;

	void backward(Expression<T> &e) override;
	void backward(Node *last) override;
};

}

#endif