#ifndef DEEP8_DEFAULTEXECUTOR_H
#define DEEP8_DEFAULTEXECUTOR_H

#include "model/Executor.h"

namespace Deep8 {

/**
 * @brief
 * DefaultExecutor is a special compute graph it will calculate the result immediately after add the Function
 * like a build a graph Z = X + Y
 * First, create 2 Parameter X and Y,
 * Than create a Function Add and the Variable Z.
 * When create the Function Add DefaultExecutor will calculate the result put it into Z immediately.
 */
class EagerExecutor : public Executor {
public:
	explicit EagerExecutor(DeviceType deviceType = DeviceType::CPU, bool flag = true);

	/**give a function and create the output Variable*/
	Node *addFunction(Function *func) override;

	void forward(Node *last) override;
	void backward(Node *last) override;
};

}

#endif