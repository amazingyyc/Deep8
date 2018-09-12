#ifndef DEEP8_DEFAULTEXECUTOR_H
#define DEEP8_DEFAULTEXECUTOR_H

#include <iostream>
#include <queue>

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
	explicit DefaultExecutor(TrainerType trainerType = TrainerType::SGD, DeviceType deviceType = DeviceType::CPU, bool flag = true) :
		Executor<T>(trainerType, deviceType), clearFlag(flag) {
	}

protected:
	void afterAddFunctionNode(Node *function, Node *variable) override {
		/**calculate the result immediate*/
		function->forward();
	}

public:
	void clearIntermediaryNodes() {
		for (auto item : nonParameterCollection) {
			nodeCollection.erase(item);

			delete item;
		}

		nonParameterCollection.clear();
	}

	void forward(Expression<T> &e) {
		DEEP8_RUNTIME_ERROR("the DefaultExecutor can not call the forward");
	}

	void backward(Expression<T> &e) {
		backward(e.node);
	}

	void forward(Node *) override {
		DEEP8_RUNTIME_ERROR("the DefaultExecutor can not call the forward");
	}

	void backward(Node *last) override {
		DEEP8_ARGUMENT_CHECK(nullptr != last && last->type == NodeType::Variable, "the last node must be a Variable");

		auto lastVariable = static_cast<VariableBase*>(last);

		DEEP8_ARGUMENT_CHECK(lastVariable->isScalar(), "the last Variable gradient must be scalar");

		/**
		 * first loop zero all the Gradient of Variable
		 */
		std::queue<Node*> queues;
		queues.push(last);

		while (!queues.empty()) {
			auto size = queues.size();

			for (unsigned long i = 0; i < size; ++i) {
				auto node = queues.front();
				queues.pop();

				if (NodeType::Variable == node->type) {
					static_cast<VariableBase*>(node)->zeroGradient();
				}

				for (auto input : node->inputs) {
					queues.push(input);
				}
			}
		}

		/**set the last Variable Gradient is 1*/
		lastVariable->setGradientOne();

		/**calculate the gradient for the Variable*/
		queues.push(last);

		while (!queues.empty()) {
			auto size = queues.size();

			for (std::queue<Node*>::size_type i = 0; i < size; ++i) {
				auto node = queues.front();
				queues.pop();

				node->backward();

				for (auto input : node->inputs) {
					queues.push(input);
				}
			}
		}

		/**update the parameter*/
		trainer->training(parameterCollection);

		/**clear the function and variable*/
		if (clearFlag) {
			this->clearIntermediaryNodes();
		}
	}
};

}

#endif