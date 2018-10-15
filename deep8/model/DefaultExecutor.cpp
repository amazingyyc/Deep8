#include "Expression.h"
#include "DefaultExecutor.h"

namespace Deep8 {

template <typename T>
DefaultExecutor<T>::DefaultExecutor(Trainer<T> *tr, DeviceType deviceType, bool flag) :
	Executor<T>(tr, deviceType), clearFlag(flag) {
}

template <typename T>
Node* DefaultExecutor<T>::addFunction(FunctionBase *function) {
	auto variable = Executor<T>::addFunction(function);

	function->forward();

	return variable;
}

template <typename T>
void DefaultExecutor<T>::clearIntermediaryNodes() {
	for (auto item : this->nonParameterCollection) {
		this->nodeCollection.erase(item);

		delete item;
	}

	this->nonParameterCollection.clear();
}

template <typename T>
void DefaultExecutor<T>::forward(Expression<T> &e) {
	DEEP8_RUNTIME_ERROR("the DefaultExecutor can not call the forward");
}

template <typename T>
void DefaultExecutor<T>::forward(Node *) {
	DEEP8_RUNTIME_ERROR("the DefaultExecutor can not call the forward");
}

template <typename T>
void DefaultExecutor<T>::backward(Expression<T> &e) {
	backward(e.node);
}

template <typename T>
void DefaultExecutor<T>::backward(Node *last) {
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
	this->trainer->training(this->parameterCollection);

	/**clear the function and variable*/
	if (clearFlag) {
		this->clearIntermediaryNodes();
	}
}

DEEP8_DECLARATION_INSTANCE(DefaultExecutor)

}