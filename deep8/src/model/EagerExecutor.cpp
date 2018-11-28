#include "Expression.h"
#include "EagerExecutor.h"

namespace Deep8 {

template <typename T>
EagerExecutor<T>::EagerExecutor(Trainer<T> *tr, DeviceType deviceType, bool flag) :
	Executor<T>(tr, deviceType), clearFlag(flag) {
}

template <typename T>
Node* EagerExecutor<T>::addFunction(FunctionBase *function) {
	auto variable = this->createVariableWithFunction(function);

	function->id = this->generateNodeId();
	variable->id = this->generateNodeId();

	this->nodeCollection.insert(function);
	this->nodeCollection.insert(variable);

	this->nonParameterCollection.insert(function);
	this->nonParameterCollection.insert(variable);

	function->forward();

	return variable;
}

template <typename T>
void EagerExecutor<T>::clearIntermediaryNodes() {
	/**clear all node output*/
	for (auto item : this->nodeCollection) {
		item->inputs.clear();
		item->outputs.clear();
	}

	for (auto item : this->nonParameterCollection) {
		this->nodeCollection.erase(item);

		delete item;
	}

	this->nonParameterCollection.clear();
}

template <typename T>
void EagerExecutor<T>::forward(Expression<T> &e) {
	DEEP8_RUNTIME_ERROR("the EagerExecutor can not call the forward");
}

template <typename T>
void EagerExecutor<T>::forward(Node *) {
	DEEP8_RUNTIME_ERROR("the EagerExecutor can not call the forward");
}

template <typename T>
void EagerExecutor<T>::backward(Expression<T> &e) {
	backward(e.node);
}

template <typename T>
void EagerExecutor<T>::backward(Node *last) {
	DEEP8_ARGUMENT_CHECK(nullptr != last && last->type == NodeType::Variable, "the last node must be a Variable");

	auto lastVariable = static_cast<VariableBase*>(last);

	DEEP8_ARGUMENT_CHECK(lastVariable->isScalar(), "the last Variable gradient must be scalar");

	/**
	 * first loop zero all the Gradient of Variable
	 */
	std::queue<Node*> que;
	que.push(last);

	while (!que.empty()) {
		auto node = que.front();
		que.pop();

		if (NodeType::Variable == node->type) {
			static_cast<VariableBase*>(node)->zeroGradient();
		}

		for (auto input : node->inputs) {
			que.push(input);
		}
	}

	/**set the last Variable Gradient is 1*/
	lastVariable->setGradientOne();

	/**calculate the gradient for the Variable*/
	que.push(last);

	while (!que.empty()) {
		auto node = que.front();
		que.pop();

		node->backward();

		for (auto input : node->inputs) {
			que.push(input);
		}
	}

	/**update the parameter*/
	this->trainer->training(this->parameterCollection);

	/**clear the function and variable*/
	if (clearFlag) {
		this->clearIntermediaryNodes();
	}
}

DEEP8_DECLARATION_INSTANCE(EagerExecutor)

}