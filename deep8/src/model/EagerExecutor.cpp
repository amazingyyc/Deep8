#include "model/Expression.h"
#include "model/EagerExecutor.h"

namespace Deep8 {

EagerExecutor::EagerExecutor(DeviceType deviceType, bool flag) :
	Executor(deviceType), clearInterim(flag) {
}

Node* EagerExecutor::addFunction(Function *function, DType type) {
	auto variable = this->createVariableByFunction(function, type);

	function->id = this->generateUniqueId();
	variable->id = this->generateUniqueId();

	this->allNodes[function->id] = function;
	this->allNodes[variable->id] = variable;

	this->allFunctions[function->id] = function;
	this->allVariables[variable->id] = variable;

	/**add to interim node*/
	this->interimNodes[function->id] = function;
	this->interimNodes[variable->id] = variable;

	/**calculate output*/
	function->forward();

	return variable;
}

void EagerExecutor::clearInterimNodes() {
	/**clear all node output*/
	for (auto item : this->allNodes) {
		item.second->inputs.clear();
		item.second->outputs.clear();
	}

	for (auto item : this->interimNodes) {
		this->allNodes.erase(item.first);
		this->allFunctions.erase(item.first);
		this->allVariables.erase(item.first);

		delete item.second;
	}

	this->interimNodes.clear();
}

void EagerExecutor::forward(Expression &e) {
	DEEP8_RUNTIME_ERROR("the EagerExecutor can not call the forward");
}

void EagerExecutor::forward(Node *) {
	DEEP8_RUNTIME_ERROR("the EagerExecutor can not call the forward");
}

void EagerExecutor::backward(Expression &e) {
	backward(e.node);
}

void EagerExecutor::backward(Node *last) {
	DEEP8_ARGUMENT_CHECK(nullptr != last && last->type == NodeType::Variable, "the last node must be a Variable");

	auto lastVariable = (Variable*)(last);

	DEEP8_ARGUMENT_CHECK(lastVariable->isScalar(), "the last Variable gradient must be scalar");

	/**
	 * first loop zero all the Gradient of Variable
	 */
	std::queue<Node*> que;
	std::unordered_set<Node*> visited;

	que.push(last);
	visited.insert(last);

	while (!que.empty()) {
		auto node = que.front();
		que.pop();

		if (NodeType::Variable == node->type) {
			((Variable*)node)->zeroGradient();
		}

		for (auto input : node->inputs) {
			if (visited.find(input) == visited.end()) {
				que.push(input);
				visited.insert(input);
			}
		}
	}

	/**set the last Variable Gradient to 1*/
	lastVariable->setGradientOne();

	/**calculate the gradient for the Variable*/
	visited.clear();
	que.push(last);
	visited.insert(last);

	while (!que.empty()) {
		auto node = que.front();
		que.pop();

		node->backward();

		for (auto input : node->inputs) {
			if (visited.find(input) == visited.end()) {
				que.push(input);
				visited.insert(input);
			}
		}
	}

	/**clear the function and variable*/
	if (clearInterim) {
		this->clearInterimNodes();
	}
}



}