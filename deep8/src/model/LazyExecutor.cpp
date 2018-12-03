#include "Batch.h"
#include "UnBatch.h"
#include "Expression.h"
#include "LazyExecutor.h"

namespace Deep8 {

template <typename T>
LazyExecutor<T>::LazyExecutor(Trainer<T> *tr, DeviceType deviceType, bool flag) :
	Executor<T>(tr, deviceType), clearFlag(flag) {
}

/**use the auto batch algorithm to optimize the compute graph*/
template <typename T>
void LazyExecutor<T>::autoBatchGraph(Node *last) {
	DEEP8_ARGUMENT_CHECK(nullptr != last, "the Parameter can not be nullptr");

	/**store the node that have 0 inputs*/
	std::queue<Node*> zeroInDegree;

	/**store the Node's in-degree*/
	std::unordered_map<Node*, int> inDegree;

	/**use the queue to loop the graph*/
	std::queue<Node*> que;
	std::unordered_set<Node*> visited;

	que.push(last);
	visited.insert(last);

	while (!que.empty()) {
		auto node = que.front();
		que.pop();

		for (auto item : node->outputs.entries) {
			inDegree[item.first]++;
		}

		if (node->inputs.empty()) {
			zeroInDegree.push(node);
		} else {
			for (auto item : node->inputs) {
				if (visited.find(item) == visited.end()) {
					que.push(item);
					visited.insert(item);
				}
			}
		}
	}

	/**Topological sorting, in every "layer", batch the Node that can be bacthed*/
	while (!zeroInDegree.empty()) {
		std::unordered_map<size_t, std::vector<Node*>> layer;

		auto size = zeroInDegree.size();

		for (unsigned long i = 0; i < size; ++i) {
			auto node = zeroInDegree.front();
			zeroInDegree.pop();

			if (0 <= node->supportAutoBatch()) {
				layer[node->autoBatchCode()].emplace_back(node);
			}

			for (auto item : node->outputs.entries) {
				inDegree[item.first]--;

				if (0 == inDegree[item.first]) {
					zeroInDegree.push(item.first);
				}
			}
		}

		for (auto item : layer) {
			autoBatchGraphLayer(item.second);
		}
	}
}

/**auto batch in every layer of graph*/
template <typename T>
void LazyExecutor<T>::autoBatchGraphLayer(std::vector<Node*> &nodes) {
	/**empty or only 1 node, return*/
	if (1 >= nodes.size()) {
		return;
	}

	/**get the batched indexes*/
	auto indexes = nodes[0]->autoBatchIndexes();
	std::unordered_map<size_t, FunctionBase*> batchNodes;

	for (auto i : indexes) {
		std::vector<Node*> inputs;
		std::vector<Shape> inputsShape;

		/**get all inputs node and shape*/
		for (auto item : nodes) {
			inputs.emplace_back(item->inputs[i]);
			inputsShape.emplace_back(item->inputs[i]->outputShape);
		}

		auto bactedShape = nodes[0]->autoBatchShape(i, inputsShape);
		batchNodes[i] = new Batch<T>(inputs, bactedShape);
		
		this->addFunction(batchNodes[i]);
	}

	std::vector<Node*> newNodeInputs;

	for (size_t i = 0; i < nodes[0]->inputs.size(); ++i) {
		if (batchNodes.find(i) != batchNodes.end()) {
			newNodeInputs.emplace_back(batchNodes[i]);
		} else {
			newNodeInputs.emplace_back(nodes[0]->inputs[i]);
		}
	}

	/**use the newNode instead of nodes*/
	auto newNode = (FunctionBase*) nodes[0]->autoBatchClone(newNodeInputs);
	this->addFunction(newNode);

	std::vector<Node*> unBatchNodes;
	std::vector<Node*> unBatchInputs({ newNode });
	size_t offset = 0;

	for (size_t i = 0; i < nodes.size(); ++i) {
		auto unBatchNode = new UnBatch<T>(unBatchInputs, offset, nodes[i]->outputShape);
		this->addFunction(unBatchNode);

		unBatchNodes.emplace_back(unBatchNode);

		offset += unBatchNode->outputShape.size();
	}

	/**should clean the nodes and it's inputs, outputs*/
	for (size_t i = 0; i < nodes.size(); ++i) {
		for (auto item : nodes[i]->inputs) {
			item->outputs.remove(nodes[i]);
		}

		for (auto item : nodes[i]->outputs.entries) {
			item.first->inputs[item.second] = unBatchNodes[i];
			unBatchNodes[i]->outputs.add(item.first, item.second);
		}

		nodes[i]->inputs.clear();
		nodes[i]->outputs.clear();

		this->nodeCollection.erase(nodes[i]);
		this->nonParameterCollection.erase(nodes[i]);

		delete nodes[i];
	}
}

/**
 * malloc Intermediary Variable
 */
template <typename T>
void LazyExecutor<T>::mallocIntermediaryVariable(Node *last) {
	std::unordered_set<Node*> visited;
	std::queue<Node*> que;

	que.push(last);
	visited.insert(last);

	while (!que.empty()) {
		auto node = que.front();
		que.pop();

		for (auto item : node->inputs) {
			if (visited.find(item) == visited.end()) {
				que.push(item);
				visited.insert(item);
			}
		}

		if (NodeType::Function != node->type) {
			continue;
		}

		auto variable = this->createVariableWithFunction((FunctionBase*) node);
		variable->id  = this->generateNodeId();

		this->nodeCollection.insert(variable);
		this->nonParameterCollection.insert(variable);

		variable->inputs.clear();
		node->outputs.remove(variable);

		for (auto item : node->outputs.entries) {
			item.first->inputs[item.second] = variable;
			variable->outputs.add(item.first, item.second);
		}

		node->outputs.clear();

		variable->inputs.emplace_back(node);
		node->outputs.add(variable, 0);
	}
}

template <typename T>
void LazyExecutor<T>::clearIntermediaryNodes() {
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
Node* LazyExecutor<T>::addFunction(FunctionBase *function) {
	function->id = this->generateNodeId();

	this->nodeCollection.insert(function);
	this->nonParameterCollection.insert(function);

	return function;
}

template <typename T>
void LazyExecutor<T>::forward(Expression<T> &e) {
	this->forward(e.node);
}

template <typename T>
void LazyExecutor<T>::backward(Expression<T> &e) {
	this->backward(e.node);
}

template <typename T>
void LazyExecutor<T>::forward(Node *last) {
	DEEP8_ARGUMENT_CHECK(nullptr != last, "the last can not be nullptr");

	/**first step autobath this graph*/
	autoBatchGraph(last);

	/**sencond step, malloc the intermediary variable*/
	mallocIntermediaryVariable(last);

	/**do the real calculation*/
	std::unordered_set<Node*> visited;
	std::queue<Node*> que;

	que.push(last);
	visited.insert(last);

	std::vector<Node*> graph;

	while (!que.empty()) {
		auto node = que.front();
		que.pop();

		graph.emplace_back(node);

		for (auto item : node->inputs) {
			if (visited.find(item) == visited.end()) {
				que.push(item);
				visited.insert(item);
			}
		}
	}

	for (auto it = graph.rbegin(); it != graph.rend(); it++) {
		(*it)->forward();
	}
}

template <typename T>
void LazyExecutor<T>::backward(Node *last) {
	VariableBase *lastVar = nullptr;

	if (NodeType::Function == last->type) {
		/**if the node is a function must have a variable output*/
		DEEP8_ARGUMENT_CHECK(1 == last->outputs.size() && NodeType::Variable == last->outputs.first()->type, "the last node must have a Variable output");

		lastVar = static_cast<VariableBase*>(last->outputs.first());
	} else if (NodeType::Variable == last->type) {
		lastVar = static_cast<VariableBase*>(last);
	} else {
		DEEP8_RUNTIME_ERROR("the node type is error");
	}

	DEEP8_ARGUMENT_CHECK(lastVar->isScalar(), "the last Variable gradient must be scalar");

	/**
	 * first loop zero all the Gradient of Variable
	 */
	std::unordered_set<Node*> visited;
	std::queue<Node*> que;

	que.push(lastVar);
	visited.insert(lastVar);

	while (!que.empty()) {
		auto node = que.front();
		que.pop();

		if (NodeType::Variable == node->type) {
			static_cast<VariableBase*>(node)->zeroGradient();
		}

		for (auto input : node->inputs) {
			if (visited.find(input) == visited.end()) {
				que.push(input);
				visited.insert(input);
			}
		}
	}

	/**set the last Variable Gradient is 1*/
	lastVar->setGradientOne();

	/**calculate the gradient for the Variable*/
	visited.clear();

	que.push(lastVar);
	visited.insert(lastVar);

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

	/**update the parameter*/
	this->trainer->training(this->parameterCollection);

	/**clear the function and variable*/
	if (clearFlag) {
		this->clearIntermediaryNodes();
	}
}

DEEP8_DECLARATION_INSTANCE(LazyExecutor)

}
