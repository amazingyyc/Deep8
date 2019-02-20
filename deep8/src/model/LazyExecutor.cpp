#include "Batch.h"
#include "UnBatch.h"
#include "Expression.h"
#include "LazyExecutor.h"

namespace Deep8 {

LazyExecutor::LazyExecutor(DeviceType deviceType, bool flag) :
	Executor(deviceType), clearInterim(flag) {
}

/**use the auto batch algorithm to optimize the compute graph*/
void LazyExecutor::autoBatchGraph(Node *last) {
	DEEP8_ARGUMENT_CHECK(nullptr != last, "the Node can not be nullptr");

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
void LazyExecutor::autoBatchGraphLayer(std::vector<Node*> &nodes) {
	/**empty or only 1 node, return*/
	if (1 >= nodes.size()) {
		return;
	}

	/**get the batched indexes*/
	auto indexes = nodes[0]->autoBatchIndexes();
	std::unordered_map<size_t, Function*> batchNodes;

	for (auto i : indexes) {
		std::vector<Node*> inputs;
		std::vector<Shape> inputsShape;

		/**get all inputs node and shape*/
		for (auto item : nodes) {
			inputs.emplace_back(item->inputs[i]);
			inputsShape.emplace_back(item->inputs[i]->shape);
		}

		auto bactedShape = nodes[0]->autoBatchShape(i, inputsShape);
		batchNodes[i] = new Batch(inputs, bactedShape);
		
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
	auto newNode = (Function*) nodes[0]->autoBatchClone(newNodeInputs);
	this->addFunction(newNode);

	std::vector<Node*> unBatchNodes;
	std::vector<Node*> unBatchInputs({ newNode });

	size_t offset = 0;

	for (size_t i = 0; i < nodes.size(); ++i) {
		auto unBatchNode = new UnBatch(unBatchInputs, offset, nodes[i]->shape);
		this->addFunction(unBatchNode);

		unBatchNodes.emplace_back(unBatchNode);

		offset += unBatchNode->shape.size();
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

		this->allNodes.erase(nodes[i]->id);
		this->allVariables.erase(nodes[i]->id);
		this->allFunctions.erase(nodes[i]->id);
		this->interimNodes.erase(nodes[i]->id);

		delete nodes[i];
	}
}

/**
 * malloc Intermediary Variable
 */
void LazyExecutor::mallocInterimVariable(Node *last) {
	/**use Topological sorting to malloc Varialbe avoid the inputs of Function is not Variable*/

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

	while (!zeroInDegree.empty()) {
		auto size = zeroInDegree.size();

		for (unsigned long i = 0; i < size; ++i) {
			auto node = zeroInDegree.front();
			zeroInDegree.pop();

			for (auto item : node->outputs.entries) {
				inDegree[item.first]--;

				if (0 == inDegree[item.first]) {
					zeroInDegree.push(item.first);
				}
			}

			if (NodeType::Function != node->type) {
				continue;
			}

			auto variable = this->createVariableByFunction((Function*) node);
			variable->id  = this->generateUniqueId();

			this->allNodes[variable->id]     = variable;
			this->allVariables[variable->id] = variable;
			this->interimNodes[variable->id] = variable;

			variable->inputs.clear();
			variable->outputs.clear();

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
}

void LazyExecutor::clearInterimNodes() {
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

Node* LazyExecutor::addFunction(Function *function) {
	function->id = this->generateUniqueId();

	this->allNodes[function->id]     = function;
	this->allFunctions[function->id] = function;
	this->interimNodes[function->id] = function;

	return function;
}

void LazyExecutor::forward(Expression &e) {
	this->forward(e.node);
}

void LazyExecutor::backward(Expression &e) {
	this->backward(e.node);
}

void LazyExecutor::forward(Node *last) {
	DEEP8_ARGUMENT_CHECK(nullptr != last, "the last can not be nullptr");

	/**first step autobath this graph*/
	autoBatchGraph(last);

	/**sencond step, malloc the intermediary variable*/
	mallocInterimVariable(last);

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

void LazyExecutor::backward(Node *last) {
	Variable *lastVar = nullptr;

	if (NodeType::Function == last->type) {
		/**if the node is a function must have a variable output*/
		DEEP8_ARGUMENT_CHECK(1 == last->outputs.size() && NodeType::Variable == last->outputs.first()->type, "the last node must have a Variable output");

		lastVar = (Variable*)(last->outputs.first());
	} else if (NodeType::Variable == last->type) {
		lastVar = (Variable*)(last);
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
			((Variable*)node)->zeroGradient();
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

	/**clear the function and variable*/
	if (clearInterim) {
		this->clearInterimNodes();
	}
}





}
