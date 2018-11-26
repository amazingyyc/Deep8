#include <queue>
#include <unordered_map>

#include "Batch.h"
#include "UnBatch.h"
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

	/**inputs store the Nodes that Node's inputs is empty*/
	std::queue<Node*> inputs;

	/**store the Node's in-degree*/
	std::unordered_map<Node*, int> inDegree;

	/**use the queue to loop the graph*/
	std::queue<Node*> que;
	que.push(last);

	while (!que.empty()) {
		auto size = que.size();

		for (unsigned long i = 0; i < size; ++i) {
			auto node = que.front();
			que.pop();

			for (auto item : node->outputs.outputs) {
				inDegree[item]++;
			}

			if (node->inputs.empty()) {
				inputs.push(node);
			} else {
				for (auto item : node->inputs) {
					que.push(input);
				}
			}
		}
	}

	/**Topological sorting, in every "layer", batch the Node that can be bacthed*/
	while (!inputs.empty()) {
		std::unordered_map<size_t, std::vector<Node*>> layer;

		auto size = inputs.size();

		for (unsigned long i = 0; i < size; ++i) {
			auto node = inputs.front();
			inputs.pop();

			if (node->supportAutoBatch()) {
				layer[node->autoBatchCode()].emplace_back(node);
			}

			for (auto item : node->outputs.outputs) {
				inDegree[item]--;

				if (0 == inDegree[item]) {
					inputs.push(item);
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
	/**empty or only 1 node just return*/
	if (1 >= nodes.size()) {
		return;
	}

	/**get the batched indexes*/
	auto indexes = nodes[0]->autoBatchIndexes();
	std::unordered_map<size_t, Node*> batchNodes;

	for (auto i : indexes) {
		std::vector<Node*> inputs;
		std::vector<Shape> inputsShape;

		/**get all inputs node and shape*/
		for (auto item : nodes) {
			inputs.emplace_back(item->inputs[i]);
			inputsShape.emplace_back(item->inputs[i]->outputShape);
		}

		batchNodes[i] = new Batch<T>(inputs, nodes[0]->autoBatchShape(i, inputsShape));
		this->addFunction(batchNodes[i]);
	}

	std::vector<Node*> inputs;

	for (size_t i = 0; i < nodes[0]->inputs.size(); ++i) {
		if (batchNodes.find(i) != batchNodes.end()) {
			inputs.emplace_back(batchNodes[i]);
		} else {
			inputs.emplace_back(nodes[0]->inputs[i]);
		}
	}

	/**use the newNode instead of nodes*/
	auto newNode = nodes[0]->autoBatchClone(inputs);
	this->addFunction(newNode);

	std::vector<Node*> unBatchNodes;
	std::vector<Node*> unBatchInputs({ newNode });
	size_t offset = 0;

	for (size_t i = 0; i < nodes.size(); ++i) {
		auto unBatchNode = new UnBatch<T>(unBatchInputs, offset, nodes[i]->outputShape);

		unBatchNodes.emplace_back(unBatchNode);

		this->addFunction(unBatchNode);

		offset += nodes[i]->outputShape.size();
	}

	/**should clear the nodes and it's inputs, outputs, not finish*/

}

template <typename T>
void LazyExecutor<T>::clearIntermediaryNodes() {
	/**clear all node output*/
	for (auto item : this->nodeCollection) {
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

}

template <typename T>
void LazyExecutor<T>::backward(Node *last) {

}

}
