#include "LazyExecutor.h"

namespace Deep8 {

template <typename T>
LazyExecutor<T>::LazyExecutor(Trainer<T> *tr, DeviceType deviceType, bool flag) :
	Executor<T>(tr, deviceType), clearFlag(flag) {
}

/**use the auto batch algorithm to optimize the compute graph*/
template <typename T>
void LazyExecutor<T>::optimizeGraph(Node *last) {
	/**first loop get all inputs-nodes that the nodes'input is empty*/

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
