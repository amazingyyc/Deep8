#ifndef DEEP8_LAZYEXECUTOR_H
#define DEEP8_LAZYEXECUTOR_H

#include "Executor.h"

namespace Deep8 {

/**
 * lazyexecutor is different with eagerexecutor, it does not calculate the result when add a function
 * it will calculate when call forward func, and will optimize the compute graph
 */
template <typename T>
class LazyExecutor : public Executor<T> {
protected:
	bool clearFlag;

	/**use the auto batch algorithm to optimize the compute graph*/
	void autoBatchGraph(Node *);

	/**auto batch in every layer of graph*/
	void autoBatchGraphLayer(std::vector<Node*>&);

	/**malloc Intermediary Variable*/
	void mallocIntermediaryVariable(Node*);

public:
	explicit LazyExecutor(Trainer<T> *tr, DeviceType deviceType = DeviceType::CPU, bool flag = true);

	void clearIntermediaryNodes();

	Node *addFunction(FunctionBase *function) override;

	void forward(Expression<T> &e) override;
	void forward(Node *) override;

	void backward(Expression<T> &e) override;
	void backward(Node *last) override;
};

}

#endif