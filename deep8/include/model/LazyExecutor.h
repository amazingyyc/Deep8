#ifndef DEEP8_LAZYEXECUTOR_H
#define DEEP8_LAZYEXECUTOR_H

#include "Executor.h"

namespace Deep8 {

/**
 * lazyexecutor is different with eagerexecutor, it does not calculate the result when add a function
 * it will calculate when call forward func, and will optimize the compute graph
 */
class LazyExecutor : public Executor {
protected:
	bool clearInterim;

	/**store the interim Nodes*/
	std::unordered_map<int64_t, Node*> interimNodes;

protected:
	/**use the auto batch algorithm to optimize the compute graph*/
	void autoBatchGraph(Node *);

	/**auto batch in every layer of graph*/
	void autoBatchGraphLayer(std::vector<Node*>&);

	/**malloc Intermediary Variable*/
	void mallocInterimVariable(Node*);

public:
	explicit LazyExecutor(DeviceType deviceType = DeviceType::CPU, bool flag = true);

	void clearInterimNodes();

	Node *addFunction(Function *func) override;

	void forward(Expression &e) override;
	void forward(Node *) override;

	void backward(Expression &e) override;
	void backward(Node *last) override;
};

}

#endif