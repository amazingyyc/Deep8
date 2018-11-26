#ifndef DEEP8_EXECUTOR_H
#define DEEP8_EXECUTOR_H

#include "Basic.h"
#include "Exception.h"
#include "Device.h"
#include "Tensor.h"
#include "Function.h"
#include "Variable.h"
#include "Parameter.h"
#include "Trainer.h"

namespace Deep8 {

template <class T> class Expression;

template <typename T>
class Executor {
protected:
	/**
	 * every node have a unique id
	 */
	int64_t nodeId;

	/**
	 * the device
	 */
	Device *device;
	
	/**
	 * store all the node that have inited
	 */
	std::unordered_set<Node*> nodeCollection;
	
	/**
	 * a collection to store the Parameter Nodes;
	 */
	std::unordered_set<Parameter<T>*> parameterCollection;
	
	/**
	 * store the non-Parameter Nodes
	 * include the Variable and Function
	 */
	std::unordered_set<Node*> nonParameterCollection;
	
	/**trainer to train the parameter*/
	Trainer<T> *trainer;

	explicit Executor(Trainer<T> *tr, DeviceType deviceType = DeviceType::CPU);

protected:
	/**init device*/
	void initDeviceCPU();

#ifdef HAVE_CUDA
	void initDeviceGPU();

	Tensor<T> createTensorGPU(Shape &shape);
#endif
	Tensor<T> createTensor(Shape &shape);
	Tensor<T> createTensorCPU(Shape &shape);

	/**generate a new Node id*/
	int64_t generateNodeId();
	
public:
	virtual ~Executor();

	/**addParameter the batch is default 1*/
	Parameter<T> *addParameter(std::vector<size_t> list, bool updateGradient = true, void *ptr = nullptr);
	Parameter<T> *addParameter(size_t batch, std::vector<size_t> list, bool updateGradient = true, void *ptr = nullptr);
	Parameter<T> *addParameter(Shape &shape, bool updateGradient = true, void *ptr = nullptr);

	virtual Node *addFunction(FunctionBase *function);

	virtual void forward(Expression<T> &e);
	virtual void backward(Expression<T> &e);
	virtual void forward(Node *last);
	virtual void backward(Node *last);
};

}

#endif //DEEP8_EXECUTER_H

















