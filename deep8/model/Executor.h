#ifndef DEEP8_EXECUTOR_H
#define DEEP8_EXECUTOR_H

#include "../basic/Basic.h"
#include "../basic/Exception.h"
#include "../basic/Device.h"
#include "../basic/Tensor.h"
#include "../nodes/Function.h"
#include "../nodes/Variable.h"
#include "../nodes/Parameter.h"
#include "../nodes/InputParameter.h"
#include "Trainer.h"

namespace Deep8 {

template <class T> class Expression;

template <typename T>
class Executor {
protected:
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
	void initDeviceGPU();

	Tensor<T> createTensorWithShapeCPU(Shape &shape);
	Tensor<T> createTensorWithShapeGPU(Shape &shape);
	Tensor<T> createTensorWithShape(Shape &shape);

	Variable<T>* createVariableByFunction(FunctionBase *function);

public:
	virtual ~Executor();

	Parameter<T> *addParameter(std::initializer_list<size_t> list);
	Parameter<T> *addParameter(Shape &shape);

	InputParameter<T> *addInputParameter(std::initializer_list<size_t> list, T *ptr = nullptr);
	InputParameter<T> *addInputParameter(Shape &shape, T *ptr = nullptr);

	virtual Node *addFunction(FunctionBase *function);

	virtual void forward(Expression<T> &e) = 0;
	virtual void backward(Expression<T> &e) = 0;
	virtual void forward(Node *last) = 0;
	virtual void backward(Node *last) = 0;
};

}

#endif //DEEP8_EXECUTER_H

















