#ifndef DEEP8_EXECUTOR_H
#define DEEP8_EXECUTOR_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Device.h"
#include "model/ElementType.h"
#include "model/Tensor.h"
#include "nodes/Function.h"
#include "nodes/Variable.h"

namespace Deep8 {

class Expression;

class Executor {
protected:
	/**every node have a unique id*/
	int64_t uniqueId;

	/**the device*/
	Device *device;

	/**store all nodes*/
	std::unordered_map<int64_t, Node*> allNodes;
	
	/**store all Variable*/
	std::unordered_map<int64_t, Variable*> allVariables;

	/**store all function*/
	std::unordered_map<int64_t, Function*> allFunctions;

	explicit Executor(DeviceType deviceType = DeviceType::CPU);

protected:
	/**init device*/
	void initDeviceCPU();

#ifdef HAVE_CUDA
	void initDeviceGPU();
#endif

	Tensor createTensor(Shape &shape, DType type);
	Tensor createTensor(Shape &shape, ElementType type);

	Tensor createTensorCPU(Shape &shape, ElementType type);

#ifdef HAVE_CUDA
	Tensor createTensorGPU(Shape &shape, ElementType type);
#endif

	/**generate a new id*/
	int64_t generateUniqueId();

	/**create a Variable to store the function output*/
	Variable* createVariableByFunction(Function *func);

public:
	virtual ~Executor();

	/**add a Variable*/
	Variable* addVariable(std::vector<size_t>,               DType type, bool updateGradient = true);
	Variable* addVariable(size_t batch, std::vector<size_t>, DType type, bool updateGradient = true);
	Variable* addVariable(Shape &shape,                      DType type, bool updateGradient = true);

    Variable* addVariable(std::vector<size_t>,               ElementType type, bool updateGradient = true);
    Variable* addVariable(size_t batch, std::vector<size_t>, ElementType type, bool updateGradient = true);
    Variable* addVariable(Shape& shape,                      ElementType type, bool updateGradient = true);

	/**get a Node/Variable/Function by Id*/
	Node*     getNodeById(int64_t id);
	Variable* getVariableById(int64_t id);
	Function* getFunctionById(int64_t id);

	/**get all trainabel parameters*/
	std::vector<Variable*> trainableParameters();

	/**give a function and create the output Variable*/
	virtual Node *addFunction(Function *func);

	virtual void forward(Node *last);
	virtual void backward(Node *last);
};

}

#endif //DEEP8_EXECUTER_H

















