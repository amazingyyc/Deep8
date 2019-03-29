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

/**
 * the NodeName Type
 */
enum class NameType {
    Node = 0,
    Variable,
    Function,
    Conv2d_Weight,
    Conv2d_Bias,
    DeConv2d_Weight,
    DeConv2d_Bias,
    MatrixMultiply_Weight,
};

class Executor {
protected:
	/**every node have a unique id*/
	int64_t uniqueId;

    /**store the name have been setted to avoid same name*/
    std::unordered_map<NameType, int64_t> nameUniqueId;
    std::unordered_set<std::string> uniqueNames;

	/**the device*/
	Device *device;

    /**store all nodes*/
    std::unordered_map<std::string, Node*> allNodes;

    /**the Variable that need to be store permanent, like trainable Parameter or some Variable need tobe use in training*/
    std::unordered_map<std::string, Variable*> permanentVariables;

    /**store the interim Nodes, the nodes will be deleted when after backward include interim Variable and Function*/
    std::unordered_map<std::string, Node*> interimNodes;

	explicit Executor(DeviceType deviceType = DeviceType::CPU);

protected:
    /**init device*/
    void initDeviceCPU();

#ifdef HAVE_CUDA
    void initDeviceGPU();
#endif

    /**get the name prefix*/
    std::string namePrefix(NameType nameType);

    /**generate a unique name*/
    std::string generateUniqueName(NameType nameType);

    /**generate a new id*/
    int64_t generateUniqueId();

	Tensor createTensor(Shape &shape, DType type);
	Tensor createTensor(Shape &shape, ElementType type);

	Tensor createTensorCPU(Shape &shape, ElementType type);

#ifdef HAVE_CUDA
	Tensor createTensorGPU(Shape &shape, ElementType type);
#endif


	/**create a Variable to store the function output*/
	Variable* createVariableByFunction(Function *func);

public:
	virtual ~Executor();

    /**remove all Variable's gradient for predict mode*/
    void removeGradient();

    /**add a Variable updateGradient: if the variable has the gradient, retain: if ratian in memory after backwarx*/
    Variable* addVariable(Shape &shape,                      DType type, bool updateGradient = true, bool retain = true);
    Variable* addVariable(std::vector<size_t>,               DType type, bool updateGradient = true, bool retain = true);
    Variable* addVariable(size_t batch, std::vector<size_t>, DType type, bool updateGradient = true, bool retain = true);

    Variable* addVariable(Shape& shape,                      ElementType type, bool updateGradient = true, bool retain = true);
    Variable* addVariable(std::vector<size_t>,               ElementType type, bool updateGradient = true, bool retain = true);
    Variable* addVariable(size_t batch, std::vector<size_t>, ElementType type, bool updateGradient = true, bool retain = true);

	/**clear interim nodes*/
	void clearInterimNodes();

	Node* nodeByName(std::string name);

	Variable* permanentVariableByName(std::string name);

	/**get all trainabel parameters*/
	std::vector<Variable*> trainableParameters();

	/**give a function and create the output Variable*/
	virtual Node *addFunction(Function *func);

	virtual void forward(Node *last);
	virtual void backward(Node *last);
};

}

#endif //DEEP8_EXECUTER_H

















