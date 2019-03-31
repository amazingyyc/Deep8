#include "nodes/Node.h"
#include "nodes/Variable.h"
#include "nodes/Function.h"
#include "model/Device.h"
#include "model/Executor.h"

namespace Deep8 {

Executor::Executor(DeviceType deviceType): uniqueId(0) {
	if (deviceType == DeviceType::CPU) {
		initDeviceCPU();
	} else {
#ifdef HAVE_CUDA
		initDeviceGPU();
#else
		DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
	}
}

Executor::~Executor() {
	for (auto item : allNodes) {
		delete item.second;
	}

	delete device;

	allNodes.clear();
	retainVariables.clear();
	interimNodes.clear();
}

void Executor::initDeviceCPU() {
	device = new CPUDevice();
}

/**get the name prefix*/
std::string Executor::namePrefix(NameType nameType) {
    switch (nameType) {
    case NameType::Node:
        return "Node_";
    case NameType::Variable:
        return "Variable_";
    case NameType::Function:
        return "Function_";
    case NameType::Conv2d_Weight:
        return "Conv2d_Weight_";
    case NameType::Conv2d_Bias:
        return "Conv2d_Bias_";
    case NameType::DeConv2d_Weight:
        return "DeConv2d_Weight_";
    case NameType::DeConv2d_Bias:
        return "DeConv2d_Bias_";
    case NameType::MatrixMultiply_Weight:
        return "MatrixMultiply_Weight_";
    default:
        DEEP8_RUNTIME_ERROR("the name type is error");
    }
}

std::string Executor::generateUniqueName(NameType type) {
    auto prefix = namePrefix(type);
    auto name   = prefix + std::to_string(nameUniqueId[prefix]++);

    while (uniqueNames.find(name) != uniqueNames.end()) {
        name = prefix + std::to_string(nameUniqueId[prefix]++);
    }

	uniqueNames.insert(name);
	
    return name;
}

int64_t Executor::generateUniqueId() {
    return uniqueId++;
}

Tensor Executor::createTensor(Shape &shape, DType type) {
	return createTensor(shape, ElementType::from(type));
}

Tensor Executor::createTensor(Shape &shape, ElementType type) {
	if (DeviceType::CPU == device->type) {
		return createTensorCPU(shape, type);
	} else {
#ifdef HAVE_CUDA
		return createTensorGPU(shape, type);
#else
		DEEP8_RUNTIME_ERROR("without a GPU");
#endif
	}
}

Tensor Executor::createTensorCPU(Shape &shape, ElementType type) {
	size_t size = type.byteWidth * shape.size();

	auto ptr    = device->malloc(size);
	auto refPtr = (size_t*) device->malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, size, device);

	return Tensor(storage, 0, shape, type);
}

Variable* Executor::createVariableByFunction(Function *func) {
	auto id   = generateUniqueId();
	auto name = generateUniqueName(NameType::Variable);

	if (func->isShared()) {
		return new Variable(id, name, this, func);
	}

	bool updateGradient = false;

	std::vector<Shape> inputShapes;
	std::vector<ElementType> inputTypes;

	for (auto item : func->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the input must be a Variable");

		auto input = (Variable*) item;

		inputShapes.emplace_back(input->shape());
		inputTypes.emplace_back(input->elementType());

		if (input->updateGradient) {
			updateGradient = true;
		}
	}

	auto shape       = func->checkShape(inputShapes);
	auto elementType = func->checkElementType(inputTypes);

	if (updateGradient) {
		auto value    = createTensor(shape, elementType);
		auto gradient = createTensor(shape, elementType);

		return new Variable(id, name, this, func, value, gradient);
	} else {
		auto value = createTensor(shape, elementType);

		return new Variable(id, name, this, func, value);
	}
}

void Executor::clearInterimNodes() {
	/**clear all node output*/
	for (auto item : this->allNodes) {
		item.second->inputs.clear();
		item.second->outputs.clear();
	}

	for (auto item : this->interimNodes) {
		this->allNodes.erase(item.first);
		this->retainVariables.erase(item.first);

		delete item.second;
	}

	this->interimNodes.clear();
}

/**remove all Variable's gradient for predict mode*/
void Executor::removeGradient() {
    for (auto item : allNodes) {
        auto node = item.second;

        if (NodeType::Variable == node->type) {
            ((Variable*) node)->removeGradient();
        }
    }
}

Variable* Executor::addVariable(Shape &shape, ElementType type, bool updateGradient, bool retain) {
	return addVariable(shape, type.id, updateGradient, retain);
}

Variable* Executor::addVariable(std::vector<size_t> list, ElementType type, bool updateGradient, bool retain) {
	return addVariable(list, type.id, updateGradient, retain);
}

Variable* Executor::addVariable(size_t batch, std::vector<size_t> list,  ElementType type, bool updateGradient, bool retain) {
	return addVariable(batch, list, type.id, updateGradient, retain);
}

Variable* Executor::addVariable(std::vector<size_t> list, DType type, bool updateGradient, bool retain) {
	Shape shape(list); 

	return addVariable(shape, type, updateGradient, retain);
}

Variable* Executor::addVariable(size_t batch, std::vector<size_t> list, DType type, bool updateGradient, bool retain) {
	Shape shape(batch, list);

	return addVariable(shape, type, updateGradient, retain);
}

Variable* Executor::addVariable(Shape &shape, DType type, bool updateGradient, bool retain) {
	auto id   = this->generateUniqueId();
	auto name = this->generateUniqueName(NameType::Variable);

	Variable *variable = nullptr;

	if (updateGradient) {
		auto value    = createTensor(shape, type);
		auto gradient = createTensor(shape, type);

		variable = new Variable(id, name, this, value, gradient);
	} else {
		auto value = createTensor(shape, type);

		variable = new Variable(id, name, this, value);
	}	

	allNodes[name] = variable;

    if (retain) {
        retainVariables[name] = variable;
    } else {
        interimNodes[name] = variable;
    }

	return variable;
}

Variable* Executor::addVariable(std::string name, Shape &shape, ElementType type, bool updateGradient, bool retain) {
	return addVariable(name, shape, type.id, updateGradient, retain);
}

Variable* Executor::addVariable(std::string name, std::vector<size_t> list, ElementType type, bool updateGradient, bool retain) {
	return addVariable(name, list, type.id, updateGradient, retain);
}

Variable* Executor::addVariable(std::string name, size_t batch, std::vector<size_t> list,  ElementType type, bool updateGradient, bool retain) {
	return addVariable(name, batch, list, type.id, updateGradient, retain);
}


Variable* Executor::addVariable(std::string name, std::vector<size_t> list, DType type, bool updateGradient, bool retain) {
	Shape shape(list);

	return addVariable(name, shape, type, updateGradient, retain);
}

Variable* Executor::addVariable(std::string name, size_t batch, std::vector<size_t> list,  DType type, bool updateGradient, bool retain) {
	Shape shape(batch, list);

	return addVariable(name, shape, type, updateGradient, retain);
}

Variable* Executor::addVariable(std::string name, Shape &shape, DType type, bool updateGradient, bool retain) {
	DEEP8_ARGUMENT_CHECK(uniqueNames.find(name) == uniqueNames.end(), "The name:" << name << "have been used, should set a uniqe name");

	auto id = this->generateUniqueId();

	/**insert this name*/
	uniqueNames.insert(name);

	Variable *variable = nullptr;

	if (updateGradient) {
		auto value    = createTensor(shape, type);
		auto gradient = createTensor(shape, type);

		variable = new Variable(id, name, this, value, gradient);
	} else {
		auto value = createTensor(shape, type);

		variable = new Variable(id, name, this, value);
	}	

	allNodes[name] = variable;

    if (retain) {
        retainVariables[name] = variable;
    } else {
        interimNodes[name] = variable;
    }

	return variable;
}

Node* Executor::nodeByName(std::string name) {
    if (allNodes.find(name) != allNodes.end()) {
        return allNodes[name];
    }

    return nullptr;
}

Variable* Executor::retainVariableByName(std::string name) {
    if (retainVariables.find(name) != retainVariables.end()) {
        return retainVariables[name];
    }

    return nullptr;
}

/**get all trainable parameters*/
std::vector<Variable*> Executor::trainableParameters() {
	std::vector<Variable*> parameters;

	for (auto item : this->retainVariables) {
		if (item.second->updateGradient) {
			parameters.emplace_back(item.second);
		}
	}

	return parameters;
}

Variable& Executor::addFunction(Function *func) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

void Executor::forward(Node *last) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

void Executor::backward(Node *last, bool clearInterim) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}


}