#include "model/Device.h"
#include "model/Executor.h"
#include "model/TensorInit.h"
#include "model/Expression.h"

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
	allVariables.clear();
	allFunctions.clear();
}

void Executor::initDeviceCPU() {
	device = new CPUDevice();
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

int64_t Executor::generateUniqueId() {
	return uniqueId++;
}

Variable* Executor::createVariableByFunction(Function *func) {
	auto type = func->outputElementType();

	if (func->isShared()) {
		auto variable = new Variable(func, func->outputShape);

		return variable;
	}

	if (func->needUpdateGradient()) {
		/**the output need gradient*/
		auto value    = createTensor(func->outputShape, type);
		auto gradient = createTensor(func->outputShape, type);

		auto variable = new Variable(func, value, gradient);

		return variable;
	} else {
		auto value = createTensor(func->outputShape, type);

		auto variable = new Variable(func, value);

		return variable;
	}
}

Variable* Executor::addVariable(std::vector<size_t> list, DType type, bool updateGradient) {
	Shape shape(list);

	return addVariable(shape, type, updateGradient);
}

Variable* Executor::addVariable(size_t batch, std::vector<size_t> list, DType type, bool updateGradient) {
	Shape shape(batch, list);

	return addVariable(shape, type, updateGradient);
}

Variable* Executor::addVariable(Shape &shape, DType type, bool updateGradient) {
	Variable *variable = nullptr;

	if (updateGradient) {
		auto value    = createTensor(shape, type);
		auto gradient = createTensor(shape, type);

		variable = new Variable(value, gradient);
	} else {
		auto value = createTensor(shape, type);

		variable = new Variable(value);
	}

	/**set a id*/
	variable->id = generateUniqueId();

	allNodes[variable->id] = variable;
	allVariables[variable->id] = variable;
	
	return variable;
}

	/**get a Node/Variable/Function by Id*/
Node* Executor::getNodeById(int64_t id) {
	if (allNodes.find(id) != allNodes.end()) {
		return allNodes[id];
	}

	return nullptr;
}

Variable* Executor::getVariableById(int64_t id) {
	if (allVariables.find(id) != allVariables.end()) {
		return allVariables[id];
	}

	return nullptr;
}

Function* Executor::getFunctionById(int64_t id) {
	if (allFunctions.find(id) != allFunctions.end()) {
		return allFunctions[id];
	}

	return nullptr;
}

Node* Executor::addFunction(Function *function) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

void Executor::forward(Expression &e) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

void Executor::backward(Expression &e) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

void Executor::forward(Node *last) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

void Executor::backward(Node *last) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}


}