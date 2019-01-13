#include "Device.h"
#include "Executor.h"
#include "TensorInit.h"

namespace Deep8 {

template <typename T>
Executor<T>::Executor(Trainer<T> *tr, DeviceType deviceType)
	:nodeId(0),
	trainer(tr), 
	nodeCollection(), 
	parameterCollection(), 
	nonParameterCollection() {

	if (deviceType == DeviceType::CPU) {
		initDeviceCPU();
	} else {
#ifdef HAVE_CUDA
		initDeviceGPU();
#else
		DEEP8_RUNTIME_ERROR("not have a GPU");
#endif
	}
}

template <typename T>
Executor<T>::~Executor() {
	for (auto node : nodeCollection) {
		delete node;
	}

	nodeCollection.clear();
	parameterCollection.clear();
	nonParameterCollection.clear();

	delete trainer;
	delete device;
}

template <typename T>
void Executor<T>::initDeviceCPU() {
	device = new CPUDevice();
}

template <typename T>
Tensor<T> Executor<T>::createTensorCPU(Shape &shape) {
	size_t size = sizeof(T) * shape.size();

	auto ptr    = device->malloc(size);
	auto refPtr = (size_t*) device->malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, size, device);

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> Executor<T>::createTensor(Shape &shape) {
	if (DeviceType::CPU == device->type) {
		return createTensorCPU(shape);
	} else {
#ifdef HAVE_CUDA
		return createTensorGPU(shape);
#else
		DEEP8_RUNTIME_ERROR("without a GPU");
#endif
	}
}

template <typename T>
int64_t Executor<T>::generateNodeId() {
	return nodeId++;
}

template <typename T>
Variable<T>* Executor<T>::createVariableWithFunction(FunctionBase *func) {
	if (func->isShared()) {
		auto variable = new Variable<T>(func, func->outputShape);

		return variable;
	} else {
		auto value    = this->createTensor(func->outputShape);
		auto gradient = this->createTensor(func->outputShape);

		auto variable = new Variable<T>(func, value, gradient);

		return variable;
	}
}

template <typename T>
Parameter<T>* Executor<T>::addParameter(std::vector<size_t> list, bool updateGradient, void *ptr) {
	Shape shape(list);

	return addParameter(shape, updateGradient, ptr);
}

template <typename T>
Parameter<T>* Executor<T>::addParameter(size_t batch, std::vector<size_t> list, bool updateGradient, void *ptr) {
	Shape shape(batch, list);

	return addParameter(shape, updateGradient, ptr);
}

template <typename T>
Parameter<T>* Executor<T>::addParameter(Shape &shape, bool updateGradient, void *ptr) {
	Parameter<T> *parameter = nullptr;

	if (updateGradient) {
		auto value    = createTensor(shape);
		auto gradient = createTensor(shape);

		parameter = new Parameter<T>(value, gradient);
	} else {
		auto value = createTensor(shape);

		parameter = new Parameter<T>(value);
	}

	/**set a id*/
	parameter->id = generateNodeId();

	nodeCollection.insert(parameter);
	parameterCollection.insert(parameter);

	if (nullptr != ptr) {
		parameter->feed(ptr);
	} else {
		TensorInit<T>().gaussian(parameter->value, 0.0, 0.1);
	}

	return parameter;
}

template <typename T>
Node* Executor<T>::addFunction(FunctionBase *function) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

template <typename T>
void Executor<T>::forward(Expression<T> &e) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

template <typename T>
void Executor<T>::backward(Expression<T> &e) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

template <typename T>
void Executor<T>::forward(Node *last) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

template <typename T>
void Executor<T>::backward(Node *last) {
	DEEP8_RUNTIME_ERROR("Can not call this function from Executor");
}

DEEP8_DECLARATION_INSTANCE(Executor)

}