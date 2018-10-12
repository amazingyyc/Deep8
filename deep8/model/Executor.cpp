#include "Executor.h"
#include "TensorInit.h"

namespace Deep8 {

template <typename T>
Executor<T>::Executor(Trainer<T> *tr, DeviceType deviceType = DeviceType::CPU) 
	:trainer(tr), nodeCollection(), parameterCollection(), nonParameterCollection() {
	if (deviceType == DeviceType::CPU) {
		initDeviceCPU();
	} else {
		initDeviceGPU();
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
Tensor<T> Executor<T>::createTensorWithShapeCPU(Shape &shape) {
	size_t size = sizeof(T) * shape.size();

	auto ptr    = device->malloc(size);
	auto refPtr = (size_t*) device->malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, size, device);

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> Executor<T>::createTensorWithShapeGPU(Shape &shape) {
	size_t size = sizeof(T) * shape.size();

	auto ptr = device->malloc(size);
	auto refPtr = (size_t*)device->mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, size, device);

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> Executor<T>::createTensorWithShape(Shape &shape) {
	if (DeviceType::CPU == device->type) {
		return createTensorWithShapeCPU(shape);
	} else {
#ifdef HAVE_CUDA
		return createTensorWithShapeGPU(shape);
#else
		DEEP8_RUNTIME_ERROR("without a GPU");
#endif
	}
}

template <typename T>
Variable<T>* Executor<T>::createVariableByFunction(FunctionBase *function) {
	if (function->shared) {
		auto variable = new Variable<T>(function, function->outputShape);

		return variable;
	} else {
		auto value    = createTensorWithShape(function->outputShape);
		auto gradient = createTensorWithShape(function->outputShape);

		auto variable = new Variable<T>(function, value, gradient);

		return variable;
	}
}

template <typename T>
Parameter<T>* Executor<T>::addParameter(std::initializer_list<size_t> list) {
	Shape shape(list);

	return addParameter(shape);
}

template <typename T>
Parameter<T>* Executor<T>::addParameter(Shape &shape) {
	auto value    = createTensorWithShape(shape);
	auto gradient = createTensorWithShape(shape);

	auto parameter = new Parameter<T>(value, gradient);

	nodeCollection.insert(parameter);
	parameterCollection.insert(parameter);

	/**init the parameter*/
	TensorInit<T>().gaussian(value);

	return parameter;
}

template <typename T>
InputParameter<T>* Executor<T>::addInputParameter(std::initializer_list<size_t> list, T *ptr = nullptr) {
	Shape shape(list);

	return addInputParameter(shape, ptr);
}

template <typename T>
InputParameter<T>* Executor<T>::addInputParameter(Shape &shape, T *ptr = nullptr) {
	auto value = createTensorWithShape(shape);

	auto inputParameter = new InputParameter<T>(value);

	/**feed data*/
	if (nullptr != ptr) {
		inputParameter->feed(ptr);
	}

	nodeCollection.insert(inputParameter);
	parameterCollection.insert(inputParameter);

	return inputParameter;
}

template <typename T>
Node* Executor<T>::addFunction(FunctionBase *function) {
	auto variable = createVariableByFunction(function);

	function->output = variable;

	nodeCollection.insert(function);
	nodeCollection.insert(variable);

	nonParameterCollection.insert(function);
	nonParameterCollection.insert(variable);

	return variable;
}

DEEP8_DECLARATION_INSTANCE(Executor)

}