#ifndef DEEP8_EXECUTOR_H
#define DEEP8_EXECUTOR_H

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

    explicit Executor(Trainer<T> *tr, DeviceType deviceType = DeviceType::CPU):
			trainer(tr), nodeCollection(), parameterCollection(), nonParameterCollection() {
        if (deviceType == DeviceType::CPU) {
            device = new CPUDevice();
		} else {
#ifdef HAVE_CUDA
			device = new GPUDevice();
#else
			DEEP8_RUNTIME_ERROR("not find a GPU");
#endif
		}
    }

protected:

	template <typename real>
	Tensor<real> createTensorWithShapeCPU(Shape &shape) {
		size_t size = sizeof(real) * shape.size();

		auto ptr    = device->malloc(size);
		auto refPtr = (size_t*)device->malloc(sizeof(size_t));

		TensorStorage storage(ptr, refPtr, size, device);

		return Tensor<real>(storage, 0, shape);
	}


#ifdef HAVE_HALF
	template <>
	Tensor<half> createTensorWithShapeCPU(Shape &shape) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif

#ifdef HAVE_CUDA

	template <typename real>
	Tensor<real> createTensorWithShapeGPU(Shape &shape) {
		size_t size = sizeof(real) * shape.size();

		auto gpuDevice = (GPUDevice*)device;

		auto ptr = gpuDevice->malloc(size);
		auto refPtr = (size_t*) gpuDevice->mallocCPU(sizeof(size_t));

		TensorStorage storage(ptr, refPtr, size, device);

		return Tensor<real>(storage, 0, shape);
	}

#endif

	Tensor<T> createTensorWithShape(Shape &shape) {
		if (DeviceType::CPU == device->type) {
			return createTensorWithShapeCPU<T>(shape);
		} else {
#ifdef HAVE_CUDA
			return createTensorWithShapeGPU<T>(shape);
#else
			DEEP8_RUNTIME_ERROR("without a GPU");
#endif
		}
	}

	Variable<T>* createVariableByFunction(FunctionBase *function) {
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

public:
	virtual ~Executor() {
		for (auto node : nodeCollection) {
			delete node;
		}

		nodeCollection.clear();
		parameterCollection.clear();
		nonParameterCollection.clear();

		delete trainer;
		delete device;
	}

	Parameter<T> *addParameter(std::initializer_list<size_t> list) {
		Shape shape(list);

		return addParameter(shape);
	}

	Parameter<T> *addParameter(Shape &shape) {
		auto value    = createTensorWithShape(shape);
		auto gradient = createTensorWithShape(shape);

		auto parameter = new Parameter<T>(value, gradient);

		nodeCollection.insert(parameter);
		parameterCollection.insert(parameter);

		/**init the parameter*/
		TensorInit::gaussian(value);

		return parameter;
	}

	InputParameter<T> *addInputParameter(std::initializer_list<size_t> list, T *ptr = nullptr) {
		Shape shape(list);

		return addInputParameter(shape, ptr);
	}

	InputParameter<T> *addInputParameter(Shape &shape, T *ptr = nullptr) {
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

	virtual Node *addFunction(FunctionBase *function) {
		auto variable = createVariableByFunction(function);

		function->output = variable;
		
		nodeCollection.insert(function);
		nodeCollection.insert(variable);

		nonParameterCollection.insert(function);
		nonParameterCollection.insert(variable);

		return variable;
	}

	virtual void forward(Expression<T> &e) = 0;
	virtual void backward(Expression<T> &e) = 0;
	virtual void forward(Node *last) = 0;
	virtual void backward(Node *last) = 0;
};

}

#endif //DEEP8_EXECUTER_H

















