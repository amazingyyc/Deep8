#ifndef DEEP8_EXECUTOR_H
#define DEEP8_EXECUTOR_H

#include <unordered_set>

#include "Device.h"
#include "Node.h"
#include "Variable.h"
#include "Parameter.h"
#include "InputParameter.h"
#include "ConstantParameter.h"
#include "Function.h"
#include "Trainer.h"

namespace Deep8 {

template <class T> class Executor;

template <typename T>
class Expression {
public:
	/**
	 * @brief the compute executor
	 */
	Executor<T> *executor;
	
	/**
	    * @brief the Node pointer that contacted to this Expression
	    */
	Node *node;
	
	explicit Expression(): executor(nullptr), node(nullptr) {
	}
	
	explicit Expression(Executor<T> *exe, Node *n): executor(exe), node(n) {
	}
};

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

    explicit Executor(TrainerType trainerType = TrainerType::SGD, DeviceType deviceType = DeviceType::CPU): 
		nodeCollection(), parameterCollection(), nonParameterCollection() {
        if (deviceType == DeviceType::CPU) {
            device = new CPUDevice();
		} else {
#ifdef HAVE_CUDA
			device = new GPUDevice();
#else
			DEEP8_RUNTIME_ERROR("not find a GPU");
#endif
		}

		switch (trainerType) {
		case Deep8::TrainerType::SGD:
			trainer = new SGDTrainer<T>();
			break;
		case Deep8::TrainerType::Adagrad:
			trainer = new AdagradTrainer<T>();
			break;
		case Deep8::TrainerType::Adam:
			trainer = new AdamTrainer<T>();
			break;
		case Deep8::TrainerType::RMSProp:
			trainer = new RMSPropTrainer<T>();
			break;
		case Deep8::TrainerType::Momentum:
			trainer = new MomentumTrainer<T>();
			break;
		default:
			DEEP8_RUNTIME_ERROR("the tainer type is error")
			break;
		}
    }

protected:

	/**
	 * the sub class to implements some special operater
	 */
	virtual void afterAddFunctionNode(Node *function, Node *variable) {
	}

	Tensor<T> createTensorWithShape(Shape &shape) {
		auto ptr = device->malloc(sizeof(T) * shape.size());

		return Tensor<T>(ptr, shape, device);
	}

	Variable<T>* createVariableByFunction(Node *function) {
		if (function->shared) {
			Tensor<T> value(nullptr, function->outputShape, device);
			Tensor<T> gradient(nullptr, function->outputShape, device);

			auto variable = new Variable<T>(function, value, gradient);
			variable->shared = true;

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

	template <typename FunctionType>
	Node *addFunction(std::vector<Node*> inputs) {
		auto function = new FunctionType(inputs);
		auto variable = createVariableByFunction(function);

		function->output = variable;

		nodeCollection.insert(function);
		nodeCollection.insert(variable);

		nonParameterCollection.insert(function);
		nonParameterCollection.insert(variable);

		afterAddFunctionNode(function, variable);

		return variable;
	}

	template <typename FunctionType, typename... Args>
	Node *addFunction(std::vector<Node*> inputs, Args&&... arguments) {
		auto function = new FunctionType(inputs, std::forward<Args>(arguments)...);
		auto variable = createVariableByFunction(function);

		function->output = variable;

		nodeCollection.insert(function);
		nodeCollection.insert(variable);

		nonParameterCollection.insert(function);
		nonParameterCollection.insert(variable);

		afterAddFunctionNode(function, variable);

		return variable;
	}

	virtual void forward(Expression<T> &e) = 0;
	virtual void backward(Expression<T> &e) = 0;
	virtual void forward(Node *last) = 0;
	virtual void backward(Node *last) = 0;
};

}

#endif //DEEP8_EXECUTER_H

















