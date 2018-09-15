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
#include "TensorInit.h"
#include "Abs.h"
#include "Add.h"
#include "AddScalar.h"
#include "AvgPooling2d.h"
#include "Conv2d.h"
#include "DeConv2d.h"
#include "Divide.h"
#include "DivideScalar.h"
#include "Exp.h"
#include "L1Norm.h"
#include "L2Norm.h"
#include "Linear.h"
#include "Log.h"
#include "LReLu.h"
#include "MatrixMultiply.h"
#include "MaxPooling2d.h"
#include "Minus.h"
#include "MinusScalar.h"
#include "Multiply.h"
#include "MultiplyScalar.h"
#include "Pow.h"
#include "ReLu.h"
#include "ReShape.h"
#include "ScalarDivide.h"
#include "ScalarMinus.h"
#include "Sigmoid.h"
#include "Softmax.h"
#include "Square.h"
#include "SumElements.h"
#include "TanH.h"

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

	/**one operand function*/
	Expression<T> abs() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Abs<T>(inputs)));
	}

	Expression<T> exp() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Exp<T>(inputs)));
	}

	Expression<T> l1Norm() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new L1Norm<T>(inputs)));
	}

	Expression<T> l2Norm() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new L2Norm<T>(inputs)));
	}

	Expression<T> linear(T a, T b) {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Linear<T>(inputs, a, b)));
	}

	Expression<T> log() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Log<T>(inputs)));
	}

	Expression<T> lReLu(T a) {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new LReLu<T>(inputs, a)));
	}

	Expression<T> reLu() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new ReLu<T>(inputs)));
	}

	Expression<T> reShape(Shape &shape) {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new ReShape<T>(inputs, shape)));
	}

	Expression<T> reShape(std::initializer_list<size_t> list) {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new ReShape<T>(inputs, list)));
	}

	Expression<T> sigmoid() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Sigmoid<T>(inputs)));
	}

	Expression<T> softmax() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Softmax<T>(inputs)));
	}

	Expression<T> square() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Square<T>(inputs)));
	}

	Expression<T> sumElements() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new SumElements<T>(inputs)));
	}

	Expression<T> tanH() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new TanH<T>(inputs)));
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

















