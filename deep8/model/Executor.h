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

class Expression;

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
    std::unordered_set<VariableBase*> parameterCollection;

    /**
     * store the non-Parameter Nodes
     * include the Variable and Function
     */
    std::unordered_set<Node*> nonParameterCollection;

    /**trainer to train the parameter*/
    Trainer *trainer;

protected:
    explicit Executor(Trainer *t, DeviceType deviceType = DeviceType::CPU):
            trainer(t), nodeCollection(), parameterCollection(), nonParameterCollection() {
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

public:
    virtual ~Executor() {
        for (auto node : nodeCollection) {
            delete node;
        }

        nodeCollection.clear();
        parameterCollection.clear();
        nonParameterCollection.clear();

        delete device;
    }

protected:
    /**
     * the sub class to implements some special operater
     */
    virtual void afterAddFunctionNode(Node *function, Node *variable) {
    }

    template <typename T>
    Tensor<T> createTensorWithShape(Shape &shape) {
        auto ptr = device->malloc(sizeof(T) * shape.size());

        return Tensor<T>(ptr, shape, device);
    }

    template <typename T>
    Variable<T>* createVariableByFunction(Node *function) {
        if (function->shared) {
            Tensor<T> value(nullptr, function->outputShape, device);
            Tensor<T> gradient(nullptr, function->outputShape, device);

            auto variable = new Variable<T>(function, value, gradient);
            variable->shared = true;

            return variable;
        } else {
            auto value = createTensorWithShape<T>(function->outputShape);
            auto gradient = createTensorWithShape<T>(function->outputShape);

            auto variable = new Variable<T>(function, value, gradient);

            return variable;
        }
    }

public:
    /**
     * add a Parameter
     */
    template <typename T>
    Parameter<T> *addParameter(std::initializer_list<size_t> list) {
        Shape shape(list);

        return this->addParameter<T>(shape);
    }

    template <typename T>
    Parameter<T> *addParameter(Shape &shape) {
        auto value    = createTensorWithShape<T>(shape);
        auto gradient = createTensorWithShape<T>(shape);

        auto parameter = new Parameter<T>(value, gradient);

        nodeCollection.insert(parameter);
        parameterCollection.insert(parameter);

        return parameter;
    }

    template <typename T>
    InputParameter<T> *addInputParameter(std::initializer_list<size_t> list, T *ptr = nullptr) {
        Shape shape(list);

        return this->addInputParameter<T>(shape, ptr);
    }

    template <typename T>
    InputParameter<T> *addInputParameter(Shape &shape, T *ptr = nullptr) {
        auto value = createTensorWithShape<T>(shape);

        auto inputParameter = new InputParameter<T>(value);

        /**feed data*/
        if (nullptr != ptr) {
            inputParameter->feed(ptr);
        }

        nodeCollection.insert(inputParameter);
        parameterCollection.insert(inputParameter);

        return inputParameter;
    }

    template <typename T, typename FunctionType>
    Node *addFunction(std::vector<Node*> inputs) {
        auto function = new FunctionType(inputs);
        auto variable = createVariableByFunction<T>(function);

        function->output = variable;

        nodeCollection.insert(function);
        nodeCollection.insert(variable);

        nonParameterCollection.insert(function);
        nonParameterCollection.insert(variable);

        afterAddFunctionNode(function, variable);

        return variable;
    }

    template <typename T, typename FunctionType, typename... Args>
    Node *addFunction(std::vector<Node*> inputs, Args&&... arguments) {
        auto function = new FunctionType(inputs, std::forward<Args>(arguments)...);
        auto variable = createVariableByFunction<T>(function);

        function->output = variable;

        nodeCollection.insert(function);
        nodeCollection.insert(variable);

        nonParameterCollection.insert(function);
        nonParameterCollection.insert(variable);

        afterAddFunctionNode(function, variable);

        return variable;
    }

    virtual void forward(Node *last)  = 0;
    virtual void backward(Node *last) = 0;

    virtual void forward(Expression &e)  = 0;
    virtual void backward(Expression &e) = 0;
};

}

#endif //DEEP8_EXECUTER_H

















