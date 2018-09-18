#ifndef DEEP8_FUNCTION_H
#define DEEP8_FUNCTION_H

namespace Deep8 {

class FunctionBase: public Node {
public:
    /**
     * @brief the output pointer
     * the Function Node must have a output
     */
    Node *output;

protected:
    explicit FunctionBase(): Node() {
        this->type = NodeType::Function;
    }

    explicit FunctionBase(std::vector<Node*> &inputs): Node(inputs), output(nullptr) {
        this->type = NodeType::Function;
    }

    virtual void check() {
        for (auto item : inputs) {
            DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be Variable type");
        }
    }

public:
    void forward() override {
    }

    void backward() override {
    }
};

template <typename T>
class Function: public FunctionBase {
protected:
    explicit Function(): FunctionBase() {
    }

    explicit Function(std::vector<Node*> &inputs): FunctionBase(inputs) {
    }

protected:
    virtual void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
        DEEP8_RUNTIME_ERROR("can not call this forwardCPU by Function class");
    }

    virtual void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
        DEEP8_RUNTIME_ERROR("can not call this forwardGPU by Function class");
    }

    virtual void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {
        DEEP8_RUNTIME_ERROR("can not call this backwardCPU by Function class");
    }

    virtual void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {
        DEEP8_RUNTIME_ERROR("can not call this backwardGPU by Function class");
    }

public:
    void forward() override {
        DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->output->type, "the output must be a Variable type");

        auto outputVariable = static_cast<Variable<T>*>(this->output);

        DEEP8_ARGUMENT_CHECK(this->outputShape == outputVariable->value.shape, "the output shape is error");

		auto outputValue = &(outputVariable->value);
		auto deviceType  = outputVariable->deviceType();

        std::vector<const Tensor<T>*> inputValues;

        for (auto item : inputs) {
            auto inputVariable = static_cast<Variable<T>*>(item);

            DEEP8_ARGUMENT_CHECK(deviceType == inputVariable->deviceType(), "the device of input must be same with output");

            inputValues.emplace_back(&(inputVariable->value));
        }

        if (DeviceType::CPU == deviceType) {
            forwardCPU(inputValues, outputValue);
        } else {
            forwardGPU(inputValues, outputValue);
        }
    }

    void backward() override {
        DEEP8_ARGUMENT_CHECK(NodeType::Variable == output->type, "the output must be Variable type");

        auto outputVariable = static_cast<Variable<T>*>(output);

        auto outputValue    = &(outputVariable->value);
        auto outputGradient = &(outputVariable->gradient);

        auto deviceType =  outputVariable->deviceType();

        std::vector<const Tensor<T>*> inputValues;

        for (auto item : inputs) {
            auto inputVariable = static_cast<Variable<T>*>(item);

            DEEP8_ARGUMENT_CHECK(deviceType == inputVariable->deviceType(), "the device of the input and output must have same device type");

            inputValues.emplace_back(&(inputVariable->value));
        }

        if (DeviceType::CPU == deviceType) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto inputVariable = static_cast<Variable<T>*>(inputs[i]);

                if (inputVariable->updateGradient) {
                    backwardCPU(inputValues, outputValue, outputGradient, i, &(inputVariable->gradient));
                }
            }
        } else {
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto inputVariable = static_cast<Variable<T>*>(inputs[i]);

                if (inputVariable->updateGradient) {
                    backwardGPU(inputValues, outputValue, outputGradient, i, &(inputVariable->gradient));
                }
            }
        }
    }
};

}



#endif //DEEP8_FUNCTION_H
