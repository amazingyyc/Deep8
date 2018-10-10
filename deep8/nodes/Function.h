#ifndef DEEP8_FUNCTION_H
#define DEEP8_FUNCTION_H

#include "Node.h"
#include "Variable.h"

namespace Deep8 {

class FunctionBase: public Node {
public:
    /**
     * @brief the output pointer
     * the Function Node must have a output
     */
    Node *output;

	/**
	 * if a Function shared is true means the output Variable shared the memory with the input Variable
	 * default is false
	 */
	bool shared;


protected:
    explicit FunctionBase(): Node(), output(nullptr), shared(false) {
        this->type = NodeType::Function;
    }

    explicit FunctionBase(std::vector<Node*> &inputs): Node(inputs), output(nullptr), shared(false) {
        this->type = NodeType::Function;
    }

    virtual void check() {
        for (auto item : inputs) {
            DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be Variable type");
        }
    }

public:
    void forward() override {}
    void backward() override {}
};

template <typename T>
class Function: public FunctionBase {
protected:
    explicit Function(): FunctionBase() {
    }

    explicit Function(std::vector<Node*> &inputs): FunctionBase(inputs) {
    }

protected:
	virtual void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output);
	virtual void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output);
	virtual void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient);
	virtual void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient);

public:
	void forward() override;
	void backward() override;
};

}



#endif //DEEP8_FUNCTION_H
