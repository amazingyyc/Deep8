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
    explicit FunctionBase();
    explicit FunctionBase(std::vector<Node*> &inputs);

    virtual void check();

public:
    void forward() override;
    void backward() override;
};

template <typename T>
class Function: public FunctionBase {
protected:
    explicit Function();
    explicit Function(std::vector<Node*> &inputs);

protected:
	virtual void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output);
	virtual void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient);

#ifdef HAVE_CUDA
    virtual void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output);
	virtual void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient);
#endif

public:
	void forward() override;
	void backward() override;
};

}



#endif //DEEP8_FUNCTION_H
