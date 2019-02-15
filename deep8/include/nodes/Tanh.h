#ifndef DEEP8_TANH_H
#define DEEP8_TANH_H

#include "Function.h"

namespace Deep8 {

class Tanh: public Function {
public:
    explicit Tanh(std::vector<Node *> &inputs);

	void check() override;

	/**if support the auto batch*/
	int supportAutoBatch() override;

	 /**auto batch code*/
	 size_t autoBatchCode() override;

	 /**
	  * return the inputs[index]'s shape if it is be batched together.
	  * the shapes is the inputs[index]'s shape that will be batched.
	  */
	 Shape autoBatchShape(size_t index, std::vector<Shape> &shapes) override;

	 /**
	  * return the inputs's index that can be auto batched
	  */
	 std::vector<size_t> autoBatchIndexes() override;

	 /**
	  * clone current node for auto batch
	  */
	 Node* autoBatchClone(std::vector<Node*> &) override; 
	 
protected:
	void forward(const std::vector<const Tensor*> &inputs, Tensor *output) override;
	void backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) override;

};

}

#endif //DEEP8_TANH_H
