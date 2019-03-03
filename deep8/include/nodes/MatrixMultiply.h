#ifndef DEEP8_MATRIXMULTIPLY_H
#define DEEP8_MATRIXMULTIPLY_H

#include "Function.h"

namespace Deep8 {

class MatrixMultiply: public Function {
public:
    explicit MatrixMultiply(std::vector<Node *> &inputs);

    /**
     * @brief for the MatrixMultiply the input size must be 2, and must be Matrix
     * @param inputs the inputs Node must be
     * @return the output Shape
     */
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

#endif //DEEP8_MATRIXMULTIPLY_H
