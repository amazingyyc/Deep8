#ifndef DEEP8_MATRIXMULTIPLY_H
#define DEEP8_MATRIXMULTIPLY_H

#include "Function.h"

namespace Deep8 {

template <typename T>
class MatrixMultiply: public Function<T> {
public:
    explicit MatrixMultiply(std::vector<Node *> &inputs);

    /**
     * @brief for the MatrixMultiply the input size must be 2, and must be Matrix
     * @param inputs the inputs Node must be
     * @return the output Shape
     */
     void check() override;

	 /**if support the auto batch*/
	 bool supportAutoBatch() override;

	 /**auto batch code*/
	 size_t autoBatchCode() override;

	 /**
	  * return the inputs[index]'s shape if it is be batched together.
	  * the shapes is the inputs[index]'s shape that will be batched.
	  */
	 Shape autoBatchShape(size_t index, std::vector<Shape> &shapes) override;

	 /**
	  * calculate the outputShape, inputShapes is the Shape of inputs
	  */
	 Shape calcOutputShape(std::vector<Shape> &inputShapes) override;

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
    void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
	void forwardGPUImpl(Device *device, const T *A, const Shape &aShape, const T *B, const Shape &bShape, T *C, const Shape &cShape);
	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardGPUImpl0(Device* device, T *aGrad, const Shape &aShape, const T *B, const Shape &bShape, const T *cGrad, const Shape &cShape);
	void backwardGPUImpl1(Device* device, const T *A, const Shape &aShape, T *bGrad, const Shape &bShape, const T *cGrad, const Shape &cShape);
	void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;
#endif
};


}

#endif //DEEP8_MATRIXMULTIPLY_H
