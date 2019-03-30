#ifndef DEEP8_RESHAPE_H
#define DEEP8_RESHAPE_H

#include "Function.h"

namespace Deep8 {

class ReShape: public Function {
public:
	Shape reShape;

    explicit ReShape(std::vector<Node *> &inputs, Shape &shape);
    explicit ReShape(std::vector<Node *> &inputs, std::vector<size_t> &shape);
	
	bool isShared() override;

	Shape checkShape(std::vector<Shape> &inputShapes) override;

	ElementType checkElementType(std::vector<ElementType> &inputTypes) override;

	void forward(const std::vector<const Tensor*> &inputs, Tensor *output) override;
	void backward(const std::vector<const Tensor*> &inputs, const Tensor *output, const Tensor *outputGradient, size_t index, Tensor *iGradient) override;

	void forward() override;
	void backward() override;
};

}

#endif //DEEP8_RESHAPE_H
