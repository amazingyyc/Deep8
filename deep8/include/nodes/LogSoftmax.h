#ifndef DEEP8_LOGSOFTMAX_H
#define DEEP8_LOGSOFTMAX_H

#include "Function.h"

namespace Deep8 {

class LogSoftmax : public Function {
public:
    int axis;

    explicit LogSoftmax(std::vector<Node *> &inputs, int axis = -1);

    void check() override;

protected:
    void forward(const std::vector<const Tensor*> &inputs, Tensor *output) override;
	void backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) override;
};

}

#endif //PROJECT_LOGSOFTMAX_H
