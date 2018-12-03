#ifndef DEEP8_MULTIPLY_H
#define DEEP8_MULTIPLY_H

#include "Function.h"

namespace Deep8 {

/**
 * @brief this is a element-wise multiply it will BroadCast the input
 */

template <typename T>
class Multiply: public Function<T> {
public:
    explicit Multiply(std::vector<Node *> &inputs);

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    template <int diffCount>
	void backwardCPUImpl(Eigen::ThreadPoolDevice *device,
						const Tensor<T> *otherValue,
						const Tensor<T> *outputGradient,
						Tensor<T> *iGradient);

	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	
    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override;
#endif

};


}



#endif //DEEP8_WISEMULTIPLY_H
