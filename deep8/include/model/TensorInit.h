#ifndef DEEP8_TENSORINIT_H
#define DEEP8_TENSORINIT_H

#include "Basic.h"
#include "Tensor.h"

namespace Deep8 {

/**
 * init the tensor
 */
template <typename T>
class TensorInit {
private:
	void constantCPU(Tensor<T> &tensor, T v);
	void uniformCPU(Tensor<T> &tensor, T left, T right);
	void gaussianCPU(Tensor<T> &tensor, T mean, T stddev);
	void positiveUnitballCPU(Tensor<T> &tensor);

#ifdef HAVE_CUDA
	void constantGPU(Tensor<T> &tensor, T v);
	void uniformGPU(Tensor<T> &tensor);
	void gaussianGPU(Tensor<T> &tensor, T mean, T stddev);
	void positiveUnitballGPU(Tensor<T> &tensor);
#endif

public:
    /**set tensor to constant*/
    void constant(Tensor<T> &tensor, T v);
    void uniform(Tensor<T> &tensor, T left = 0.0, T right = 1.0);
    void gaussian(Tensor<T> &tensor, T mean = 0.0, T stddev = 1.0);
    void positiveUnitball(Tensor<T> &tensor);
};

}

#endif //DEEP8_TENSORINIT_H
