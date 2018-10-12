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

	void constantGPU(Tensor<T> &tensor, T v);
	void uniformGPU(Tensor<T> &tensor);
	void gaussianGPU(Tensor<T> &tensor, T mean, T stddev);
	void positiveUnitballGPU(Tensor<T> &tensor);

public:
    /**set tensor to constant*/
    void constant(Tensor<T> &tensor, T v) {
        if (DeviceType::CPU == tensor.device()->type) {
            constantCPU(tensor, v);
        } else {
            constantGPU(tensor, v);
        }
    }

    void uniform(Tensor<T> &tensor, T left = 0.0, T right = 1.0) {
        if (DeviceType::CPU == tensor.device()->type) {
            uniformCPU(tensor, left, right);
        } else {
            uniformGPU(tensor);
        }
    }

    void gaussian(Tensor<T> &tensor, T mean = 0.0, T stddev = 1.0) {
        if (DeviceType::CPU == tensor.device()->type) {
            gaussianCPU(tensor, mean, stddev);
        } else {
            gaussianGPU(tensor, mean, stddev);
        }
    }

    void positiveUnitball(Tensor<T> &tensor) {
        if (DeviceType::CPU == tensor.device()->type) {
            positiveUnitballCPU(tensor);
        } else {
            positiveUnitballGPU(tensor);
        }
    }
};

}

#endif //DEEP8_TENSORINIT_H
