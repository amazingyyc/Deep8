#ifndef DEEP8_ADD_H
#define DEEP8_ADD_H

#include "Function.h"

namespace Deep8 {

/**
 * Z = X + Y
 */

template <typename T>
class Add: public Function<T> {
public:
    explicit Add(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
    }

    void check() override;

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    template <int diffCount>
    void backwardCPUImpl(Eigen::ThreadPoolDevice *device, const Tensor<T> *outputGradient, Tensor<T> *iGradient);

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;
	
#ifdef HAVE_CUDA
    void forwardGPUImpl(const T *x, const int *xdims, const int *xstrides,
						const T *y, const int *ydims, const int *ystrides,
							  T *z, const int *zdims, const int *zstrides, const int N);

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardGPUImpl(T *inGrad, const int *inShape, const int *inDims, const T *outGrad, const int *outShape, const int *outDims, const int N);

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#endif
};

}

#endif //DEEP8_ADD_H
