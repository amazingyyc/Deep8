#ifndef DEEP8_MATH_RELU_H
#define DEEP8_MATH_RELU_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = relu(x)
 */
void ReLu(const Tensor &x, Tensor &y);

void ReLuCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void ReLuGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of relu
 */
void ReLuGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void ReLuGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void ReLuGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif