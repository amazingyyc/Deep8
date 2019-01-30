#ifndef DEEP8_MATH_SIGMOID_H
#define DEEP8_MATH_SIGMOID_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = sigmoid(x)
 */
void Sigmoid(const Tensor &x, Tensor &y);

void SigmoidCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void SigmoidGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of Sigmoid
 */
void SigmoidGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void SigmoidGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void SigmoidGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif


}
}


#endif