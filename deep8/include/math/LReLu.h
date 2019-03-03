#ifndef DEEP8_MATH_LRELU_H
#define DEEP8_MATH_LRELU_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = lrelu(x)
 */
void LReLu(const Tensor &x, const float a, Tensor &y);

void LReLuCPU(const Tensor &x, const float a, Tensor &y);

#ifdef HAVE_CUDA
void LReLuGPU(const Tensor &x, const float a, Tensor &y);
#endif

/**
 * calculate the grad(x) of lrelu
 */
void LReLuGrad(const Tensor &x, Tensor &dx, const float a, const Tensor &y, const Tensor &dy);

void LReLuGradCPU(const Tensor &x, Tensor &dx, const float a, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void LReLuGradGPU(const Tensor &x, Tensor &dx, const float a, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif