#ifndef DEEP8_MATH_LINEAR_H
#define DEEP8_MATH_LINEAR_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = Linear(x)
 */
void Linear(const Tensor &x, const float a, const float b, Tensor &y);

void LinearCPU(const Tensor &x, const float a, const float b, Tensor &y);

#ifdef HAVE_CUDA
void LinearGPU(const Tensor &x, const float a, const float b, Tensor &y);
#endif

/**
 * calculate the grad(x) of Linear
 */
void LinearGrad(const Tensor &x, Tensor &dx, const float a, const float b, const Tensor &y, const Tensor &dy);

void LinearGradCPU(const Tensor &x, Tensor &dx, const float a, const float b, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void LinearGradGPU(const Tensor &x, Tensor &dx, const float a, const float b, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif