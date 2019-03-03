#ifndef DEEP8_MATH_EXP_H
#define DEEP8_MATH_EXP_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = exp(x)
 */
void Exp(const Tensor &x, Tensor &y);

void ExpCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void ExpGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of Exp
 */
void ExpGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void ExpGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void ExpGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif