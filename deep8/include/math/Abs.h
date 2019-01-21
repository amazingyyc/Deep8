#ifndef DEEP8_MATH_ABS_H
#define DEEP8_MATH_ABS_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = abs(x)
 */
void Abs(const Tensor &x, Tensor &y);

void AbsCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void AbsGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of Abs
 */
void AbsGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void AbsGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void AbsGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif