#ifndef DEEP8_MATH_SQRT_H
#define DEEP8_MATH_SQRT_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = sqrt(x)
 */
void Sqrt(const Tensor &x, Tensor &y);

void SqrtCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void SqrtGPU(const Tensor &x, Tensor &y);
#endif

void SqrtGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void SqrtGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void SqrtGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif