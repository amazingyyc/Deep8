#ifndef DEEP8_MATH_TANH_H
#define DEEP8_MATH_TANH_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = tanh(x)
 */
void Tanh(const Tensor &x, Tensor &y);

void TanhCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void TanhGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of tanh
 */
void TanhGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void TanhGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void TanhGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}


#endif