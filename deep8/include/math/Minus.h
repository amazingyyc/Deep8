#ifndef DEEP8_MATH_MINUS_H
#define DEEP8_MATH_MINUS_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * z = x - y 
 */
void Minus(const Tensor &x, const Tensor &y, Tensor &z);

void MinusCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void MinusGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

/**
 * calculate grad(x) for z = x + y
 */
void MinusGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void MinusGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void MinusGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

/**
 * calculate grad(y) for z = x + y
 */
void MinusGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void MinusGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void MinusGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif
