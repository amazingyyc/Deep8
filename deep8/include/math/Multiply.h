#ifndef DEEP8_MULTIPLY_MINUS_H
#define DEEP8_MULTIPLY_MINUS_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * z = x * y 
 */
void Multiply(const Tensor &x, const Tensor &y, Tensor &z);

void MultiplyCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void MultiplyGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

/**
 * calculate grad(x) for z = x * y
 */
void MultiplyGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void MultiplyGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void MultiplyGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

/**
 * calculate grad(y) for z = x * y
 */
void MultiplyGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void MultiplyGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void MultiplyGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif