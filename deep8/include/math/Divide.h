#ifndef DEEP8_MATH_DIVIDE_H
#define DEEP8_MATH_DIVIDE_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * z = x / y 
 */
void Divide(const Tensor &x, const Tensor &y, Tensor &z);

void DivideCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void DivideGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

/**
 * calculate grad(x) for z = x / y
 */
void DivideGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void DivideGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void DivideGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

/**
 * calculate grad(y) for z = x / y
 */
void DivideGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void DivideGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void DivideGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif