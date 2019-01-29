#ifndef DEEP8_MATH_ADD_H
#define DEEP8_MATH_ADD_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {


namespace Math {

/**
 * z = x + y
 * support broadcast
 */
void Add(const Tensor &x, const Tensor &y, Tensor &z);

void AddCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void AddGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

/**
 * calculate grad(x) for z = x + y
 */
void AddGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void AddGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void AddGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

/**
 * calculate grad(y) for z = x + y
 */
void AddGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void AddGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void AddGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif